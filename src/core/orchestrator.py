"""
Orchestrator — swing trading, market-calendar-driven schedule.

100% rule-based. No LLM anywhere in this pipeline — see
src/agents/thesis_builder.py and src/backtest/engine.py for the validated
breakout/pullback trend-following rules this runs on (out-of-sample profit
factor 1.60 across 140 real symbols, 5 years, real fees + leverage).

Architecture:
  SIGNAL PLANE     → ScreeningAgent (deterministic technical filter)
                   → thesis_builder (deterministic TradingThesis from the signal)
  EXECUTION PLANE  → RiskGate → ExecutionAgent (no LLM)
  REVIEW PLANE     → PositionReviewAgent (deterministic technical check, 1×/day)
  MAINTENANCE      → TrailingStopAgent (every 60 min, deterministic)

Schedule per region (5 min after market open):
  US:     CronTrigger(hour=9, min=35, tz="America/New_York")
  EU:     CronTrigger(hour=9, min=5,  tz="Europe/Berlin")
  ASIA:   CronTrigger(hour=9, min=5,  tz="Asia/Tokyo")
  CRYPTO: every 6 hours UTC

Daily:
  Position review:  07:00 UTC (before any equity market opens)
  Daily summary:    23:00 UTC
  Trailing stops:   every 60 min
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv

from src.agents import risk_gate as risk_gate_module
from src.agents.execution_agent import ExecutionAgent, size_position
from src.agents.notification_agent import NotificationAgent
from src.agents.position_review_agent import PositionReviewAgent
from src.agents.screening_agent import ScreeningAgent, ScreeningResult
from src.agents.thesis_builder import build_thesis
from src.agents.trailing_stop_agent import TrailingStopAgent
from src.config.universe import get_symbols, get_instrument_id as cache_instrument_id
from src.core.etoro_client import EtoroClient
from src.core.market_calendar import is_trading_day
from src.core.state import ProjectState
from src.mcp_clients.mcp_manager import MCPManager

load_dotenv()
logger = logging.getLogger(__name__)

_REGIONS = [r.strip().upper() for r in os.getenv("WATCH_REGIONS", "US,EU,ASIA,CRYPTO").split(",")]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Orchestrator:
    def __init__(self):
        # Load persisted state from disk (positions survive restarts).
        # get_portfolio() reconciliation runs at startup to sync with broker.
        self.state = ProjectState.load()

        self.client = EtoroClient()
        self.mcp_manager = MCPManager()

        self.notification_agent = NotificationAgent(self.state)
        self.screening_agent = ScreeningAgent(self.client)
        self.execution_agent = ExecutionAgent(self.client, self.state, self.notification_agent)
        self.trailing_stop_agent = TrailingStopAgent(self.client, self.state)
        self.position_review_agent = PositionReviewAgent(
            execution_agent=self.execution_agent,
            notification_agent=self.notification_agent,
            state=self.state,
        )

        self._scheduler = AsyncIOScheduler(timezone="UTC")
        self._instrument_cache: dict[str, str] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        async with self.client:
            logger.info("Starting MCP servers...")
            await self.mcp_manager.start()
            try:
                await self._validate_credentials()
                await self._reconcile_open_positions()  # P0-2: sync with broker
                await self._prefetch_instruments()
                self._setup_schedules()
                self._scheduler.start()
                mode = os.getenv("ETORO_MODE", "demo").upper()
                all_symbols = [s for r in _REGIONS for s in get_symbols(r)]
                logger.info(
                    "Orchestrator running — mode=%s regions=%s symbols≈%d positions=%d",
                    mode, _REGIONS, len(all_symbols), len(self.state.open_positions),
                )
                await self.notification_agent.send_startup(mode, all_symbols[:10])
                while True:
                    await asyncio.sleep(60)
            except (KeyboardInterrupt, asyncio.CancelledError):
                logger.info("Orchestrator shutting down...")
                self._scheduler.shutdown(wait=False)
            finally:
                await self.mcp_manager.stop()

    # ── Schedule setup ────────────────────────────────────────────────────────

    def _setup_schedules(self):
        region_schedules = {
            "US":   dict(hour=9, minute=35, timezone="America/New_York"),
            "EU":   dict(hour=9, minute=5,  timezone="Europe/Berlin"),
            "ASIA": dict(hour=9, minute=5,  timezone="Asia/Tokyo"),
        }
        for region in _REGIONS:
            if region == "CRYPTO":
                self._scheduler.add_job(
                    self._screen_region,
                    IntervalTrigger(hours=6),
                    args=[region],
                    id=f"screen_{region}",
                    name=f"Screening {region}",
                    max_instances=1,
                    coalesce=True,
                )
            elif region in region_schedules:
                self._scheduler.add_job(
                    self._screen_region,
                    CronTrigger(**region_schedules[region]),
                    args=[region],
                    id=f"screen_{region}",
                    name=f"Screening {region}",
                    max_instances=1,
                    coalesce=True,
                )

        self._scheduler.add_job(
            self._review_positions,
            CronTrigger(hour=7, minute=0),
            id="position_review",
            name="Daily position review",
            max_instances=1,
            coalesce=True,
        )
        self._scheduler.add_job(
            self._trailing_stop_cycle,
            IntervalTrigger(minutes=60),
            id="trailing_stops",
            name="Trailing stop adjustment",
            max_instances=1,
            coalesce=True,
        )
        self._scheduler.add_job(
            self._daily_summary,
            CronTrigger(hour=23, minute=0),
            id="daily_summary",
            name="Daily P&L summary",
        )

    # ── Scheduled jobs ────────────────────────────────────────────────────────

    async def _screen_region(self, region: str):
        logger.info("=== Screening %s — %s ===", region, _utcnow().isoformat())
        self.state.reset_daily_if_needed()

        if region != "CRYPTO" and not is_trading_day(region):
            logger.info("Screening %s: not a trading day — skipping", region)
            return

        balance = await self._get_balance()
        if balance <= 0:
            logger.warning("Cannot fetch balance — skipping %s cycle", region)
            return

        symbols = get_symbols(region)
        if not symbols:
            return

        try:
            shortlist = await self.screening_agent.run(symbols)
        except Exception as exc:
            logger.error("Screening failed for %s: %s", region, exc, exc_info=True)
            await self.notification_agent.send_critical_error(f"Screening failed for {region}: {exc}")
            shortlist = []

        if not shortlist:
            logger.info("Screening %s: empty shortlist", region)
            return

        logger.info(
            "Screening %s: shortlist=%s", region, [r.symbol for r in shortlist]
        )

        # Compute unrealized P&L once before the execution loop (P1-4)
        unrealized_pnl = self._unrealized_pnl()

        for result in shortlist:
            try:
                await self._build_and_execute(result, balance, unrealized_pnl)
            except Exception as exc:
                logger.error("Error on %s: %s", result.symbol, exc, exc_info=True)
                await self.notification_agent.send_critical_error(
                    f"Error processing {result.symbol}: {exc}"
                )

        self.state.last_updated = _utcnow()
        logger.info("=== Screening %s complete ===", region)

    async def _build_and_execute(
        self, result: ScreeningResult, balance: float, unrealized_pnl: float
    ):
        thesis = build_thesis(result)
        symbol = thesis.symbol

        approved, reason = risk_gate_module.validate(
            thesis, self.state, balance, unrealized_pnl=unrealized_pnl
        )
        if not approved:
            logger.info("%s: rejected — %s", symbol, reason)
            if thesis.action != "hold":
                await self.notification_agent.send_thesis_rejected(thesis, reason)
            if self.state.is_risk_blocked:
                await self.notification_agent.send_risk_blocked(self.state.risk_block_reason)
            return

        instrument_id = self._instrument_cache.get(symbol)
        if not instrument_id:
            logger.warning("%s: no instrument_id — skipping", symbol)
            return

        current_price, atr = await self._fetch_price_and_atr(symbol)
        if current_price <= 0:
            logger.warning("%s: no valid price data — skipping execution", symbol)
            return
        order = size_position(thesis, instrument_id, balance, current_price, atr)
        logger.info(
            "%s: placing order amount=$%.2f stop=%.4f%% horizon=%dd",
            symbol, order.amount_usd, order.stop_loss_pct, thesis.horizon_days,
        )
        await self.execution_agent.execute(order)

    async def _review_positions(self):
        logger.info("=== Daily position review — %s ===", _utcnow().isoformat())
        try:
            await self.position_review_agent.review_all()
        except Exception as exc:
            logger.error("Position review error: %s", exc, exc_info=True)
            await self.notification_agent.send_critical_error(f"Position review cycle failed: {exc}")

    async def _trailing_stop_cycle(self):
        try:
            await self.trailing_stop_agent.adjust_all()
        except Exception as exc:
            logger.error("Trailing stop cycle error: %s", exc, exc_info=True)
            await self.notification_agent.send_critical_error(f"Trailing stop cycle failed: {exc}")

    async def _daily_summary(self):
        try:
            await self.notification_agent.send_daily_summary()
        except Exception as exc:
            logger.error("Daily summary error: %s", exc, exc_info=True)
            await self.notification_agent.send_critical_error(f"Daily summary failed: {exc}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _validate_credentials(self):
        balance = await self._get_balance()
        if balance > 0:
            logger.info("Credentials OK — balance: $%.2f", balance)
        else:
            logger.warning("Balance=0 — check credentials or ETORO_MODE")

    async def _reconcile_open_positions(self):
        """Sync in-memory state against live broker portfolio at startup (P0-2)."""
        try:
            portfolio = await self.client.get_portfolio()
        except Exception as exc:
            logger.warning(
                "Portfolio reconcile failed: %s — using saved state (%d positions)",
                exc, len(self.state.open_positions),
            )
            return

        if not isinstance(portfolio, list):
            logger.warning("Unexpected portfolio format: %s", type(portfolio))
            return

        broker_ids: dict[str, dict] = {}
        for item in portfolio:
            pid = str(item.get("positionId") or item.get("id") or "")
            if pid:
                broker_ids[pid] = item

        # Update current rates for known positions; remove positions closed externally
        for pos in list(self.state.open_positions):
            if pos.position_id in broker_ids:
                item = broker_ids[pos.position_id]
                pos.current_rate = float(
                    item.get("currentRate") or item.get("rate") or pos.current_rate
                )
            else:
                logger.warning(
                    "Position %s (%s) not in broker portfolio — removing from state",
                    pos.position_id, pos.symbol,
                )
                self.state.remove_position(pos.position_id)

        # Positions present on the broker but not in our state are NOT adopted —
        # the bot only manages (reviews, trails stops on, closes) positions it
        # opened itself. This account had 29 pre-existing manual positions when
        # the bot was first connected; silently adopting them would put the bot
        # in control of trades it never analysed. Log them for visibility only.
        known_ids = {p.position_id for p in self.state.open_positions}
        foreign_ids = [pid for pid in broker_ids if pid not in known_ids]
        if foreign_ids:
            logger.info(
                "%d broker position(s) not opened by this bot — left untouched: %s",
                len(foreign_ids), foreign_ids,
            )

        logger.info(
            "Reconciliation complete — %d position(s) in state", len(self.state.open_positions)
        )
        self.state.save()

    async def _get_balance(self) -> float:
        try:
            return await self.client.get_balance()
        except Exception as exc:
            logger.error("get_balance failed: %s", exc)
            return 0.0

    def _unrealized_pnl(self) -> float:
        """Estimate unrealized P&L using cached current rates vs entry rates (P1-4)."""
        total = 0.0
        for pos in self.state.open_positions:
            if pos.current_rate and pos.entry_rate:
                if pos.is_buy:
                    total += (pos.current_rate - pos.entry_rate) / pos.entry_rate * pos.amount_usd
                else:
                    total += (pos.entry_rate - pos.current_rate) / pos.entry_rate * pos.amount_usd
        return total

    async def _prefetch_instruments(self):
        all_symbols: list[str] = []
        for region in _REGIONS:
            all_symbols.extend(get_symbols(region))

        for symbol in all_symbols:
            cached = cache_instrument_id(symbol)
            if cached:
                self._instrument_cache[symbol] = cached
                continue
            try:
                instr_id = await self.client.get_instrument_id(symbol)
                if instr_id:
                    self._instrument_cache[symbol] = instr_id
            except Exception as exc:
                logger.debug("Instrument resolution failed for %s: %s", symbol, exc)

        logger.info(
            "Instrument cache: %d/%d symbols resolved",
            len(self._instrument_cache), len(all_symbols),
        )

    async def _fetch_price_and_atr(self, symbol: str) -> tuple[float, float]:
        try:
            result = await self.mcp_manager.call_tool(
                "indicators_full_analysis",
                {"symbol": symbol, "interval": "D1", "count": 60},
            )
            price = float(result.get("last_close") or 0)
            atr = float(result.get("atr_14") or 0)
            return price, atr
        except Exception:
            try:
                rates = await self.client.get_rates([symbol])
                rate_info = rates.get(symbol, {})
                price = float(rate_info.get("close") or rate_info.get("bid") or 0)
                return price, 0.0
            except Exception:
                return 0.0, 0.0
