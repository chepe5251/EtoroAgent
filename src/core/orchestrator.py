"""
Orchestrator — swing trading, market-calendar-driven schedule.

Architecture:
  REASONING PLANE  → ScreeningAgent (deterministic+LLM fast filter)
                   → ResearchAgent (full ReAct per shortlisted symbol)
  EXECUTION PLANE  → RiskGate → ExecutionAgent (no LLM)
  REVIEW PLANE     → PositionReviewAgent (ReAct, once/day per position)
  MAINTENANCE      → TrailingStopAgent (every 1h, deterministic)

Schedule per region (at market open + 5 min):
  US:     CronTrigger(hour=9, min=35, tz="America/New_York")
  EU:     CronTrigger(hour=9, min=5,  tz="Europe/Berlin")
  ASIA:   CronTrigger(hour=9, min=5,  tz="Asia/Tokyo")
  CRYPTO: every 6 hours UTC

Daily:
  Position review: 7:00 UTC (before any market opens)
  Daily summary:   23:00 UTC
  Trailing stops:  every 60 min
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
from src.agents.research_agent import ResearchAgent
from src.agents.screening_agent import ScreeningAgent
from src.agents.trailing_stop_agent import TrailingStopAgent
from src.config.universe import get_symbols, get_instrument_id as cache_instrument_id
from src.core.etoro_client import EtoroClient
from src.core.market_calendar import is_trading_day, get_market_status
from src.core.state import Position, ProjectState
from src.mcp_clients.mcp_manager import MCPManager

load_dotenv()
logger = logging.getLogger(__name__)

_REGIONS = [r.strip().upper() for r in os.getenv("WATCH_REGIONS", "US,EU,ASIA,CRYPTO").split(",")]
_MAX_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "5"))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Orchestrator:
    def __init__(self):
        self.state = ProjectState()
        self.client = EtoroClient()
        self.mcp_manager = MCPManager()

        self.notification_agent = NotificationAgent(self.state)
        self.research_agent = ResearchAgent(self.mcp_manager)
        self.screening_agent = ScreeningAgent(self.client)
        self.execution_agent = ExecutionAgent(self.client, self.state, self.notification_agent)
        self.trailing_stop_agent = TrailingStopAgent(self.client, self.state)
        self.position_review_agent = PositionReviewAgent(
            mcp_manager=self.mcp_manager,
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
                await self._prefetch_instruments()
                self._setup_schedules()
                self._scheduler.start()
                mode = os.getenv("ETORO_MODE", "demo").upper()
                all_symbols = [s for r in _REGIONS for s in get_symbols(r)]
                logger.info(
                    "Orchestrator running — mode=%s regions=%s symbols≈%d",
                    mode, _REGIONS, len(all_symbols),
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
        # Per-region screening + research cycles
        region_schedules = {
            "US":    dict(hour=9, minute=35, timezone="America/New_York"),
            "EU":    dict(hour=9, minute=5,  timezone="Europe/Berlin"),
            "ASIA":  dict(hour=9, minute=5,  timezone="Asia/Tokyo"),
            "CRYPTO": None,  # handled by interval below
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
                kwargs = region_schedules[region]
                self._scheduler.add_job(
                    self._screen_region,
                    CronTrigger(**kwargs),
                    args=[region],
                    id=f"screen_{region}",
                    name=f"Screening {region}",
                    max_instances=1,
                    coalesce=True,
                )

        # Daily position review (07:00 UTC — before any equity market opens)
        self._scheduler.add_job(
            self._review_positions,
            CronTrigger(hour=7, minute=0),
            id="position_review",
            name="Daily position review",
            max_instances=1,
            coalesce=True,
        )

        # Trailing stops every 60 min
        self._scheduler.add_job(
            self._trailing_stop_cycle,
            IntervalTrigger(minutes=60),
            id="trailing_stops",
            name="Trailing stop adjustment",
            max_instances=1,
            coalesce=True,
        )

        # Daily summary at 23:00 UTC
        self._scheduler.add_job(
            self._daily_summary,
            CronTrigger(hour=23, minute=0),
            id="daily_summary",
            name="Daily P&L summary",
        )

    # ── Scheduled jobs ────────────────────────────────────────────────────────

    async def _screen_region(self, region: str):
        """Screen all symbols in a region and deep-research the shortlist."""
        logger.info("=== Screening %s — %s ===", region, _utcnow().isoformat())
        self.state.reset_daily_if_needed()

        # Skip equity regions on non-trading days
        if region != "CRYPTO" and not is_trading_day(region):
            logger.info("Screening %s: not a trading day — skipping", region)
            return

        balance = await self._get_balance()
        if balance <= 0:
            logger.warning("Cannot fetch balance — skipping %s cycle", region)
            return

        symbols = get_symbols(region)
        if not symbols:
            logger.warning("No symbols for region %s", region)
            return

        # Stage 1: Screen
        try:
            shortlist = await self.screening_agent.run(symbols)
        except Exception as exc:
            logger.error("Screening failed for %s: %s", region, exc, exc_info=True)
            shortlist = []

        if not shortlist:
            logger.info("Screening %s: empty shortlist", region)
            return

        logger.info("Screening %s: researching shortlist=%s", region, shortlist)

        # Stage 2: Deep research on shortlist
        for symbol in shortlist:
            try:
                await self._research_and_execute(symbol, balance)
            except Exception as exc:
                logger.error("Error on %s: %s", symbol, exc, exc_info=True)
                await self.notification_agent.send_critical_error(
                    f"Error processing {symbol}: {exc}"
                )

        self.state.last_updated = _utcnow()
        logger.info("=== Screening %s complete ===", region)

    async def _research_and_execute(self, symbol: str, balance: float):
        # ── Deep ReAct research ───────────────────────────────────────────
        thesis = await self.research_agent.run(symbol)
        if thesis is None:
            logger.warning("%s: no thesis produced", symbol)
            return

        # ── Risk gate ─────────────────────────────────────────────────────
        approved, reason = risk_gate_module.validate(thesis, self.state, balance)
        if not approved:
            logger.info("%s: rejected — %s", symbol, reason)
            if thesis.action != "hold":
                await self.notification_agent.send_thesis_rejected(thesis, reason)
            if self.state.is_risk_blocked:
                await self.notification_agent.send_risk_blocked(self.state.risk_block_reason)
            return

        # ── Size + execute ────────────────────────────────────────────────
        instrument_id = self._instrument_cache.get(symbol)
        if not instrument_id:
            logger.warning("%s: no instrument_id — skipping", symbol)
            return

        current_price, atr = await self._fetch_price_and_atr(symbol)
        order = size_position(thesis, instrument_id, balance, current_price, atr)
        logger.info(
            "%s: placing order amount=$%.2f stop=%.4f%%",
            symbol, order.amount_usd, order.stop_loss_pct,
        )
        await self.execution_agent.execute(order)

    async def _review_positions(self):
        logger.info("=== Daily position review — %s ===", _utcnow().isoformat())
        try:
            await self.position_review_agent.review_all()
        except Exception as exc:
            logger.error("Position review error: %s", exc, exc_info=True)

    async def _trailing_stop_cycle(self):
        try:
            await self.trailing_stop_agent.adjust_all()
        except Exception as exc:
            logger.error("Trailing stop cycle error: %s", exc, exc_info=True)

    async def _daily_summary(self):
        try:
            await self.notification_agent.send_daily_summary()
        except Exception as exc:
            logger.error("Daily summary error: %s", exc, exc_info=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _validate_credentials(self):
        balance = await self._get_balance()
        if balance > 0:
            logger.info("Credentials OK — balance: $%.2f", balance)
        else:
            logger.warning("Balance=0 — check credentials or ETORO_MODE")

    async def _get_balance(self) -> float:
        try:
            data = await self.client.get_balance()
            return float(
                data.get("availableToTrade", data.get("balance", data.get("equity", 0)))
            )
        except Exception as exc:
            logger.error("get_balance failed: %s", exc)
            return 0.0

    async def _prefetch_instruments(self):
        """Resolve symbol → instrument_id at startup. Uses cache when available."""
        all_symbols: list[str] = []
        for region in _REGIONS:
            all_symbols.extend(get_symbols(region))

        for symbol in all_symbols:
            # Try discovery cache first
            cached = cache_instrument_id(symbol)
            if cached:
                self._instrument_cache[symbol] = cached
                continue
            # Fall back to live lookup
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
                price = float(rate_info.get("close", rate_info.get("bid", 0)))
                return price, 0.0
            except Exception:
                return 0.0, 0.0
