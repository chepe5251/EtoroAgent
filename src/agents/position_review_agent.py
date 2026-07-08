"""
PositionReviewAgent — daily review of open swing positions. No LLM.

For each open position it:
  1. Enforces the hard exit, measured in trading-day BARS held — matching
     src/backtest/engine.py's max_hold_days convention exactly (20 bars =
     20 trading days ≈ 28 calendar days). Bars held is computed by counting
     the symbol's own daily candles dated after the position was opened,
     using the same candle fetch the trend-break check already needs — so
     this self-corrects across restarts/missed cycles rather than relying
     on a persisted counter that could drift out of sync.
     (Previously this counted raw calendar days via Position.days_open —
     ~14 trading days, a real divergence from the backtest that's now fixed.)
  2. Checks the same trend_break exit condition validated in the backtest
     (src/backtest/engine.py, exit_mode="trend_break"): if the daily close
     is below EMA50, the trend that justified the trade is gone — exit.
  3. Otherwise holds. Stop-tightening is handled separately, every 60
     minutes, by TrailingStopAgent (already deterministic, ATR-based).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.core.state import Position, ProjectState, get_hard_exit_days
from src.tools.technical import ema as _ema

if TYPE_CHECKING:
    from src.agents.execution_agent import ExecutionAgent
    from src.agents.notification_agent import NotificationAgent

logger = logging.getLogger(__name__)

_EMA_PERIOD = 50
_CANDLE_COUNT = 60


def _parse_candle_date(raw: dict) -> datetime | None:
    date_str = raw.get("fromDate") or raw.get("date")
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _bars_held(candles: list[dict], opened_at: datetime) -> int:
    """Count candles dated strictly after `opened_at` — the number of
    trading-day bars this position has been held, matching engine.py's
    bar-count time-limit convention (not raw calendar days)."""
    count = 0
    for c in candles:
        dt = _parse_candle_date(c)
        if dt is not None and dt > opened_at:
            count += 1
    return count


class PositionReviewAgent:
    """Reviews open positions daily and acts on stale or invalidated trades."""

    def __init__(
        self,
        execution_agent: "ExecutionAgent",
        notification_agent: "NotificationAgent",
        state: ProjectState,
    ):
        self.execution_agent = execution_agent
        self.notification_agent = notification_agent
        self.state = state

    async def review_all(self):
        """Review every open position. Called once per day."""
        positions = list(self.state.open_positions)
        if not positions:
            logger.info("PositionReview: no open positions to review")
            return

        logger.info("PositionReview: reviewing %d open position(s)", len(positions))
        for pos in positions:
            try:
                await self._review_position(pos)
            except Exception as exc:
                logger.error("Error reviewing position %s: %s", pos.symbol, exc, exc_info=True)
                await self.notification_agent.send_critical_error(f"Error reviewing position {pos.symbol}: {exc}")

    async def _review_position(self, pos: Position):
        try:
            candles = await self.execution_agent.client.get_candles(
                pos.symbol, interval="D1", count=_CANDLE_COUNT
            )
        except Exception as exc:
            logger.warning("PositionReview: candle fetch failed for %s: %s", pos.symbol, exc)
            return

        bars_held = _bars_held(candles, pos.opened_at)
        hard_exit_bars = get_hard_exit_days()
        logger.info(
            "PositionReview: %s — bar %d/%d held (horizon=%d)",
            pos.symbol, bars_held, hard_exit_bars, pos.horizon_days,
        )

        # ── Hard exit: absolute bar-count limit (deterministic) ───────────
        # Matches src/backtest/engine.py's max_hold_days: BARS held, not
        # calendar days (20 bars ≈ 28 calendar days, not 20 calendar days).
        if bars_held >= hard_exit_bars:
            logger.warning(
                "PositionReview: %s hit %d-bar hard limit — forcing EXIT", pos.symbol, hard_exit_bars
            )
            await self._close(pos, reason=f"{hard_exit_bars}-bar hard exit limit reached ({bars_held} bars held)")
            return

        # ── Trend-break exit: close < EMA50 (deterministic) ────────────────
        # Same condition validated in the backtest's exit_mode="trend_break".
        verdict = self._technical_review(candles)
        if verdict is None:
            logger.warning("PositionReview: %s — no price data, skipping", pos.symbol)
            return

        action = verdict["action"]
        reason = verdict["reason"]
        logger.info("PositionReview: %s → %s (%s)", pos.symbol, action.upper(), reason)

        if action == "exit":
            await self._close(pos, reason=reason)
        else:
            logger.info("PositionReview: %s → HOLD, no action", pos.symbol)

    def _technical_review(self, candles: list[dict]) -> dict | None:
        """Deterministic trend-break check: exit if close < EMA50."""
        closes = [float(c.get("close") or c.get("c") or 0) for c in candles]
        if len(closes) < _EMA_PERIOD:
            return None

        ema_series = _ema(closes, _EMA_PERIOD)
        if not ema_series:
            return None

        last_close = closes[-1]
        last_ema = ema_series[-1]

        if last_close < last_ema:
            return {
                "action": "exit",
                "reason": f"Trend break: close {last_close:.2f} < EMA50 {last_ema:.2f}",
            }
        return {
            "action": "hold",
            "reason": f"Trend intact: close {last_close:.2f} >= EMA50 {last_ema:.2f}",
        }

    async def _close(self, pos: Position, reason: str):
        """Close a position and notify."""
        try:
            closed = await self.execution_agent.close_position(pos.position_id, reason=reason)
            if not closed:
                logger.warning("PositionReview: close failed for %s — %s", pos.symbol, reason)
                return
            logger.info("PositionReview: closed %s — %s", pos.symbol, reason)
        except Exception as exc:
            logger.error("PositionReview: close failed for %s: %s", pos.symbol, exc)
