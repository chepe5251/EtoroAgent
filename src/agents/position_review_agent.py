"""
PositionReviewAgent — daily review of open swing positions. No LLM.

For each open position it:
  1. Enforces the 20-day hard exit (deterministic).
  2. Checks the same trend_break exit condition validated in the backtest
     (src/backtest/engine.py, exit_mode="trend_break"): if the daily close
     is below EMA50, the trend that justified the trade is gone — exit.
  3. Otherwise holds. Stop-tightening is handled separately, every 60
     minutes, by TrailingStopAgent (already deterministic, ATR-based).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.state import Position, ProjectState, get_hard_exit_days
from src.tools.technical import ema as _ema

if TYPE_CHECKING:
    from src.agents.execution_agent import ExecutionAgent
    from src.agents.notification_agent import NotificationAgent

logger = logging.getLogger(__name__)

_EMA_PERIOD = 50
_CANDLE_COUNT = 60


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
        logger.info(
            "PositionReview: %s — day %d/%d (horizon=%d)",
            pos.symbol, pos.days_open, get_hard_exit_days(), pos.horizon_days,
        )

        # ── Hard exit: 20-day absolute limit (deterministic) ──────────────
        if pos.is_past_hard_exit:
            logger.warning(
                "PositionReview: %s hit 20-day hard limit — forcing EXIT", pos.symbol
            )
            await self._close(pos, reason=f"20-day hard exit limit reached (opened {pos.days_open}d ago)")
            return

        # ── Trend-break exit: close < EMA50 (deterministic) ────────────────
        # Same condition validated in the backtest's exit_mode="trend_break".
        verdict = await self._technical_review(pos)
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

    async def _technical_review(self, pos: Position) -> dict | None:
        """Deterministic trend-break check: exit if close < EMA50."""
        try:
            candles = await self.execution_agent.client.get_candles(
                pos.symbol, interval="D1", count=_CANDLE_COUNT
            )
        except Exception as exc:
            logger.warning("PositionReview: candle fetch failed for %s: %s", pos.symbol, exc)
            return None

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
