"""
TrailingStopAgent — deterministic, no LLM.
Runs every 60 minutes. Tightens stop-loss when price moves favourably
by more than 0.5 × ATR from the entry price, then pushes the new stop
to the broker via EtoroClient.update_stop_loss().
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.state import Position, ProjectState

if TYPE_CHECKING:
    from src.core.etoro_client import EtoroClient

logger = logging.getLogger(__name__)

_TRAILING_THRESHOLD_ATR_MULTIPLE = 0.5  # move favourably by this × ATR to tighten
_NEW_STOP_ATR_MULTIPLE = 1.5            # new stop = current ± this × ATR


class TrailingStopAgent:
    def __init__(self, client: "EtoroClient", state: ProjectState):
        self.client = client
        self.state = state

    async def adjust_all(self):
        """Main entry point. Called by orchestrator every 60 min."""
        if not self.state.open_positions:
            return

        symbols = list({p.symbol for p in self.state.open_positions})
        try:
            rates = await self.client.get_rates(symbols)
        except Exception as exc:
            logger.error("TrailingStopAgent: get_rates failed: %s", exc)
            return

        for pos in list(self.state.open_positions):
            try:
                await self._maybe_tighten(pos, rates)
            except Exception as exc:
                logger.error("TrailingStopAgent: error adjusting %s: %s", pos.symbol, exc)

    async def _maybe_tighten(self, pos: Position, rates: dict):
        rate_info = rates.get(pos.symbol, {})
        if not rate_info:
            return

        current = float(
            rate_info.get("close", rate_info.get("bid", rate_info.get("rate", 0)))
        )
        if not current:
            return

        pos.current_rate = current
        atr = pos.atr or 0.0
        if not atr:
            return

        threshold = _TRAILING_THRESHOLD_ATR_MULTIPLE * atr
        new_sl_pct: float | None = None

        if pos.is_buy:
            gain = current - pos.entry_rate
            if gain <= threshold:
                return
            new_sl_price = current - _NEW_STOP_ATR_MULTIPLE * atr
            new_sl_pct = (1 - new_sl_price / current) * 100
        else:
            gain = pos.entry_rate - current
            if gain <= threshold:
                return
            new_sl_price = current + _NEW_STOP_ATR_MULTIPLE * atr
            new_sl_pct = (new_sl_price / current - 1) * 100

        if new_sl_pct is None or new_sl_pct >= pos.stop_loss_pct:
            return  # only tighten, never widen

        old_pct = pos.stop_loss_pct
        pos.stop_loss_pct = round(new_sl_pct, 4)

        logger.info(
            "TrailingStopAgent: %s %s stop %.4f%% → %.4f%% "
            "(price=%.4f entry=%.4f gain=%.4f ATR=%.4f)",
            pos.symbol, "LONG" if pos.is_buy else "SHORT",
            old_pct, new_sl_pct, current, pos.entry_rate, gain, atr,
        )

        # Push the new stop to the broker.
        # NOTE: verify update_stop_loss() endpoint/payload with eToro API docs.
        try:
            await self.client.update_stop_loss(
                pos.position_id, pos.instrument_id, pos.stop_loss_pct
            )
        except Exception as exc:
            # Revert in-memory change so we retry next cycle rather than silently
            # diverging from the broker's actual stop.
            pos.stop_loss_pct = old_pct
            logger.warning(
                "TrailingStopAgent: could not update stop on broker for %s: %s "
                "— reverted in-memory, will retry next cycle",
                pos.symbol, exc,
            )
