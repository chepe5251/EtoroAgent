"""
TrailingStopAgent — deterministic, no LLM.
Runs every 5 minutes. Tightens stop-loss when price moves favourably
by more than 0.5 × ATR from the entry price.
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
    """
    Checks all open positions against current market rates and
    tightens trailing stops when the trade is in profit by > 0.5 × ATR.
    """

    def __init__(self, client: "EtoroClient", state: ProjectState):
        self.client = client
        self.state = state

    async def adjust_all(self):
        """Main entry point. Called by orchestrator every 5 minutes."""
        if not self.state.open_positions:
            return

        symbols = list({p.symbol for p in self.state.open_positions})
        try:
            rates = await self.client.get_rates(symbols)
        except Exception as exc:
            logger.error("TrailingStopAgent: get_rates failed: %s", exc)
            return

        for pos in self.state.open_positions:
            try:
                self._maybe_tighten(pos, rates)
            except Exception as exc:
                logger.error(
                    "TrailingStopAgent: error adjusting %s: %s", pos.symbol, exc
                )

    def _maybe_tighten(self, pos: Position, rates: dict):
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

        if pos.is_buy:
            gain = current - pos.entry_rate
            if gain <= threshold:
                return
            # Move stop up: new stop price = current − 1.5×ATR
            new_sl_price = current - _NEW_STOP_ATR_MULTIPLE * atr
            # Convert to % of current price (how far from current the stop is)
            new_sl_pct = (1 - new_sl_price / current) * 100
            old_pct = pos.stop_loss_pct
            # Only tighten (never widen)
            if new_sl_pct < old_pct:
                pos.stop_loss_pct = round(new_sl_pct, 4)
                logger.info(
                    "TrailingStopAgent: %s LONG stop tightened %.4f%% → %.4f%% "
                    "(price=%.4f entry=%.4f gain=%.4f ATR=%.4f)",
                    pos.symbol, old_pct, new_sl_pct,
                    current, pos.entry_rate, gain, atr,
                )
        else:
            gain = pos.entry_rate - current
            if gain <= threshold:
                return
            # Move stop down: new stop price = current + 1.5×ATR
            new_sl_price = current + _NEW_STOP_ATR_MULTIPLE * atr
            new_sl_pct = (new_sl_price / current - 1) * 100
            old_pct = pos.stop_loss_pct
            if new_sl_pct < old_pct:
                pos.stop_loss_pct = round(new_sl_pct, 4)
                logger.info(
                    "TrailingStopAgent: %s SHORT stop tightened %.4f%% → %.4f%%",
                    pos.symbol, old_pct, new_sl_pct,
                )
