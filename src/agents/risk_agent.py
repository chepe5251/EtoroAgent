import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from src.core.state import Position, ProjectState, Signal

if TYPE_CHECKING:
    from src.core.etoro_client import EtoroClient

load_dotenv()
logger = logging.getLogger(__name__)


class RiskAgent:
    """
    Enforces daily loss limits and manages trailing stops.
    Deterministic — no LLM calls.
    """

    def __init__(self, state: ProjectState, client: "EtoroClient"):
        self.state = state
        self.client = client
        self.daily_loss_limit_pct = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "3.0"))

    def check_risk(self, signal: Signal, balance: float) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        Called by DecisionAgent before approving any order.
        """
        # 1. Daily reset
        self.state.reset_daily_if_needed()

        # 2. Check if risk is already blocked
        if self.state.is_risk_blocked:
            return False, f"Risk blocked: {self.state.risk_block_reason}"

        # 3. Check daily P&L limit
        if balance > 0:
            loss_pct = (-self.state.daily_pnl / balance) * 100
            if loss_pct >= self.daily_loss_limit_pct:
                reason = (
                    f"Daily loss limit reached: {loss_pct:.2f}% >= {self.daily_loss_limit_pct}%"
                )
                self._block(reason)
                return False, reason

        return True, "ok"

    def _block(self, reason: str):
        self.state.is_risk_blocked = True
        self.state.risk_block_reason = reason
        logger.warning("RiskAgent: BLOCKED — %s", reason)

    def record_closed_trade(self, pnl: float):
        """Update daily P&L after a trade closes."""
        self.state.daily_pnl += pnl
        logger.info(
            "RiskAgent: trade closed pnl=%.2f daily_pnl=%.2f", pnl, self.state.daily_pnl
        )

    async def adjust_trailing_stops(self):
        """
        Check every open position and tighten the stop-loss if price moved
        favorably by more than 0.5x ATR since entry.
        """
        if not self.state.open_positions:
            return

        symbols = [p.symbol for p in self.state.open_positions]
        try:
            rates = await self.client.get_rates(symbols)
        except Exception as exc:
            logger.error("RiskAgent.adjust_trailing_stops: get_rates failed: %s", exc)
            return

        for pos in self.state.open_positions:
            try:
                rate_info = rates.get(pos.symbol, {})
                if not rate_info:
                    continue

                current = float(
                    rate_info.get("close", rate_info.get("bid", rate_info.get("rate", 0)))
                )
                if not current:
                    continue

                pos.current_rate = current
                atr = pos.atr
                if not atr:
                    # Fall back to indicator store
                    atr = (
                        self.state.market_data.get(pos.symbol, {})
                        .get("indicators", {})
                        .get("atr_14") or 0
                    )
                if not atr:
                    continue

                threshold = 0.5 * atr
                if pos.is_buy:
                    gain = current - pos.entry_rate
                    if gain > threshold:
                        # Move stop-loss up: new stop = current - 1.5*ATR
                        new_sl_price = current - 1.5 * atr
                        new_sl_pct = (new_sl_price / current) * 100  # pct of current price
                        if new_sl_pct > (100 - pos.stop_loss_pct):
                            pos.stop_loss_pct = 100 - new_sl_pct
                            logger.info(
                                "RiskAgent: trailing stop tightened for %s LONG → stop=%.4f (%.2f%%)",
                                pos.symbol, new_sl_price, pos.stop_loss_pct,
                            )
                else:
                    gain = pos.entry_rate - current
                    if gain > threshold:
                        new_sl_price = current + 1.5 * atr
                        new_sl_pct = ((new_sl_price / current) - 1) * 100
                        if new_sl_pct < pos.stop_loss_pct:
                            pos.stop_loss_pct = new_sl_pct
                            logger.info(
                                "RiskAgent: trailing stop tightened for %s SHORT → stop=%.4f (%.2f%%)",
                                pos.symbol, new_sl_price, pos.stop_loss_pct,
                            )
            except Exception as exc:
                logger.error(
                    "RiskAgent: error adjusting stop for %s: %s", pos.symbol, exc, exc_info=True
                )
