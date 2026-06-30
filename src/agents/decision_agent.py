import logging
import os
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from dotenv import load_dotenv

from src.core.state import Position, ProjectState, Signal

if TYPE_CHECKING:
    from src.agents.risk_agent import RiskAgent

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class ApprovedOrder:
    signal: Signal
    instrument_id: str
    amount_usd: float
    stop_loss_pct: float
    atr: float


@dataclass
class RejectedOrder:
    signal: Signal
    reason: str


class DecisionAgent:
    """
    Applies position-sizing rules and validates against RiskAgent.
    Deterministic — no LLM calls.
    """

    def __init__(self, state: ProjectState, risk_agent: "RiskAgent"):
        self.state = state
        self.risk_agent = risk_agent
        self.max_position_size_pct = float(os.getenv("MAX_POSITION_SIZE_PCT", "2.0"))
        self.max_open_positions = int(os.getenv("MAX_OPEN_POSITIONS", "3"))

    async def run(
        self,
        signals: list[Signal],
        balance: float,
        instrument_ids: dict[str, str],
    ) -> tuple[list[ApprovedOrder], list[RejectedOrder]]:
        approved: list[ApprovedOrder] = []
        rejected: list[RejectedOrder] = []

        for signal in signals:
            if signal.action == "HOLD":
                continue

            reason = self._pre_check(signal, balance)
            if reason:
                rejected.append(RejectedOrder(signal=signal, reason=reason))
                logger.info("DecisionAgent: %s rejected — %s", signal.symbol, reason)
                continue

            allowed, risk_reason = self.risk_agent.check_risk(signal, balance)
            if not allowed:
                rejected.append(RejectedOrder(signal=signal, reason=risk_reason))
                logger.info("DecisionAgent: %s blocked by RiskAgent — %s", signal.symbol, risk_reason)
                continue

            instrument_id = instrument_ids.get(signal.symbol)
            if not instrument_id:
                rejected.append(
                    RejectedOrder(signal=signal, reason=f"No instrument_id for {signal.symbol}")
                )
                continue

            amount_usd = balance * (self.max_position_size_pct / 100.0)
            atr = (
                self.state.market_data.get(signal.symbol, {})
                .get("indicators", {})
                .get("atr_14") or 0
            )
            last_close = (
                self.state.market_data.get(signal.symbol, {})
                .get("indicators", {})
                .get("last_close") or 1
            )
            # Stop-loss distance = 1.5 * ATR; convert to % of current price
            sl_distance = 1.5 * atr if atr else last_close * 0.02
            stop_loss_pct = (sl_distance / last_close) * 100 if last_close else 2.0

            order = ApprovedOrder(
                signal=signal,
                instrument_id=instrument_id,
                amount_usd=round(amount_usd, 2),
                stop_loss_pct=round(stop_loss_pct, 4),
                atr=atr,
            )
            approved.append(order)
            logger.info(
                "DecisionAgent: APPROVED %s %s $%.2f sl=%.2f%%",
                signal.symbol, signal.action, amount_usd, stop_loss_pct,
            )

        return approved, rejected

    def _pre_check(self, signal: Signal, balance: float) -> Optional[str]:
        """Returns a rejection reason string, or None if checks pass."""
        if len(self.state.open_positions) >= self.max_open_positions:
            return f"Max open positions ({self.max_open_positions}) reached"

        existing = self.state.get_open_position(signal.symbol)
        if existing:
            return f"Position already open for {signal.symbol}"

        min_trade = 10.0
        amount = balance * (self.max_position_size_pct / 100.0)
        if amount < min_trade:
            return f"Insufficient balance for trade (${amount:.2f} < ${min_trade})"

        return None
