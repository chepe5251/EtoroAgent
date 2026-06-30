"""
ExecutionAgent — 100% deterministic order execution.
No LLM. Receives an approved, fully-sized order and submits it to eToro.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.core.state import Position, ProjectState
from src.core.thesis import TradingThesis

if TYPE_CHECKING:
    from src.agents.notification_agent import NotificationAgent
    from src.core.etoro_client import EtoroClient

logger = logging.getLogger(__name__)

_MAX_POSITION_SIZE_PCT = float(os.getenv("MAX_POSITION_SIZE_PCT", "2.0"))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SizedOrder:
    """Fully computed order ready for execution. Created by size_position()."""
    thesis: TradingThesis
    instrument_id: str
    amount_usd: float
    stop_loss_pct: float
    atr: float


def size_position(
    thesis: TradingThesis,
    instrument_id: str,
    balance: float,
    current_price: float,
    atr: float,
) -> SizedOrder:
    """
    Deterministic position sizing.
    - Amount: MAX_POSITION_SIZE_PCT % of available balance
    - Stop-loss: thesis.suggested_stop_loss_atr_multiple × ATR, expressed
      as a % of current price
    """
    amount_usd = balance * (_MAX_POSITION_SIZE_PCT / 100.0)
    amount_usd = round(amount_usd, 2)

    sl_distance = thesis.suggested_stop_loss_atr_multiple * atr if atr else current_price * 0.02
    stop_loss_pct = (sl_distance / current_price * 100) if current_price else 2.0
    stop_loss_pct = round(stop_loss_pct, 4)

    return SizedOrder(
        thesis=thesis,
        instrument_id=instrument_id,
        amount_usd=amount_usd,
        stop_loss_pct=stop_loss_pct,
        atr=atr,
    )


class ExecutionAgent:
    """
    Submits a SizedOrder to eToro and updates shared state.
    Retries once on failure, then notifies and aborts.
    """

    def __init__(
        self,
        client: "EtoroClient",
        state: ProjectState,
        notification_agent: "NotificationAgent",
    ):
        self.client = client
        self.state = state
        self.notification_agent = notification_agent

    async def execute(self, order: SizedOrder) -> bool:
        """
        Open a position. Returns True on success, False on failure.
        """
        is_buy = order.thesis.action == "buy"
        symbol = order.thesis.symbol
        logger.info(
            "ExecutionAgent: opening %s %s $%.2f sl=%.4f%%",
            "BUY" if is_buy else "SELL", symbol, order.amount_usd, order.stop_loss_pct,
        )

        result = await self._try_open(order, is_buy, attempt=1)
        if result is None:
            result = await self._try_open(order, is_buy, attempt=2)

        if result is None:
            msg = f"Failed to open {symbol} after 2 attempts"
            logger.error("ExecutionAgent: %s", msg)
            await self.notification_agent.send_critical_error(msg)
            return False

        position_id = str(
            result.get("positionId", result.get("id", result.get("position_id", f"unknown-{symbol}")))
        )
        rate = float(result.get("rate", result.get("openRate", 0)))

        pos = Position(
            position_id=position_id,
            instrument_id=order.instrument_id,
            symbol=symbol,
            is_buy=is_buy,
            amount_usd=order.amount_usd,
            entry_rate=rate,
            stop_loss_pct=order.stop_loss_pct,
            opened_at=_utcnow(),
            current_rate=rate,
            atr=order.atr,
        )
        self.state.open_positions.append(pos)
        logger.info(
            "ExecutionAgent: position opened id=%s %s rate=%.4f",
            position_id, symbol, rate,
        )
        await self.notification_agent.send_position_opened(pos)
        return True

    async def close_position(self, position_id: str, reason: str = "manual") -> bool:
        """Close a specific open position."""
        pos = next((p for p in self.state.open_positions if p.position_id == position_id), None)
        if not pos:
            logger.warning("ExecutionAgent: position %s not found in state", position_id)
            return False

        logger.info("ExecutionAgent: closing %s (%s) — reason: %s", pos.symbol, position_id, reason)
        try:
            result = await self.client.close_position(position_id, pos.instrument_id)
        except Exception as exc:
            logger.error("ExecutionAgent: close failed for %s: %s", position_id, exc)
            await self.notification_agent.send_critical_error(
                f"Failed to close {pos.symbol} ({position_id}): {exc}"
            )
            return False

        close_rate = float(result.get("rate", result.get("closeRate", pos.current_rate)))
        if pos.is_buy:
            pnl = (close_rate - pos.entry_rate) / pos.entry_rate * pos.amount_usd
        else:
            pnl = (pos.entry_rate - close_rate) / pos.entry_rate * pos.amount_usd

        duration = _utcnow() - pos.opened_at
        self.state.remove_position(position_id)

        from src.agents import risk_gate
        risk_gate.record_closed_pnl(self.state, pnl)

        logger.info("ExecutionAgent: %s closed pnl=%.2f duration=%s", pos.symbol, pnl, duration)
        await self.notification_agent.send_position_closed(pos, pnl, duration)
        return True

    async def _try_open(self, order: SizedOrder, is_buy: bool, attempt: int):
        try:
            return await self.client.open_position(
                instrument_id=order.instrument_id,
                amount_usd=order.amount_usd,
                is_buy=is_buy,
                stop_loss_pct=order.stop_loss_pct,
                trailing_stop=True,
            )
        except Exception as exc:
            logger.warning(
                "ExecutionAgent: open attempt %d failed for %s: %s",
                attempt, order.thesis.symbol, exc,
            )
            return None
