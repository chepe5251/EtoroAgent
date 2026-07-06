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

from src.core import trade_log
from src.core.state import Position, ProjectState
from src.core.thesis import TradingThesis

if TYPE_CHECKING:
    from src.agents.notification_agent import NotificationAgent
    from src.core.etoro_client import EtoroClient

logger = logging.getLogger(__name__)

_MAX_POSITION_SIZE_PCT = float(os.getenv("MAX_POSITION_SIZE_PCT", "10.0"))
_RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))
_LEVERAGE = float(os.getenv("LEVERAGE", "1.0"))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SizedOrder:
    """Fully computed order ready for execution. Created by size_position().

    amount_usd is NOTIONAL exposure (matches src/backtest/engine.py's
    accounting) — ExecutionAgent converts to broker margin (amount_usd /
    leverage) when submitting the real order.
    """
    thesis: TradingThesis
    instrument_id: str
    amount_usd: float
    stop_loss_pct: float
    current_price: float
    atr: float
    leverage: float = _LEVERAGE


def size_position(
    thesis: TradingThesis,
    instrument_id: str,
    balance: float,
    current_price: float,
    atr: float,
) -> SizedOrder:
    """
    Risk-based position sizing — matches src/backtest/engine.py exactly:
      - Stop distance = suggested_stop_loss_atr_multiple × ATR
      - risk_amount = balance × RISK_PER_TRADE_PCT%
      - units = risk_amount / stop_distance ; notional = units × price
      - capped at MAX_POSITION_SIZE_PCT% × LEVERAGE of balance
    """
    sl_distance = thesis.suggested_stop_loss_atr_multiple * atr if atr else current_price * 0.02
    stop_loss_pct = (sl_distance / current_price * 100) if current_price else 2.0
    stop_loss_pct = round(stop_loss_pct, 4)

    notional = 0.0
    if sl_distance > 0 and current_price > 0 and balance > 0:
        risk_amount = balance * (_RISK_PER_TRADE_PCT / 100.0)
        units = risk_amount / sl_distance
        notional = units * current_price

    max_notional = balance * (_MAX_POSITION_SIZE_PCT / 100.0 * _LEVERAGE)
    amount_usd = min(notional, max_notional)
    amount_usd = round(amount_usd, 2)

    return SizedOrder(
        thesis=thesis,
        instrument_id=instrument_id,
        amount_usd=amount_usd,
        stop_loss_pct=stop_loss_pct,
        current_price=current_price,
        atr=atr,
        leverage=_LEVERAGE,
    )


class ExecutionAgent:
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
        """Open a position. Returns True on success, False on failure.

        Single attempt only — EtoroClient does NOT retry writes to avoid
        duplicate positions.  If the call fails, the caller should reconcile
        against get_portfolio() before retrying.
        """
        is_buy = order.thesis.action == "buy"
        symbol = order.thesis.symbol
        leverage = max(1, round(order.leverage))
        margin_usd = round(order.amount_usd / leverage, 2)
        logger.info(
            "ExecutionAgent: opening %s %s notional=$%.2f margin=$%.2f leverage=%dx sl=%.4f%%",
            "BUY" if is_buy else "SELL", symbol, order.amount_usd, margin_usd, leverage, order.stop_loss_pct,
        )

        try:
            result = await self.client.open_position(
                instrument_id=order.instrument_id,
                amount_usd=margin_usd,
                is_buy=is_buy,
                stop_loss_pct=order.stop_loss_pct,
                current_price=order.current_price,
                trailing_stop=True,
                leverage=leverage,
            )
        except Exception as exc:
            msg = f"Failed to open {symbol}: {exc}"
            logger.error("ExecutionAgent: %s", msg)
            await self.notification_agent.send_critical_error(msg)
            return False

        position_id = str(
            result.get("positionId") or result.get("id") or result.get("position_id") or f"unknown-{symbol}"
        )
        rate = float(result.get("rate") or result.get("openRate") or 0)

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
            horizon_days=order.thesis.horizon_days,
            invalidation_condition=order.thesis.invalidation_condition,
            leverage=order.leverage,
        )
        self.state.open_positions.append(pos)
        self.state.save()  # persist immediately so state survives a crash
        trade_log.log_open(
            position_id=position_id,
            symbol=symbol,
            is_buy=is_buy,
            amount_usd=order.amount_usd,
            entry_rate=rate,
            stop_loss_pct=order.stop_loss_pct,
            horizon_days=pos.horizon_days,
        )
        logger.info(
            "ExecutionAgent: position opened id=%s %s rate=%.4f horizon=%dd",
            position_id, symbol, rate, pos.horizon_days,
        )
        await self.notification_agent.send_position_opened(pos)
        return True

    async def close_position(self, position_id: str, reason: str = "manual") -> bool:
        """Close a specific open position."""
        pos = next((p for p in self.state.open_positions if p.position_id == position_id), None)
        if not pos:
            logger.warning("ExecutionAgent: position %s not found in state, cannot close.", position_id)
            return False # Added return statement to exit early if position is not found.

        logger.info("ExecutionAgent: closing %s (%s) — reason: %s", pos.symbol, position_id, reason)
        try:
            result = await self.client.close_position(position_id, pos.instrument_id)
        except Exception as exc:
            logger.error("ExecutionAgent: close failed for %s: %s", position_id, exc)
            await self.notification_agent.send_critical_error(
                f"Failed to close {pos.symbol} ({position_id}): {exc}"
            )
            return False

        # First try to get the actual close rate from the API response
        close_rate_raw = result.get("rate") or result.get("closeRate")
        if close_rate_raw is not None:
            close_rate = float(close_rate_raw)
        else:
            # The close-position response never includes a rate — fetch a live
            # quote instead of trusting a possibly-stale/zero current_rate.
            try:
                rates = await self.client.get_rates([pos.symbol])
                quote = rates.get(pos.symbol)
                live_rate = quote.get("lastExecution") or quote.get("bid") if quote else None
            except Exception as exc:
                logger.warning("get_rates failed for %s close: %s", pos.symbol, exc)
                live_rate = None
            close_rate = float(live_rate) if live_rate else pos.current_rate
        if pos.is_buy:
            pnl = (close_rate - pos.entry_rate) / pos.entry_rate * pos.amount_usd
        else:
            pnl = (pos.entry_rate - close_rate) / pos.entry_rate * pos.amount_usd

        duration = _utcnow() - pos.opened_at
        pos.current_rate = close_rate
        self.state.remove_position(position_id)

        from src.agents import risk_gate
        risk_gate.record_closed_pnl(self.state, pnl)
        self.state.save()  # persist position removal + P&L update immediately

        trade_log.log_close(
            position_id=position_id,
            symbol=pos.symbol,
            is_buy=pos.is_buy,
            amount_usd=pos.amount_usd,
            entry_rate=pos.entry_rate,
            close_rate=close_rate,
            pnl=pnl,
            duration_hours=duration.total_seconds() / 3600,
            reason=reason,
        )
        logger.info("ExecutionAgent: %s closed pnl=%.2f duration=%s", pos.symbol, pnl, duration)
        await self.notification_agent.send_position_closed(pos, pnl, duration)
        return True
