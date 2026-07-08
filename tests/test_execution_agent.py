"""
Tests for ExecutionAgent persistence behavior.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.execution_agent import ExecutionAgent, size_position
from src.agents import risk_gate
from src.core.state import Position, ProjectState
from src.core.thesis import TradingThesis


class _Client:
    async def close_position(self, position_id: str, instrument_id: str):
        return {"rate": 110.0}


@pytest.mark.asyncio
async def test_close_position_persists_pnl_before_save(monkeypatch):
    state = ProjectState()
    state.open_positions = [
        Position(
            position_id="p1",
            instrument_id="i1",
            symbol="BTC",
            is_buy=True,
            amount_usd=1000.0,
            entry_rate=100.0,
            stop_loss_pct=2.0,
            opened_at=datetime.now(timezone.utc),
        )
    ]

    events: list[str] = []

    def _record_closed_pnl(state_obj, pnl):
        events.append("pnl")
        state_obj.daily_pnl += pnl

    def _save():
        events.append("save")

    monkeypatch.setattr(risk_gate, "record_closed_pnl", _record_closed_pnl)
    monkeypatch.setattr(state, "save", _save)

    notification = AsyncMock()
    agent = ExecutionAgent(_Client(), state, notification)

    assert await agent.close_position("p1") is True
    assert events == ["pnl", "save"]


@pytest.mark.asyncio
async def test_close_position_updates_position_current_rate_before_notification():
    state = ProjectState()
    state.open_positions = [
        Position(
            position_id="p1",
            instrument_id="i1",
            symbol="BTC",
            is_buy=True,
            amount_usd=1000.0,
            entry_rate=100.0,
            stop_loss_pct=2.0,
            opened_at=datetime.now(timezone.utc),
            current_rate=99.0,
        )
    ]

    notification = AsyncMock()
    agent = ExecutionAgent(_Client(), state, notification)

    assert await agent.close_position("p1") is True
    closed_pos = notification.send_position_closed.call_args.args[0]
    assert closed_pos.current_rate == 110.0


def _make_thesis(symbol="AAPL", sl_atr_multiple=1.5) -> TradingThesis:
    return TradingThesis(
        symbol=symbol,
        action="buy",
        confidence=0.75,
        reasoning="Trend up (EMA50>EMA200). breakout confirmed with volume 2.0x.",
        signals_used=["trend_ema50_gt_ema200", "breakout", "relative_volume_confirmation"],
        suggested_stop_loss_atr_multiple=sl_atr_multiple,
        horizon_days=15,
        invalidation_condition="Daily close falls below EMA50 (trend break)",
    )


def test_size_position_is_risk_based_not_flat_percent(monkeypatch):
    """Notional must come from risk_amount / stop_distance, not a flat % of balance."""
    monkeypatch.setenv("RISK_PER_TRADE_PCT", "1.0")
    monkeypatch.setenv("MAX_POSITION_SIZE_PCT", "50.0")  # cap high enough to not bind
    monkeypatch.setenv("LEVERAGE", "1.0")
    import importlib
    import src.agents.execution_agent as ea
    importlib.reload(ea)

    thesis = _make_thesis(sl_atr_multiple=1.5)
    atr = 2.0  # stop distance = 1.5 * 2.0 = 3.0
    order = ea.size_position(thesis, "42", balance=1000.0, current_price=100.0, atr=atr)

    risk_amount = 1000.0 * 0.01     # $10
    stop_distance = 1.5 * atr       # 3.0
    expected_units = risk_amount / stop_distance
    expected_notional = expected_units * 100.0
    assert order.amount_usd == pytest.approx(expected_notional, abs=0.01)
    importlib.reload(ea)  # restore module-level constants for other tests


def test_size_position_caps_at_leveraged_notional(monkeypatch):
    """The notional cap scales with leverage — a tight ATR stop must not blow past it."""
    monkeypatch.setenv("RISK_PER_TRADE_PCT", "1.0")
    monkeypatch.setenv("MAX_POSITION_SIZE_PCT", "10.0")
    monkeypatch.setenv("LEVERAGE", "5.0")
    import importlib
    import src.agents.execution_agent as ea
    importlib.reload(ea)

    thesis = _make_thesis(sl_atr_multiple=1.5)
    atr = 0.01  # tiny stop distance -> huge uncapped notional
    order = ea.size_position(thesis, "42", balance=1000.0, current_price=100.0, atr=atr)

    max_notional = 1000.0 * (10.0 / 100.0 * 5.0)  # $500
    assert order.amount_usd == pytest.approx(max_notional, rel=1e-6)
    importlib.reload(ea)


def test_size_position_risk_pct_override_takes_precedence(monkeypatch):
    """The explicit risk_pct argument (set by the account drawdown hard stop
    in orchestrator.py) must override the configured RISK_PER_TRADE_PCT."""
    monkeypatch.setenv("RISK_PER_TRADE_PCT", "8.0")
    monkeypatch.setenv("MAX_POSITION_SIZE_PCT", "300.0")  # cap high enough to not bind
    monkeypatch.setenv("LEVERAGE", "1.0")
    import importlib
    import src.agents.execution_agent as ea
    importlib.reload(ea)

    thesis = _make_thesis(sl_atr_multiple=1.5)
    atr = 2.0
    stop_distance = 1.5 * atr

    # No override -> uses the configured 8%.
    order_default = ea.size_position(thesis, "42", balance=1000.0, current_price=100.0, atr=atr)
    assert order_default.amount_usd == pytest.approx((1000.0 * 0.08 / stop_distance) * 100.0, abs=0.01)

    # Override to 3% (simulating an active drawdown hard stop) -> smaller position.
    order_reduced = ea.size_position(
        thesis, "42", balance=1000.0, current_price=100.0, atr=atr, risk_pct=3.0
    )
    assert order_reduced.amount_usd == pytest.approx((1000.0 * 0.03 / stop_distance) * 100.0, abs=0.01)
    assert order_reduced.amount_usd < order_default.amount_usd
    importlib.reload(ea)


@pytest.mark.asyncio
async def test_execute_sends_margin_not_notional_to_broker(monkeypatch):
    """The broker order must receive amount=notional/leverage and the real
    leverage flag — not the full notional as cash."""
    monkeypatch.setenv("LEVERAGE", "5.0")
    import importlib
    import src.agents.execution_agent as ea
    importlib.reload(ea)

    captured = {}

    class _OpenClient:
        async def open_position(self, **kwargs):
            captured.update(kwargs)
            return {"positionId": "9001", "openRate": 100.0}

    thesis = _make_thesis()
    order = ea.SizedOrder(
        thesis=thesis, instrument_id="42", amount_usd=500.0,
        stop_loss_pct=2.0, current_price=100.0, atr=2.0, leverage=5.0,
    )

    state = ProjectState()
    notification = AsyncMock()
    agent = ea.ExecutionAgent(_OpenClient(), state, notification)

    assert await agent.execute(order) is True
    assert captured["leverage"] == 5
    assert captured["amount_usd"] == pytest.approx(100.0)  # 500 / 5
    assert state.open_positions[0].amount_usd == pytest.approx(500.0)  # notional preserved
    assert state.open_positions[0].leverage == pytest.approx(5.0)
    importlib.reload(ea)
