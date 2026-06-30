"""
Tests for RiskAgent.
Run with: pytest tests/test_risk_agent.py -v
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("ETORO_PUBLIC_API_KEY", "test")
os.environ.setdefault("ETORO_USER_KEY", "test")
os.environ.setdefault("ETORO_MODE", "demo")
os.environ.setdefault("DAILY_LOSS_LIMIT_PCT", "3.0")

from src.agents.risk_agent import RiskAgent
from src.core.state import Position, ProjectState, Signal


@pytest.fixture
def state():
    return ProjectState()


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.get_rates = AsyncMock(return_value={})
    return client


@pytest.fixture
def risk_agent(state, mock_client):
    return RiskAgent(state, mock_client)


# ---------------------------------------------------------------------------
# check_risk tests
# ---------------------------------------------------------------------------

def test_check_risk_allows_when_no_loss(risk_agent, state):
    signal = Signal(symbol="BTC", action="BUY", confidence=0.8, reasoning="test")
    state.daily_pnl = 0.0
    allowed, reason = risk_agent.check_risk(signal, balance=10000.0)
    assert allowed is True
    assert reason == "ok"


def test_check_risk_blocks_when_daily_loss_exceeded(risk_agent, state):
    signal = Signal(symbol="BTC", action="BUY", confidence=0.8, reasoning="test")
    state.daily_pnl = -310.0  # -310 on 10000 = 3.1% > 3%
    allowed, reason = risk_agent.check_risk(signal, balance=10000.0)
    assert allowed is False
    assert "Daily loss limit" in reason


def test_check_risk_blocks_when_already_blocked(risk_agent, state):
    signal = Signal(symbol="ETH", action="BUY", confidence=0.9, reasoning="test")
    state.is_risk_blocked = True
    state.risk_block_reason = "test block"
    allowed, reason = risk_agent.check_risk(signal, balance=10000.0)
    assert allowed is False
    assert "Risk blocked" in reason


def test_check_risk_exactly_at_limit(risk_agent, state):
    # exactly at 3% should block
    signal = Signal(symbol="BTC", action="BUY", confidence=0.8, reasoning="test")
    state.daily_pnl = -300.0  # exactly 3% of 10000
    allowed, _ = risk_agent.check_risk(signal, balance=10000.0)
    assert allowed is False


def test_check_risk_just_below_limit(risk_agent, state):
    signal = Signal(symbol="BTC", action="BUY", confidence=0.8, reasoning="test")
    state.daily_pnl = -299.0  # 2.99% of 10000 — should be allowed
    allowed, _ = risk_agent.check_risk(signal, balance=10000.0)
    assert allowed is True


# ---------------------------------------------------------------------------
# record_closed_trade tests
# ---------------------------------------------------------------------------

def test_record_closed_trade_profit(risk_agent, state):
    risk_agent.record_closed_trade(50.0)
    assert state.daily_pnl == 50.0


def test_record_closed_trade_loss(risk_agent, state):
    risk_agent.record_closed_trade(-100.0)
    assert state.daily_pnl == -100.0


def test_record_closed_trade_cumulative(risk_agent, state):
    risk_agent.record_closed_trade(50.0)
    risk_agent.record_closed_trade(-80.0)
    assert state.daily_pnl == pytest.approx(-30.0)


# ---------------------------------------------------------------------------
# daily reset tests
# ---------------------------------------------------------------------------

def test_daily_reset_clears_block(state):
    state.daily_pnl = -500.0
    state.is_risk_blocked = True
    state.risk_block_reason = "test"
    state._daily_reset_date = "2020-01-01"  # force stale date
    did_reset = state.reset_daily_if_needed()
    assert did_reset is True
    assert state.daily_pnl == 0.0
    assert state.is_risk_blocked is False
    assert state.risk_block_reason == ""


def test_daily_reset_not_triggered_same_day(state):
    # reset date is today, so no reset should happen
    did_reset = state.reset_daily_if_needed()
    assert did_reset is False


# ---------------------------------------------------------------------------
# adjust_trailing_stops tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adjust_trailing_stops_no_positions(risk_agent, state):
    # Should return immediately without calling get_rates
    await risk_agent.adjust_trailing_stops()
    risk_agent.client.get_rates.assert_not_called()


@pytest.mark.asyncio
async def test_adjust_trailing_stops_tightens_long(risk_agent, state, mock_client):
    entry_rate = 50000.0
    atr = 500.0
    current = entry_rate + 1.5 * atr  # moved up 1.5x ATR, well above threshold

    pos = Position(
        position_id="pos-1",
        instrument_id="42",
        symbol="BTC",
        is_buy=True,
        amount_usd=200.0,
        entry_rate=entry_rate,
        stop_loss_pct=2.0,
        opened_at=datetime.now(timezone.utc),
        current_rate=entry_rate,
        atr=atr,
    )
    state.open_positions.append(pos)
    mock_client.get_rates = AsyncMock(
        return_value={"BTC": {"close": current}}
    )

    original_sl = pos.stop_loss_pct
    await risk_agent.adjust_trailing_stops()
    # Stop-loss should have tightened (smaller % distance)
    assert pos.stop_loss_pct < original_sl
