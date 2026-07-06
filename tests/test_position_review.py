"""
Tests for PositionReviewAgent (100% deterministic, no LLM).
Focus: 20-day hard exit limit, trend_break (close < EMA50) exit logic.
"""
import sys
import importlib
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.position_review_agent import PositionReviewAgent
from src.core.state import Position, ProjectState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _make_position(
    symbol: str = "BTC",
    is_buy: bool = True,
    days_open: int = 5,
    horizon_days: int = 10,
    invalidation_condition: str = "Daily close falls below EMA50 (trend break)",
    entry_rate: float = 50000.0,
    atr: float = 1200.0,
) -> Position:
    opened_at = _utcnow() - timedelta(days=days_open)
    return Position(
        position_id=f"pos_{symbol}",
        instrument_id=f"instr_{symbol}",
        symbol=symbol,
        is_buy=is_buy,
        amount_usd=1000.0,
        entry_rate=entry_rate,
        stop_loss_pct=2.0,
        opened_at=opened_at,
        atr=atr,
        horizon_days=horizon_days,
        invalidation_condition=invalidation_condition,
    )


def _make_state(*positions: Position) -> ProjectState:
    state = ProjectState()
    state.open_positions = list(positions)
    return state


def _rising_closes(n: int = 60, start: float = 100.0, end: float = 150.0) -> list[float]:
    step = (end - start) / (n - 1)
    return [start + step * i for i in range(n)]


def _candles_from_closes(closes: list[float]) -> list[dict]:
    return [{"open": c, "high": c, "low": c, "close": c, "volume": 1_000_000} for c in closes]


def _make_agent(state: ProjectState, candles: list[dict] | None = None) -> PositionReviewAgent:
    mock_client = MagicMock()
    mock_client.get_candles = AsyncMock(return_value=candles if candles is not None else [])
    mock_exec = MagicMock()
    mock_exec.client = mock_client
    mock_exec.close_position = AsyncMock(return_value=True)
    mock_notif = MagicMock()
    mock_notif.send_critical_error = AsyncMock()
    return PositionReviewAgent(
        execution_agent=mock_exec,
        notification_agent=mock_notif,
        state=state,
    )


# ── Position.is_past_hard_exit ────────────────────────────────────────────────

def test_hard_exit_flag_true_at_20_days():
    pos = _make_position(days_open=20)
    assert pos.is_past_hard_exit is True


def test_hard_exit_flag_true_at_21_days():
    pos = _make_position(days_open=21)
    assert pos.is_past_hard_exit is True


def test_hard_exit_flag_false_at_19_days():
    pos = _make_position(days_open=19)
    assert pos.is_past_hard_exit is False


def test_days_open_computed_correctly():
    pos = _make_position(days_open=7)
    assert 6 <= pos.days_open <= 8  # allow 1-day rounding


# ── Hard exit is enforced (no technical check needed) ────────────────────────

@pytest.mark.asyncio
async def test_hard_exit_at_20_days_closes_position():
    """A 20-day-old position must be closed without even checking price data."""
    pos = _make_position(days_open=20)
    state = _make_state(pos)
    agent = _make_agent(state)

    with patch.object(agent, "_technical_review", new=AsyncMock()) as mock_review:
        await agent._review_position(pos)

    mock_review.assert_not_called()
    agent.execution_agent.close_position.assert_called_once()


@pytest.mark.asyncio
async def test_hard_exit_close_reason_mentions_20_days():
    """Close reason must mention the 20-day limit for auditing."""
    pos = _make_position(days_open=22)
    state = _make_state(pos)
    agent = _make_agent(state)

    with patch.object(agent, "_technical_review", new=AsyncMock()):
        await agent._review_position(pos)

    call_args = agent.execution_agent.close_position.call_args
    reason = call_args[1].get("reason", "") or (call_args[0][1] if len(call_args[0]) > 1 else "")
    assert "20" in reason or "hard" in reason.lower()


# ── Deterministic trend-break review ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_close_below_ema50_triggers_exit():
    """A falling price series (close < EMA50) must trigger an exit."""
    pos = _make_position(days_open=5)
    state = _make_state(pos)
    falling = list(reversed(_rising_closes(n=60, start=100.0, end=150.0)))
    agent = _make_agent(state, candles=_candles_from_closes(falling))

    await agent._review_position(pos)

    agent.execution_agent.close_position.assert_called_once()


@pytest.mark.asyncio
async def test_close_above_ema50_holds():
    """A steadily rising price series (close >= EMA50) must NOT trigger an exit."""
    pos = _make_position(days_open=5)
    state = _make_state(pos)
    rising = _rising_closes(n=60, start=100.0, end=150.0)
    agent = _make_agent(state, candles=_candles_from_closes(rising))

    await agent._review_position(pos)

    agent.execution_agent.close_position.assert_not_called()


@pytest.mark.asyncio
async def test_insufficient_candle_history_skips_action():
    """Fewer than EMA_PERIOD candles -> no verdict -> no action."""
    pos = _make_position(days_open=5)
    state = _make_state(pos)
    agent = _make_agent(state, candles=_candles_from_closes(_rising_closes(n=10)))

    await agent._review_position(pos)

    agent.execution_agent.close_position.assert_not_called()


@pytest.mark.asyncio
async def test_candle_fetch_failure_skips_action():
    """If the candle fetch raises, no verdict -> no action, no crash."""
    pos = _make_position(days_open=5)
    state = _make_state(pos)
    agent = _make_agent(state)
    agent.execution_agent.client.get_candles = AsyncMock(side_effect=Exception("network error"))

    await agent._review_position(pos)

    agent.execution_agent.close_position.assert_not_called()


@pytest.mark.asyncio
async def test_close_does_not_log_success_when_execution_fails(caplog):
    pos = _make_position(days_open=20)
    state = _make_state(pos)
    agent = _make_agent(state)
    agent.execution_agent.close_position = AsyncMock(return_value=False)

    with caplog.at_level(logging.INFO):
        await agent._close(pos, reason="test failure")

    assert "closed BTC" not in caplog.text


# ── review_all ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_review_all_with_no_positions_does_nothing():
    state = _make_state()
    agent = _make_agent(state)
    await agent.review_all()
    agent.execution_agent.close_position.assert_not_called()


@pytest.mark.asyncio
async def test_review_all_reviews_each_position():
    pos1 = _make_position("BTC", days_open=5)
    pos2 = _make_position("ETH", days_open=3)
    state = _make_state(pos1, pos2)
    agent = _make_agent(state)

    reviewed = []

    async def track_review(pos):
        reviewed.append(pos.symbol)
        return None

    with patch.object(agent, "_technical_review", new=AsyncMock(side_effect=track_review)):
        await agent.review_all()

    assert set(reviewed) == {"BTC", "ETH"}


# ── RiskGate horizon_days rule ─────────────────────────────────────────────────

def test_risk_gate_rejects_horizon_too_short():
    """horizon_days < 5 should be rejected by risk gate."""
    from src.agents import risk_gate
    from src.core.thesis import TradingThesis

    thesis = TradingThesis(
        symbol="BTC",
        action="buy",
        confidence=0.80,
        reasoning="RSI oversold at 28, MACD histogram turned positive, volume 2x average.",
        signals_used=["indicators_full_analysis", "cryptopanic_get_sentiment_summary"],
        suggested_stop_loss_atr_multiple=1.5,
        horizon_days=3,  # too short
        invalidation_condition="EMA50 break",
    )
    state = ProjectState()
    approved, reason = risk_gate.validate(thesis, state, balance=10000.0)
    assert approved is False
    assert "horizon" in reason or "3" in reason


def test_risk_gate_rejects_horizon_too_long():
    """horizon_days > 20 should be rejected by risk gate."""
    from src.agents import risk_gate
    from src.core.thesis import TradingThesis

    thesis = TradingThesis(
        symbol="BTC",
        action="buy",
        confidence=0.80,
        reasoning="RSI oversold at 28, MACD histogram turned positive, volume 2x average.",
        signals_used=["indicators_full_analysis", "cryptopanic_get_sentiment_summary"],
        suggested_stop_loss_atr_multiple=1.5,
        horizon_days=25,  # too long
        invalidation_condition="EMA50 break",
    )
    state = ProjectState()
    approved, reason = risk_gate.validate(thesis, state, balance=10000.0)
    assert approved is False
    assert "horizon" in reason or "25" in reason


def test_risk_gate_approves_valid_horizon():
    """horizon_days=10 (default) should pass risk gate."""
    from src.agents import risk_gate
    from src.core.thesis import TradingThesis

    thesis = TradingThesis(
        symbol="BTC",
        action="buy",
        confidence=0.80,
        reasoning="RSI oversold at 28, MACD histogram turned positive, volume 2x average.",
        signals_used=["indicators_full_analysis", "cryptopanic_get_sentiment_summary"],
        suggested_stop_loss_atr_multiple=1.5,
        horizon_days=10,
        invalidation_condition="Daily close below EMA50",
    )
    state = ProjectState()
    approved, _ = risk_gate.validate(thesis, state, balance=10000.0)
    assert approved is True


def test_hard_exit_days_is_configurable(monkeypatch):
    monkeypatch.setenv("SWING_HARD_EXIT_DAYS", "25")
    import src.core.state as state_module

    reloaded = importlib.reload(state_module)
    pos = reloaded.Position(
        position_id="pos_cfg",
        instrument_id="instr_cfg",
        symbol="BTC",
        is_buy=True,
        amount_usd=1000.0,
        entry_rate=50000.0,
        stop_loss_pct=2.0,
        opened_at=_utcnow() - timedelta(days=20),
    )

    assert reloaded.get_hard_exit_days() == 25
    assert pos.is_past_hard_exit is False
