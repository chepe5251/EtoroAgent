"""
Tests for PositionReviewAgent (100% deterministic, no LLM).
Focus: bar-count hard exit limit, trend_break (close < EMA50) exit logic.

The hard exit is measured in trading-day BARS held (matching
src/backtest/engine.py's max_hold_days convention), computed by counting
candles dated after the position's opened_at — not raw calendar days. Tests
build candle sets with explicit dates to control bars_held precisely.
"""
import sys
import importlib
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.position_review_agent import PositionReviewAgent, _bars_held
from src.core.state import Position, ProjectState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _make_position(
    symbol: str = "BTC",
    is_buy: bool = True,
    opened_at: datetime | None = None,
    horizon_days: int = 10,
    invalidation_condition: str = "Daily close falls below EMA50 (trend break)",
    entry_rate: float = 50000.0,
    atr: float = 1200.0,
) -> Position:
    if opened_at is None:
        opened_at = _utcnow() - timedelta(days=5)
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
    """No dates — bars_held will be 0 for all of these (no entry-relative timing)."""
    return [{"open": c, "high": c, "low": c, "close": c, "volume": 1_000_000} for c in closes]


def _dated_candles(closes: list[float], dates: list[datetime]) -> list[dict]:
    assert len(closes) == len(dates)
    return [
        {"open": c, "high": c, "low": c, "close": c, "volume": 1_000_000,
         "fromDate": d.strftime("%Y-%m-%dT00:00:00Z")}
        for c, d in zip(closes, dates)
    ]


def _candles_with_bars_held(n_total: int, bars_after_entry: int, opened_at: datetime,
                             closes: list[float] | None = None) -> list[dict]:
    """
    Build n_total daily candles where exactly `bars_after_entry` of them are
    dated strictly after `opened_at` (the rest dated at/before it) — giving
    precise control over what _bars_held() will compute.
    """
    if closes is None:
        closes = _rising_closes(n=n_total)
    n_before = n_total - bars_after_entry
    dates = [opened_at - timedelta(days=(n_before - i)) for i in range(n_before)]
    dates += [opened_at + timedelta(days=i + 1) for i in range(bars_after_entry)]
    return _dated_candles(closes, dates)


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


# ── _bars_held() — pure function ───────────────────────────────────────────────

def test_bars_held_counts_candles_after_opened_at():
    opened_at = _utcnow() - timedelta(days=10)
    candles = _candles_with_bars_held(n_total=60, bars_after_entry=7, opened_at=opened_at)
    assert _bars_held(candles, opened_at) == 7


def test_bars_held_ignores_candles_before_or_at_opened_at():
    opened_at = _utcnow()
    dates = [opened_at - timedelta(days=1), opened_at, opened_at + timedelta(days=1)]
    candles = _dated_candles([100.0, 101.0, 102.0], dates)
    assert _bars_held(candles, opened_at) == 1


def test_bars_held_ignores_undated_candles():
    candles = _candles_from_closes(_rising_closes(n=10))
    assert _bars_held(candles, _utcnow()) == 0


# ── Hard exit is enforced by bar count, not calendar days ─────────────────────

@pytest.mark.asyncio
async def test_hard_exit_at_20_bars_closes_position():
    """A position with exactly 20 bars held since entry must be closed —
    without even running the trend-break technical check."""
    opened_at = _utcnow() - timedelta(days=40)
    pos = _make_position(opened_at=opened_at)
    state = _make_state(pos)
    candles = _candles_with_bars_held(n_total=60, bars_after_entry=20, opened_at=opened_at)
    agent = _make_agent(state, candles=candles)

    with patch.object(agent, "_technical_review") as mock_review:
        await agent._review_position(pos)

    mock_review.assert_not_called()
    agent.execution_agent.close_position.assert_called_once()


@pytest.mark.asyncio
async def test_hard_exit_close_reason_mentions_bar_limit():
    """Close reason must mention the bar limit for auditing."""
    opened_at = _utcnow() - timedelta(days=40)
    pos = _make_position(opened_at=opened_at)
    state = _make_state(pos)
    candles = _candles_with_bars_held(n_total=60, bars_after_entry=22, opened_at=opened_at)
    agent = _make_agent(state, candles=candles)

    with patch.object(agent, "_technical_review"):
        await agent._review_position(pos)

    call_args = agent.execution_agent.close_position.call_args
    reason = call_args[1].get("reason", "") or (call_args[0][1] if len(call_args[0]) > 1 else "")
    assert "20" in reason or "bar" in reason.lower()


@pytest.mark.asyncio
async def test_under_bar_limit_does_not_hard_exit():
    """19 bars held (one short of the 20-bar limit) must NOT force-close —
    the trend-break check still runs."""
    opened_at = _utcnow() - timedelta(days=40)
    pos = _make_position(opened_at=opened_at)
    state = _make_state(pos)
    # Rising closes -> trend intact -> technical review holds, not exits.
    candles = _candles_with_bars_held(
        n_total=60, bars_after_entry=19, opened_at=opened_at,
        closes=_rising_closes(n=60),
    )
    agent = _make_agent(state, candles=candles)

    await agent._review_position(pos)

    agent.execution_agent.close_position.assert_not_called()


# ── Deterministic trend-break review ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_close_below_ema50_triggers_exit():
    """A falling price series (close < EMA50) must trigger an exit."""
    opened_at = _utcnow() - timedelta(days=5)
    pos = _make_position(opened_at=opened_at)
    state = _make_state(pos)
    falling = list(reversed(_rising_closes(n=60, start=100.0, end=150.0)))
    candles = _candles_with_bars_held(n_total=60, bars_after_entry=5, opened_at=opened_at, closes=falling)
    agent = _make_agent(state, candles=candles)

    await agent._review_position(pos)

    agent.execution_agent.close_position.assert_called_once()


@pytest.mark.asyncio
async def test_close_above_ema50_holds():
    """A steadily rising price series (close >= EMA50) must NOT trigger an exit."""
    opened_at = _utcnow() - timedelta(days=5)
    pos = _make_position(opened_at=opened_at)
    state = _make_state(pos)
    rising = _rising_closes(n=60, start=100.0, end=150.0)
    candles = _candles_with_bars_held(n_total=60, bars_after_entry=5, opened_at=opened_at, closes=rising)
    agent = _make_agent(state, candles=candles)

    await agent._review_position(pos)

    agent.execution_agent.close_position.assert_not_called()


@pytest.mark.asyncio
async def test_insufficient_candle_history_skips_action():
    """Fewer than EMA_PERIOD candles -> no verdict -> no action."""
    pos = _make_position()
    state = _make_state(pos)
    agent = _make_agent(state, candles=_candles_from_closes(_rising_closes(n=10)))

    await agent._review_position(pos)

    agent.execution_agent.close_position.assert_not_called()


@pytest.mark.asyncio
async def test_candle_fetch_failure_skips_action():
    """If the candle fetch raises, no verdict -> no action, no crash."""
    pos = _make_position()
    state = _make_state(pos)
    agent = _make_agent(state)
    agent.execution_agent.client.get_candles = AsyncMock(side_effect=Exception("network error"))

    await agent._review_position(pos)

    agent.execution_agent.close_position.assert_not_called()


@pytest.mark.asyncio
async def test_close_does_not_log_success_when_execution_fails(caplog):
    pos = _make_position()
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
    pos1 = _make_position("BTC", opened_at=_utcnow() - timedelta(days=5))
    pos2 = _make_position("ETH", opened_at=_utcnow() - timedelta(days=3))
    state = _make_state(pos1, pos2)
    agent = _make_agent(state, candles=_candles_from_closes(_rising_closes(n=60)))

    reviewed = []

    def track_review(candles):
        reviewed.append(True)
        return None

    with patch.object(agent, "_technical_review", side_effect=track_review):
        await agent.review_all()

    assert len(reviewed) == 2


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
    """get_hard_exit_days() (the bar-count threshold) reads SWING_HARD_EXIT_DAYS."""
    monkeypatch.setenv("SWING_HARD_EXIT_DAYS", "25")
    import src.core.state as state_module

    reloaded = importlib.reload(state_module)
    assert reloaded.get_hard_exit_days() == 25
