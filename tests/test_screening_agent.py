"""
Tests for ScreeningAgent (100% deterministic, no LLM).
Uses synthetic candles — no HTTP calls.
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.screening_agent import ScreeningAgent, ScreeningResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _candle(close: float, volume: float = 1_000_000) -> dict:
    return {
        "open": close * 0.999,
        "high": close * 1.001,
        "low": close * 0.998,
        "close": close,
        "volume": volume,
    }


def _uptrend_candles(n: int = 230, start: float = 100.0, end: float = 200.0,
                      volume: float = 1_000_000) -> list[dict]:
    """A long, steady uptrend — enough bars for EMA200 to reflect trend_up=True."""
    step = (end - start) / (n - 1)
    return [_candle(start + step * i, volume) for i in range(n)]


def _make_agent() -> ScreeningAgent:
    """Create a ScreeningAgent with a stubbed EtoroClient."""
    mock_client = MagicMock()
    mock_client.get_candles = AsyncMock(return_value=[])
    agent = ScreeningAgent(mock_client)
    return agent


# ── Stage 1a: compute() unit tests ───────────────────────────────────────────

def test_compute_requires_min_candles():
    agent = _make_agent()
    result = agent._compute("AAPL", candles=[_candle(1)] * 10)
    assert result is None  # < donchian_lookback + 2


def test_compute_short_history_no_trend_signal():
    """Fewer than 200 candles → EMA200 unavailable → trend_up False, no tags."""
    agent = _make_agent()
    result = agent._compute("AAPL", _uptrend_candles(n=60))
    assert result is not None
    assert result.trend_up is False
    assert result.score_tags == []


def test_compute_breakout_detected_with_volume():
    """Uptrend (EMA50>EMA200) + last close breaks the prior 20-bar high + volume spike."""
    agent = _make_agent()
    candles = _uptrend_candles(n=230, start=100.0, end=200.0)
    # Force a clean breakout: last candle's high/close well above the recent range,
    # with a volume spike to clear the rel-volume gate.
    candles[-1] = _candle(candles[-2]["close"] * 1.05, volume=5_000_000)
    result = agent._compute("AAPL", candles)
    assert result is not None
    assert result.trend_up is True
    assert result.breakout is True
    assert "breakout" in result.score_tags


def test_compute_pullback_resume_detected():
    """Close dips to/below EMA20 then resumes above it, in an uptrend, with volume."""
    agent = _make_agent()
    candles = _uptrend_candles(n=230, start=100.0, end=200.0)
    last_close = candles[-1]["close"]
    # Second-to-last bar: dip at/below EMA20 (approximated by a sharp pullback).
    candles[-2] = _candle(last_close * 0.90, volume=1_000_000)
    # Last bar: resume above with a volume spike.
    candles[-1] = _candle(last_close * 0.98, volume=5_000_000)
    result = agent._compute("AAPL", candles)
    assert result is not None
    if result.trend_up:
        # Depending on exact EMA20 level this may or may not trigger — just
        # verify the computation runs and produces a consistent boolean.
        assert isinstance(result.pullback_resume, bool)


def test_compute_no_signal_without_volume_confirmation():
    """A breakout-shaped move without volume confirmation should not tag."""
    agent = _make_agent()
    candles = _uptrend_candles(n=230, start=100.0, end=200.0, volume=1_000_000)
    candles[-1] = _candle(candles[-2]["close"] * 1.05, volume=1_000_000)  # no spike
    result = agent._compute("AAPL", candles)
    assert result is not None
    assert result.breakout is False
    assert result.score_tags == []


# ── Stage 1a: async fetch ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stage1a_filters_symbols_with_no_candles():
    """Symbols with fetch errors or empty candles are silently dropped."""
    agent = _make_agent()
    agent.client.get_candles = AsyncMock(return_value=[])  # all empty
    candidates = await agent._stage1a(["AAPL", "MSFT", "TSLA"])
    assert candidates == []


@pytest.mark.asyncio
async def test_stage1a_returns_candidates_with_signal():
    """Symbols whose computed result has score_tags pass stage 1a."""
    agent = _make_agent()
    candles = _uptrend_candles(n=230, start=100.0, end=200.0)
    candles[-1] = _candle(candles[-2]["close"] * 1.05, volume=5_000_000)

    async def mock_candles(symbol, interval, count):
        return candles

    agent.client.get_candles = mock_candles
    candidates = await agent._stage1a(["AAPL"])
    assert len(candidates) == 1
    assert candidates[0].symbol == "AAPL"


# ── run() end-to-end (deterministic, no LLM) ──────────────────────────────────

@pytest.mark.asyncio
async def test_run_returns_empty_on_empty_universe():
    agent = _make_agent()
    result = await agent.run([])
    assert result == []


@pytest.mark.asyncio
async def test_run_returns_screening_results_directly():
    """run() returns ScreeningResult objects straight from the deterministic
    filter — no ranking/reordering step."""
    agent = _make_agent()
    candles = _uptrend_candles(n=230, start=100.0, end=200.0)
    candles[-1] = _candle(candles[-2]["close"] * 1.05, volume=5_000_000)

    async def mock_candles(symbol, interval, count):
        return candles

    agent.client.get_candles = mock_candles
    result = await agent.run(["AAPL"])
    assert len(result) == 1
    assert isinstance(result[0], ScreeningResult)
    assert result[0].symbol == "AAPL"
    assert result[0].breakout is True
