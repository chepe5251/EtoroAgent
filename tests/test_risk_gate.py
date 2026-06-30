"""
Tests for risk_gate — 100% deterministic, zero LLM mocks needed.
Every test case should be obvious from the production code's rules.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("ETORO_PUBLIC_API_KEY", "test")
os.environ.setdefault("ETORO_USER_KEY", "test")
os.environ.setdefault("ETORO_MODE", "demo")
os.environ.setdefault("MAX_OPEN_POSITIONS", "3")
os.environ.setdefault("DAILY_LOSS_LIMIT_PCT", "3.0")
os.environ.setdefault("MIN_SIGNAL_CONFIDENCE", "0.65")
os.environ.setdefault("MIN_SIGNALS_REQUIRED", "2")

from src.agents import risk_gate
from src.core.state import Position, ProjectState
from src.core.thesis import TradingThesis

from datetime import datetime, timezone


def _make_thesis(**kwargs) -> TradingThesis:
    """Helper: build a valid thesis with optional overrides."""
    defaults = {
        "symbol": "BTC",
        "action": "buy",
        "confidence": 0.75,
        "reasoning": "RSI oversold at 28, MACD histogram turned positive, CryptoPanic bullish_count=12.",
        "signals_used": ["indicators_full_analysis", "cryptopanic_get_sentiment_summary"],
        "suggested_stop_loss_atr_multiple": 1.5,
    }
    defaults.update(kwargs)
    return TradingThesis(**defaults)


def _make_state(**kwargs) -> ProjectState:
    state = ProjectState()
    for k, v in kwargs.items():
        setattr(state, k, v)
    return state


def _now():
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Tests that should PASS (approved=True)
# ─────────────────────────────────────────────────────────────────────────────

def test_valid_buy_thesis_approved():
    thesis = _make_thesis()
    state = _make_state()
    approved, reason = risk_gate.validate(thesis, state, balance=10000.0)
    assert approved is True
    assert reason == "approved"


def test_valid_sell_thesis_approved():
    thesis = _make_thesis(action="sell")
    state = _make_state()
    approved, _ = risk_gate.validate(thesis, state, balance=10000.0)
    assert approved is True


def test_confidence_exactly_at_threshold():
    # 0.65 is the minimum allowed confidence
    thesis = _make_thesis(confidence=0.65)
    state = _make_state()
    approved, _ = risk_gate.validate(thesis, state, balance=10000.0)
    assert approved is True


def test_two_positions_open_third_allowed():
    pos1 = Position("p1", "i1", "ETH", True, 200, 2000, 2.0, _now())
    pos2 = Position("p2", "i2", "AAPL", True, 200, 150, 2.0, _now())
    state = _make_state(open_positions=[pos1, pos2])
    thesis = _make_thesis(symbol="BTC")  # third symbol, not yet open
    approved, _ = risk_gate.validate(thesis, state, balance=10000.0)
    assert approved is True


# ─────────────────────────────────────────────────────────────────────────────
# Tests that should FAIL (approved=False)
# ─────────────────────────────────────────────────────────────────────────────

def test_hold_action_rejected():
    thesis = _make_thesis(action="hold")
    state = _make_state()
    approved, reason = risk_gate.validate(thesis, state, balance=10000.0)
    assert approved is False
    assert "hold" in reason


def test_low_confidence_rejected():
    thesis = _make_thesis(confidence=0.64)
    state = _make_state()
    approved, reason = risk_gate.validate(thesis, state, balance=10000.0)
    assert approved is False
    assert "confidence" in reason


def test_single_signal_rejected():
    thesis = _make_thesis(signals_used=["indicators_full_analysis"])  # only 1
    state = _make_state()
    approved, reason = risk_gate.validate(thesis, state, balance=10000.0)
    assert approved is False
    assert "signal" in reason


def test_no_signals_rejected():
    thesis = _make_thesis(signals_used=[])
    state = _make_state()
    approved, reason = risk_gate.validate(thesis, state, balance=10000.0)
    assert approved is False


def test_empty_reasoning_rejected():
    thesis = _make_thesis(reasoning="")
    state = _make_state()
    approved, reason = risk_gate.validate(thesis, state, balance=10000.0)
    assert approved is False
    assert "reasoning" in reason


def test_too_short_reasoning_rejected():
    thesis = _make_thesis(reasoning="RSI low, buy it.")  # < 50 chars
    state = _make_state()
    approved, reason = risk_gate.validate(thesis, state, balance=10000.0)
    assert approved is False
    assert "reasoning" in reason


def test_risk_blocked_state_rejected():
    state = _make_state(is_risk_blocked=True, risk_block_reason="daily loss exceeded")
    thesis = _make_thesis()
    approved, reason = risk_gate.validate(thesis, state, balance=10000.0)
    assert approved is False
    assert "block" in reason.lower()


def test_max_positions_reached_rejected():
    positions = [
        Position(f"p{i}", f"i{i}", f"SYM{i}", True, 200, 100, 2.0, _now())
        for i in range(3)  # MAX_OPEN_POSITIONS = 3
    ]
    state = _make_state(open_positions=positions)
    thesis = _make_thesis(symbol="BTC")
    approved, reason = risk_gate.validate(thesis, state, balance=10000.0)
    assert approved is False
    assert "max" in reason.lower() or "position" in reason.lower()


def test_duplicate_symbol_rejected():
    existing = Position("p1", "i1", "BTC", True, 200, 50000, 2.0, _now())
    state = _make_state(open_positions=[existing])
    thesis = _make_thesis(symbol="BTC")
    approved, reason = risk_gate.validate(thesis, state, balance=10000.0)
    assert approved is False
    assert "BTC" in reason or "already" in reason


def test_daily_loss_limit_exceeded_rejected():
    state = _make_state(daily_pnl=-310.0)  # -310 on 10000 = 3.1% > 3%
    thesis = _make_thesis()
    approved, reason = risk_gate.validate(thesis, state, balance=10000.0)
    assert approved is False
    assert "loss" in reason.lower() or "limit" in reason.lower()


def test_daily_loss_exactly_at_limit_rejected():
    state = _make_state(daily_pnl=-300.0)  # exactly 3.0% of 10000
    thesis = _make_thesis()
    approved, reason = risk_gate.validate(thesis, state, balance=10000.0)
    assert approved is False


def test_daily_loss_just_under_limit_approved():
    state = _make_state(daily_pnl=-299.0)  # 2.99% — under the 3% limit
    thesis = _make_thesis()
    approved, _ = risk_gate.validate(thesis, state, balance=10000.0)
    assert approved is True


def test_daily_loss_blocks_state_after_validation():
    state = _make_state(daily_pnl=-300.0)
    thesis = _make_thesis()
    risk_gate.validate(thesis, state, balance=10000.0)
    assert state.is_risk_blocked is True


# ─────────────────────────────────────────────────────────────────────────────
# record_closed_pnl
# ─────────────────────────────────────────────────────────────────────────────

def test_record_pnl_profit():
    state = _make_state()
    risk_gate.record_closed_pnl(state, 50.0)
    assert state.daily_pnl == pytest.approx(50.0)


def test_record_pnl_loss():
    state = _make_state()
    risk_gate.record_closed_pnl(state, -80.0)
    assert state.daily_pnl == pytest.approx(-80.0)


def test_record_pnl_cumulative():
    state = _make_state()
    risk_gate.record_closed_pnl(state, 50.0)
    risk_gate.record_closed_pnl(state, -120.0)
    assert state.daily_pnl == pytest.approx(-70.0)


# ─────────────────────────────────────────────────────────────────────────────
# TradingThesis schema
# ─────────────────────────────────────────────────────────────────────────────

def test_thesis_from_dict_normalises_action():
    d = {"symbol": "eth", "action": "BUY", "confidence": 0.8,
         "reasoning": "test", "signals_used": ["a", "b"]}
    t = TradingThesis.from_dict(d)
    assert t.action == "buy"
    assert t.symbol == "ETH"


def test_thesis_from_dict_unknown_action_defaults_hold():
    d = {"symbol": "BTC", "action": "MAYBE", "confidence": 0.8,
         "reasoning": "test", "signals_used": ["a", "b"]}
    t = TradingThesis.from_dict(d)
    assert t.action == "hold"


def test_thesis_from_json_strips_fence():
    json_str = '```json\n{"symbol":"BTC","action":"buy","confidence":0.7,' \
               '"reasoning":"test reason here","signals_used":["a","b"],' \
               '"suggested_stop_loss_atr_multiple":1.5}\n```'
    t = TradingThesis.from_json(json_str)
    assert t.symbol == "BTC"
    assert t.action == "buy"


def test_thesis_is_actionable():
    assert _make_thesis(action="buy").is_actionable() is True
    assert _make_thesis(action="sell").is_actionable() is True
    assert _make_thesis(action="hold").is_actionable() is False
