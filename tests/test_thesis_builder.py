"""Tests for the deterministic thesis_builder (no LLM)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.screening_agent import ScreeningResult
from src.agents.thesis_builder import build_thesis


def test_build_thesis_from_breakout():
    result = ScreeningResult(
        symbol="AAPL", rsi=60.0, rel_volume=2.1, trend_up=True, breakout=True,
        score_tags=["breakout"],
    )
    thesis = build_thesis(result)
    assert thesis.symbol == "AAPL"
    assert thesis.action == "buy"
    assert "breakout" in thesis.signals_used
    assert len(thesis.reasoning) >= 50
    assert 5 <= thesis.horizon_days <= 20


def test_build_thesis_from_pullback():
    result = ScreeningResult(
        symbol="MSFT", rsi=45.0, rel_volume=1.8, trend_up=True, pullback_resume=True,
        score_tags=["pullback_resume"],
    )
    thesis = build_thesis(result)
    assert thesis.symbol == "MSFT"
    assert "pullback_resume" in thesis.signals_used


def test_build_thesis_clears_risk_gate_minimums():
    """The thesis must clear risk_gate's MIN_SIGNAL_CONFIDENCE / MIN_SIGNALS_REQUIRED."""
    from src.agents import risk_gate
    from src.core.state import ProjectState

    result = ScreeningResult(symbol="TSLA", breakout=True, score_tags=["breakout"])
    thesis = build_thesis(result)
    state = ProjectState()
    approved, reason = risk_gate.validate(thesis, state, balance=10000.0)
    assert approved is True, reason
