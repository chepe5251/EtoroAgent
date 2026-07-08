"""
Tests for ProjectState — peak-balance tracking and the account drawdown
hard stop (effective_risk_pct).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.state import ProjectState


def test_update_peak_balance_tracks_all_time_high():
    state = ProjectState()
    assert state.update_peak_balance(1000.0) == 0.0
    assert state.peak_balance == 1000.0

    # New high — peak moves up, no drawdown.
    assert state.update_peak_balance(1200.0) == 0.0
    assert state.peak_balance == 1200.0

    # Balance drops below peak — drawdown % reported, peak unchanged.
    dd = state.update_peak_balance(1080.0)
    assert dd == 10.0
    assert state.peak_balance == 1200.0


def test_effective_risk_pct_unaffected_below_threshold():
    state = ProjectState()
    state.update_peak_balance(1000.0)
    # 5% drawdown — below the 10% default hard-stop threshold.
    risk = state.effective_risk_pct(configured_pct=8.0, balance=950.0)
    assert risk == 8.0


def test_effective_risk_pct_drops_at_hard_stop_threshold():
    state = ProjectState()
    state.update_peak_balance(1000.0)
    # Exactly 10% drawdown — hard stop triggers (>=, not >).
    risk = state.effective_risk_pct(configured_pct=8.0, balance=900.0)
    assert risk == 3.0


def test_effective_risk_pct_recovers_once_balance_recovers():
    state = ProjectState()
    state.update_peak_balance(1000.0)
    assert state.effective_risk_pct(configured_pct=8.0, balance=880.0) == 3.0
    # Balance recovers back above the 10%-drawdown line.
    assert state.effective_risk_pct(configured_pct=8.0, balance=950.0) == 8.0


def test_effective_risk_pct_zero_peak_is_safe():
    """No balance observed yet (peak_balance == 0) must not divide by zero."""
    state = ProjectState()
    risk = state.effective_risk_pct(configured_pct=8.0, balance=0.0)
    assert risk == 8.0


def test_peak_balance_persists_across_save_load(tmp_path):
    path = tmp_path / "state.json"
    state = ProjectState()
    state.update_peak_balance(5000.0)
    state.save(path)

    reloaded = ProjectState.load(path)
    assert reloaded.peak_balance == 5000.0
