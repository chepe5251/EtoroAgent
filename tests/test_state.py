"""
Tests for ProjectState — peak-balance tracking and the account drawdown
hard stop (effective_risk_pct).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.core.state as state_module
from src.core.state import ProjectState

# Read dynamically (not captured once at import time): another test module
# (test_position_review.py) does importlib.reload(state_module) to test an
# env-var override, which mutates this module's globals in place for the
# rest of the session — a fresh lookup per-test stays correct regardless of
# test ordering/reloads.
def _threshold() -> float:
    return state_module._ACCOUNT_DRAWDOWN_HARD_STOP_PCT


def _reduced() -> float:
    return state_module._REDUCED_RISK_PCT


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
    # Half the configured threshold's drawdown — must stay under it.
    below_balance = 1000.0 * (1 - (_threshold() / 2) / 100.0)
    risk = state.effective_risk_pct(configured_pct=8.0, balance=below_balance)
    assert risk == 8.0


def test_effective_risk_pct_drops_at_hard_stop_threshold():
    state = ProjectState()
    state.update_peak_balance(1000.0)
    # Exactly at the configured threshold — hard stop triggers (>=, not >).
    at_threshold_balance = 1000.0 * (1 - _threshold() / 100.0)
    risk = state.effective_risk_pct(configured_pct=8.0, balance=at_threshold_balance)
    assert risk == _reduced()


def test_effective_risk_pct_recovers_once_balance_recovers():
    state = ProjectState()
    state.update_peak_balance(1000.0)
    past_threshold_balance = 1000.0 * (1 - (_threshold() + 2) / 100.0)
    assert state.effective_risk_pct(configured_pct=8.0, balance=past_threshold_balance) == _reduced()
    # Balance recovers back above the threshold-drawdown line.
    recovered_balance = 1000.0 * (1 - (_threshold() / 2) / 100.0)
    assert state.effective_risk_pct(configured_pct=8.0, balance=recovered_balance) == 8.0


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
