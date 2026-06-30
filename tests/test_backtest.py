"""
Tests for the backtester (engine + metrics).

Rules:
- No network calls, no EtoroClient, no LLM.
- All tests use synthetic DataFrames.
- Assertions verify: no look-ahead, correct fills, correct P&L accounting,
  proper IS/OOS split, and metric computations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.engine import (
    BacktestConfig,
    RunResult,
    Trade,
    _add_indicators,
    _entry_signal,
    run,
    split_run,
    walk_forward,
)
from src.backtest.metrics import compute


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(n: int, close_fn=None, high_fn=None, low_fn=None, vol_fn=None) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame of n bars."""
    closes  = [close_fn(i)  if close_fn  else 100.0 + i * 0.01 for i in range(n)]
    highs   = [high_fn(i)   if high_fn   else closes[i] * 1.01  for i in range(n)]
    lows    = [low_fn(i)    if low_fn    else closes[i] * 0.99  for i in range(n)]
    volumes = [vol_fn(i)    if vol_fn    else 1_000.0            for i in range(n)]
    return pd.DataFrame({
        "open":   closes,       # open == prev close (simplification)
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volumes,
    })


def _rsi_dip_bounce(n: int, dip_start: int, dip_depth: int = 30) -> pd.DataFrame:
    """
    Build a DataFrame where price dips below RSI-30 threshold and then recovers.
    Warmup bars are neutral; dip bars drop price; recovery bars rise.
    """
    assert n > dip_start + 40, "need enough bars after the dip"
    closes = []
    price = 100.0
    for i in range(n):
        if i < dip_start:
            price = 100.0 + 0.1 * i      # gentle uptrend
        elif i < dip_start + 10:
            price -= 2.0                 # sharp drop
        elif i < dip_start + 20:
            price += 1.5                 # recovery
        else:
            price += 0.2                 # drift up
        closes.append(max(price, 1.0))

    highs   = [c * 1.01 for c in closes]
    lows    = [c * 0.99 for c in closes]
    volumes = [1_000.0 + (2_000.0 if dip_start <= i < dip_start + 25 else 0)
               for i in range(n)]
    return pd.DataFrame({
        "open":   closes,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volumes,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Indicator tests
# ─────────────────────────────────────────────────────────────────────────────

def test_add_indicators_no_future_data():
    """
    Verify no look-ahead: indicators at row i must not depend on rows i+1..N-1.
    We do this by checking that truncating the df at bar i changes indicators
    ONLY for the last row (which now falls at position -1 in the shorter df),
    not for any earlier row.
    """
    n = 300
    df = _make_df(n)
    cfg = BacktestConfig()
    df_full = _add_indicators(df, cfg)

    # Compare indicators at bar 250 vs. a df truncated at bar 251
    trunc = _add_indicators(df.iloc[:251], cfg)
    for col in ["rsi", "ema20", "ema50", "sma200", "atr", "rel_vol"]:
        if col in df_full.columns and col in trunc.columns:
            full_val  = df_full.iloc[250][col]
            trunc_val = trunc.iloc[250][col]
            assert abs(full_val - trunc_val) < 1e-8 or (
                pd.isna(full_val) and pd.isna(trunc_val)
            ), f"Look-ahead detected in column '{col}' at bar 250"


def test_sma200_requires_200_bars():
    """SMA200 must be NaN for bars 0..198 (need 200 bars to compute)."""
    df = _make_df(250)
    cfg = BacktestConfig()
    df_ind = _add_indicators(df, cfg)
    # bar 198 (0-indexed) has only 199 values — not enough for min_periods=200
    assert pd.isna(df_ind["sma200"].iloc[198])
    # bar 199 (0-indexed) is the 200th value
    assert not pd.isna(df_ind["sma200"].iloc[199])


def test_rel_vol_uses_shifted_average():
    """
    Relative volume at bar i should NOT include bar i in the moving average.
    A spike in volume at bar t should NOT inflate rel_vol at bar t itself.
    """
    n = 260
    vols = [100.0] * n
    vols[240] = 100_000.0       # huge spike at bar 240

    df = _make_df(n, vol_fn=lambda i: vols[i])
    cfg = BacktestConfig()
    df_ind = _add_indicators(df, cfg)

    rv_at_spike = df_ind["rel_vol"].iloc[240]
    # At bar 240: avg of bars 220..239 = 100; vol[240]=100000 → rel_vol=1000
    # This is correct (current bar ÷ prior average).
    # The key is: bar 240's volume is NOT in the 20-bar trailing avg at bar 240.
    assert rv_at_spike == pytest.approx(100_000.0 / 100.0, rel=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Entry signal tests
# ─────────────────────────────────────────────────────────────────────────────

def test_entry_signal_requires_valid_atr():
    """No signal when ATR is NaN or zero."""
    row = pd.Series({
        "atr": float("nan"), "rel_vol": 2.0,
        "sma200": 90.0, "close": 100.0,
        "rsi": 31.0, "rsi_prev": 28.0, "ema_cross_recent": False,
    })
    cfg = BacktestConfig()
    assert _entry_signal(row, cfg) is None


def test_entry_signal_trend_filter_blocks_bearish():
    """When close < SMA200, no long entry should fire."""
    row = pd.Series({
        "atr": 2.0, "rel_vol": 2.0,
        "sma200": 120.0, "close": 100.0,   # below SMA200
        "rsi": 31.0, "rsi_prev": 28.0, "ema_cross_recent": True,
    })
    cfg = BacktestConfig(use_trend_filter=True)
    assert _entry_signal(row, cfg) is None


def test_entry_signal_rsi_reversal():
    """RSI crossing from below 30 to above 30 with volume confirms entry."""
    row = pd.Series({
        "atr": 2.0, "rel_vol": 2.0,
        "sma200": 90.0, "close": 100.0,
        "rsi": 31.0, "rsi_prev": 27.0,    # cross up through 30
        "ema_cross_recent": False,
    })
    cfg = BacktestConfig()
    assert _entry_signal(row, cfg) == "rsi_reversal"


def test_entry_signal_low_volume_no_rsi_signal():
    """RSI reversal without volume confirmation → no signal."""
    row = pd.Series({
        "atr": 2.0, "rel_vol": 1.2,       # below 1.5× threshold
        "sma200": 90.0, "close": 100.0,
        "rsi": 31.0, "rsi_prev": 27.0,
        "ema_cross_recent": False,
    })
    cfg = BacktestConfig()
    assert _entry_signal(row, cfg) is None


def test_entry_signal_ema_cross():
    """EMA cross with volume confirms entry when no RSI signal present."""
    row = pd.Series({
        "atr": 2.0, "rel_vol": 2.0,
        "sma200": 90.0, "close": 100.0,
        "rsi": 55.0, "rsi_prev": 52.0,    # no RSI signal
        "ema_cross_recent": True,
    })
    cfg = BacktestConfig()
    assert _entry_signal(row, cfg) == "ema_cross"


# ─────────────────────────────────────────────────────────────────────────────
# Engine integration tests
# ─────────────────────────────────────────────────────────────────────────────

def test_run_empty_if_too_few_bars():
    """With fewer bars than warmup+1, no trades should be generated."""
    df = _make_df(210)          # warmup = 210 (sma_trend=200 + 10)
    cfg = BacktestConfig()
    result = run(df, cfg, symbol="X")
    assert len(result.trades) == 0


def test_run_returns_equity_curve_same_length_as_df():
    """Equity curve must have one entry per bar."""
    n = 300
    df = _make_df(n)
    cfg = BacktestConfig()
    result = run(df, cfg, symbol="X")
    assert len(result.equity_curve) == n


def test_no_look_ahead_entry_fills_at_next_open():
    """
    Entry must fill at bar i+1 OPEN, not at bar i close.
    We inject a scenario with a known signal and verify entry_price.
    """
    n = 280
    df = _rsi_dip_bounce(n, dip_start=220)
    cfg = BacktestConfig(
        use_ema_signal=False,       # RSI-only for predictability
        use_trend_filter=False,     # remove SMA200 filter for this test
    )
    result = run(df, cfg, symbol="TEST")
    if not result.trades:
        pytest.skip("No trades generated — adjust synthetic data")

    trade = result.trades[0]
    entry_bar = trade.entry_bar
    expected_open = df.iloc[entry_bar]["open"]
    expected_price = expected_open * (1 + cfg.cost_per_side)
    assert trade.entry_price == pytest.approx(expected_price, rel=1e-6), (
        f"Entry price {trade.entry_price} != next-bar open × (1+cost) "
        f"{expected_price} at bar {entry_bar}"
    )


def test_stop_loss_fills_at_stop_price():
    """
    When bar low dips below stop, exit fills at stop price (not at low).
    Verify by engineering a bar where low < stop in a known position.
    """
    n = 260
    closes = [100.0 + i * 0.1 for i in range(n)]

    # After the warmup+signal window, artificially crash low on one bar
    crash_bar = 250
    lows = [c * 0.99 for c in closes]
    lows[crash_bar] = 1.0       # guaranteed to hit any stop

    df = pd.DataFrame({
        "open":   closes,
        "high":   [c * 1.01 for c in closes],
        "low":    lows,
        "close":  closes,
        "volume": [2_000.0] * n,  # high volume to help trigger signals
    })
    cfg = BacktestConfig(use_trend_filter=False, use_ema_signal=False)
    result = run(df, cfg, symbol="X")

    stop_exits = [t for t in result.trades if t.exit_reason == "stop_loss"]
    if not stop_exits:
        pytest.skip("No stop-loss exits — signal conditions not met")

    for t in stop_exits:
        # Exit price should be near the stop, not the low (1.0)
        assert t.exit_price > 50.0, (
            f"Stop loss exit filled at {t.exit_price}, expected near stop price not floor"
        )


def test_pnl_accounting_is_consistent():
    """Total trade P&L must reconcile with equity change."""
    n = 300
    df = _rsi_dip_bounce(n, dip_start=230)
    cfg = BacktestConfig(use_trend_filter=False)
    result = run(df, cfg, symbol="X")

    if not result.trades:
        pytest.skip("No trades in this run")

    total_pnl = sum(t.pnl for t in result.trades)
    equity_change = result.equity_curve[-1] - result.initial_equity
    assert total_pnl == pytest.approx(equity_change, rel=1e-6)


def test_max_positions_respected():
    """Engine must not open more than max_positions simultaneously."""
    n = 400
    # Use multiple symbols is not tested here (single symbol), but
    # open positions per run() call should not exceed max_positions
    df = _rsi_dip_bounce(n, dip_start=230)
    cfg = BacktestConfig(max_positions=1, use_trend_filter=False)
    result = run(df, cfg, symbol="X")

    # At any single bar, there should never be more than max_positions open
    # We verify indirectly: after entry, no re-entry until closed
    entries = sorted(t.entry_bar for t in result.trades)
    exits   = sorted(t.exit_bar  for t in result.trades)
    open_count = 0
    max_open   = 0
    all_events = [(b, "entry") for b in entries] + [(b, "exit") for b in exits]
    for _, kind in sorted(all_events):
        if kind == "entry":
            open_count += 1
        else:
            open_count -= 1
        max_open = max(max_open, open_count)

    assert max_open <= cfg.max_positions, (
        f"max_positions={cfg.max_positions} violated; max_open={max_open}"
    )


def test_time_limit_exit():
    """Positions must exit after max_hold_days bars."""
    n = 280
    df = _rsi_dip_bounce(n, dip_start=220)
    cfg = BacktestConfig(
        max_hold_days=5,
        use_trend_filter=False,
        exit_mode="trailing",       # trailing so TP doesn't fire before time limit
        trail_atr_multiple=99.0,    # huge trailing stop — won't fire
    )
    result = run(df, cfg, symbol="X")
    if not result.trades:
        pytest.skip("No trades generated")

    for t in result.trades:
        if t.exit_reason == "time_limit":
            assert t.holding_days <= cfg.max_hold_days + 1, (
                f"Trade held {t.holding_days} days, limit={cfg.max_hold_days}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# IS/OOS split tests
# ─────────────────────────────────────────────────────────────────────────────

def test_split_run_no_overlap():
    """IS and OOS trades must not share entry bars."""
    n = 500
    df = _rsi_dip_bounce(n, dip_start=260)
    cfg = BacktestConfig(use_trend_filter=False)
    is_r, oos_r = split_run(df, cfg, symbol="X", split_ratio=0.7)

    is_bars  = {t.entry_bar for t in is_r.trades}
    oos_bars = {t.entry_bar for t in oos_r.trades}
    assert is_bars.isdisjoint(oos_bars), (
        f"IS and OOS overlap on bars: {is_bars & oos_bars}"
    )


def test_split_run_equity_curves_cover_full_period():
    """IS + OOS equity curve lengths must cover the entire df."""
    n = 500
    df = _make_df(n)
    cfg = BacktestConfig(use_trend_filter=False)
    is_r, oos_r = split_run(df, cfg, symbol="X", split_ratio=0.7)
    assert len(is_r.equity_curve) + len(oos_r.equity_curve) == n


def test_split_oos_initial_equity_equals_is_final():
    """OOS starting equity must equal the final equity at the IS/OOS boundary."""
    n = 500
    df = _make_df(n)
    cfg = BacktestConfig(use_trend_filter=False)
    is_r, oos_r = split_run(df, cfg, symbol="X", split_ratio=0.7)
    if is_r.equity_curve:
        assert oos_r.initial_equity == pytest.approx(is_r.equity_curve[-1], rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward tests
# ─────────────────────────────────────────────────────────────────────────────

def test_walk_forward_returns_n_folds():
    n = 800
    df = _make_df(n)
    cfg = BacktestConfig(use_trend_filter=False)
    folds = walk_forward(df, cfg, symbol="X", n_splits=4)
    assert 1 <= len(folds) <= 4


def test_walk_forward_oos_labels():
    n = 800
    df = _make_df(n)
    cfg = BacktestConfig(use_trend_filter=False)
    folds = walk_forward(df, cfg, symbol="X", n_splits=3)
    for i, (_, oos_r) in enumerate(folds):
        assert "wf_fold" in oos_r.period_label, (
            f"Fold {i} OOS label missing 'wf_fold': {oos_r.period_label}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics tests
# ─────────────────────────────────────────────────────────────────────────────

def test_metrics_no_trades():
    result = RunResult(
        trades=[],
        equity_curve=[10_000.0] * 50,
        initial_equity=10_000.0,
        n_bars=50,
        period_label="test",
    )
    m = compute(result)
    assert m.n_trades == 0
    assert m.win_rate == 0.0
    assert m.total_pnl == 0.0


def test_metrics_all_wins():
    trades = [
        Trade(
            symbol="X", entry_date=0, exit_date=10, entry_bar=0, exit_bar=10,
            entry_price=100.0, exit_price=110.0, notional=1000.0,
            pnl=100.0, pnl_pct=10.0, holding_days=10,
            exit_reason="tp_rsi", signal_type="rsi_reversal", asset_class="equity",
        )
        for _ in range(5)
    ]
    curve = [10_000.0 + i * 100 for i in range(60)]
    result = RunResult(
        trades=trades,
        equity_curve=curve,
        initial_equity=10_000.0,
        n_bars=60,
        period_label="test",
    )
    m = compute(result)
    assert m.win_rate == pytest.approx(1.0)
    assert m.total_pnl == pytest.approx(500.0)
    assert m.profit_factor == float("inf")


def test_metrics_drawdown():
    """Max drawdown should reflect a 50% drop from peak."""
    curve = [10_000.0, 12_000.0, 6_000.0, 8_000.0]
    result = RunResult(
        trades=[],
        equity_curve=curve,
        initial_equity=10_000.0,
        n_bars=4,
        period_label="test",
    )
    m = compute(result)
    # Peak=12000, trough=6000 → DD = 50%
    assert m.max_drawdown_pct == pytest.approx(50.0, abs=0.1)


def test_metrics_sharpe_positive_for_trend():
    """Rising equity should produce a positive Sharpe ratio."""
    curve = [10_000.0 * (1.001 ** i) for i in range(252)]
    result = RunResult(
        trades=[],
        equity_curve=curve,
        initial_equity=10_000.0,
        n_bars=252,
        period_label="test",
    )
    m = compute(result)
    assert m.sharpe > 0.0
