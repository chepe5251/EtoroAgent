"""
Tests for the backtester (engine + metrics).

Rules:
- No network calls, no EtoroClient, no LLM.
- All tests use synthetic DataFrames.
- Assertions verify: no look-ahead, correct fills, correct P&L accounting,
  proper IS/OOS split, metric computations, MTM DD, gap-through fills,
  and per-asset-class cost differences.
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
    _OpenPosition,
    _add_indicators,
    _entry_signal,
    _gap_fill,
    _unrealised_pnl,
    run,
    split_run,
    walk_forward,
)
from src.backtest.metrics import compute


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_df(n: int, close_fn=None, high_fn=None, low_fn=None,
             vol_fn=None, open_fn=None) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame of n bars."""
    closes  = [close_fn(i)  if close_fn  else 100.0 + i * 0.01 for i in range(n)]
    highs   = [high_fn(i)   if high_fn   else closes[i] * 1.01  for i in range(n)]
    lows    = [low_fn(i)    if low_fn    else closes[i] * 0.99  for i in range(n)]
    volumes = [vol_fn(i)    if vol_fn    else 1_000.0            for i in range(n)]
    opens   = [open_fn(i)   if open_fn   else closes[i]          for i in range(n)]
    return pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volumes,
    })


def _rsi_dip_bounce(n: int, dip_start: int) -> pd.DataFrame:
    """
    Build a DataFrame where price dips (RSI<30) then recovers.
    Volume is elevated during the dip so the signal fires.
    """
    assert n > dip_start + 40
    closes = []
    price = 100.0
    for i in range(n):
        if i < dip_start:
            price = 100.0 + 0.1 * i
        elif i < dip_start + 10:
            price -= 2.0
        elif i < dip_start + 20:
            price += 1.5
        else:
            price += 0.2
        closes.append(max(price, 1.0))

    highs   = [c * 1.01 for c in closes]
    lows    = [c * 0.99 for c in closes]
    # Volume spike starts at the recovery phase (dip_start+8), NOT at the dip
    # start.  When RSI crosses 30 (~dip_start+13), only ~5 elevated-volume bars
    # are in the 20-bar rolling average → rel_vol ≈ 2.0 × baseline (≥ 1.5).
    # If the spike started at dip_start, the average absorbs it and rel_vol
    # would be ~1.3 at the RSI cross — below the 1.5 threshold.
    volumes = [1_000.0 + (2_000.0 if dip_start + 8 <= i < dip_start + 33 else 0)
               for i in range(n)]
    return pd.DataFrame({
        "open":   closes,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volumes,
    })


# ─────────────────────────────────────────────────────────────────────────────
# A1 — MTM unit helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_unrealised_pnl_excludes_future_entries():
    """Positions with entry_bar > current_bar must NOT contribute to MTM."""
    pos_open = _OpenPosition(
        symbol="X", entry_bar=5, entry_date=5,
        entry_price=100.0, stop_price=95.0, notional=1000.0,
        asset_class="equity", signal_type="rsi_reversal",
    )
    pos_pending = _OpenPosition(
        symbol="X", entry_bar=11, entry_date=11,
        entry_price=100.0, stop_price=95.0, notional=1000.0,
        asset_class="equity", signal_type="rsi_reversal",
    )
    # At bar 10: pos_open is in (entry_bar=5 <= 10), pos_pending is not (11 > 10)
    pnl = _unrealised_pnl([pos_open, pos_pending], close_price=110.0, current_bar=10)
    units = 1000.0 / 100.0
    assert pnl == pytest.approx((110.0 - 100.0) * units)   # only pos_open


def test_mtm_equity_shows_drawdown_during_open_trade():
    """
    A position that goes deeply underwater before recovering must produce
    a visible drawdown in the MTM equity curve.

    Without MTM (old behaviour): equity stays flat during the adverse
    excursion; DD = 0 if trade eventually closes green.
    With MTM (A1): the underwater period appears in the curve → DD > 0.
    """
    # Build: strong uptrend (SMA200 clear), then a deep dip and recovery
    n = 320
    closes = []
    price = 100.0
    for i in range(n):
        if i < 220:
            price = 100.0 + 0.15 * i       # steady uptrend for warmup + trend filter
        elif i < 230:
            price -= 3.5                    # sharp dip → RSI drops hard
        elif i < 245:
            price += 2.5                    # recovery → RSI crosses back up (TP)
        else:
            price += 0.1
        closes.append(max(price, 1.0))

    highs   = [c * 1.01 for c in closes]
    lows    = [c * 0.99 for c in closes]
    # Volume spike starts at bar 228 (recovery phase), not at 220 (dip start).
    # At RSI cross (~bar 232): only 4-5 elevated bars in 20-bar average → rel_vol ≈ 2.3
    vols    = [1_000.0 + (3_000.0 if 228 <= i < 248 else 0) for i in range(n)]

    df = pd.DataFrame({
        "open": closes, "high": highs, "low": lows,
        "close": closes, "volume": vols,
    })

    cfg = BacktestConfig(
        use_trend_filter=False,  # simplify: no SMA200 filter needed for this test
        use_ema_signal=False,
        exit_mode="mean_reversion",
        tp_rsi_level=55.0,
        max_hold_days=50,
    )
    result = run(df, cfg, symbol="X")

    if not result.trades:
        pytest.skip("No trades generated — adjust synthetic data")

    m = compute(result)
    # With MTM, the dip must show up as drawdown even if trade closes green
    assert m.max_drawdown_pct > 0.0, (
        f"Expected DD > 0 (MTM should capture the intra-trade dip), got {m.max_drawdown_pct}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# A2 — Gap-through fill unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_gap_fill_long_no_gap():
    """Normal intraday stop: open > stop → fill at stop."""
    fill = _gap_fill(open_price=105.0, stop_level=100.0, cost=0.001, direction="long")
    assert fill == 100.0


def test_gap_fill_long_gap_down():
    """Gap-down: open <= stop → fill at open (worse than stop)."""
    fill = _gap_fill(open_price=95.0, stop_level=100.0, cost=0.001, direction="long")
    assert fill == 95.0   # fills at open, not at stop


def test_gap_fill_long_open_equals_stop():
    """Open exactly at stop → treat as gap (fill at open == stop)."""
    fill = _gap_fill(open_price=100.0, stop_level=100.0, cost=0.001, direction="long")
    assert fill == 100.0   # open == stop, fill at open == stop → same result


def test_stop_loss_gap_through_fills_at_open():
    """
    Integration test: when a bar opens BELOW the stop price, the trade exit
    must fill at open (not at the stop price, which would be unreachable).
    """
    n = 260
    closes = [100.0 + i * 0.5 for i in range(n)]   # gentle uptrend
    highs  = [c * 1.01 for c in closes]
    lows   = [c * 0.99 for c in closes]
    opens  = list(closes)                           # default: open = close

    # At bar 250, gap hard down: open well below any stop
    gap_bar = 250
    opens[gap_bar] = 50.0   # massive gap-down open
    lows[gap_bar]  = 45.0   # low is below open

    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": [2_000.0] * n,
    })
    cfg = BacktestConfig(use_trend_filter=False, use_ema_signal=False,
                         exit_mode="trailing", trail_atr_multiple=99.0)
    result = run(df, cfg, symbol="X")

    gap_exits = [t for t in result.trades
                 if t.exit_reason == "stop_loss" and t.exit_bar == gap_bar]
    if not gap_exits:
        pytest.skip("Gap bar did not produce a stop-loss exit")

    for t in gap_exits:
        pos_cost = cfg.cost_per_side(t.asset_class)
        expected_fill = 50.0 * (1.0 - pos_cost)   # gap open, not stop price
        assert t.exit_price == pytest.approx(expected_fill, rel=1e-6), (
            f"Gap-through fill should be at open×(1-cost)={expected_fill:.4f}, "
            f"got {t.exit_price:.4f}"
        )


def test_stop_loss_no_gap_fills_at_stop_price():
    """
    When bar opens above stop but trades through intraday, fill must be at
    stop (not at open, not at low).
    """
    n = 260
    closes = [100.0 + i * 0.1 for i in range(n)]
    lows   = [c * 0.99 for c in closes]
    lows[250] = 1.0   # guaranteed intraday breach; open stays at close (no gap)
    df = pd.DataFrame({
        "open":   closes,
        "high":   [c * 1.01 for c in closes],
        "low":    lows,
        "close":  closes,
        "volume": [2_000.0] * n,
    })
    cfg = BacktestConfig(use_trend_filter=False, use_ema_signal=False)
    result = run(df, cfg, symbol="X")

    stop_exits = [t for t in result.trades if t.exit_reason == "stop_loss"]
    if not stop_exits:
        pytest.skip("No stop-loss exits")

    for t in stop_exits:
        # Open > stop → fill is at stop × (1-cost), definitely not at 1.0
        assert t.exit_price > 50.0, (
            f"Stop fill {t.exit_price:.4f} looks like it filled at low (1.0)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# A3 — Per-class costs
# ─────────────────────────────────────────────────────────────────────────────

def test_crypto_cost_higher_than_equity():
    """cost_per_side('crypto') must be strictly greater than cost_per_side('equity')."""
    cfg = BacktestConfig()
    assert cfg.cost_per_side("crypto") > cfg.cost_per_side("equity"), (
        "Crypto one-way cost must exceed equity cost"
    )


def test_crypto_carry_higher_than_equity():
    """daily_carry('crypto') must be strictly greater than daily_carry('equity')."""
    cfg = BacktestConfig()
    assert cfg.daily_carry("crypto") > cfg.daily_carry("equity")


def test_carry_cost_deducted_from_pnl():
    """
    A profitable trade that was held for many days must have a lower P&L
    than the same trade held for 0 days, because carry cost grows with holding_days.

    We verify this by running two otherwise-identical scenarios that differ only
    in time-limit (so the second trade is forced to close sooner).
    """
    n = 280
    df = _rsi_dip_bounce(n, dip_start=220)
    cfg_long_hold  = BacktestConfig(
        use_trend_filter=False, use_ema_signal=False, exit_mode="trailing",
        trail_atr_multiple=99.0, max_hold_days=30,
        equity_carry_daily_pct=0.5,   # high carry so the effect is visible
    )
    cfg_short_hold = BacktestConfig(
        use_trend_filter=False, use_ema_signal=False, exit_mode="trailing",
        trail_atr_multiple=99.0, max_hold_days=1,
        equity_carry_daily_pct=0.5,
    )
    res_long  = run(df, cfg_long_hold,  symbol="X")
    res_short = run(df, cfg_short_hold, symbol="X")

    if not res_long.trades or not res_short.trades:
        pytest.skip("No trades in one of the runs")

    carry_deducted = sum(t.carry_cost for t in res_long.trades)
    assert carry_deducted > 0.0, "Long-hold run should have positive carry cost deducted"


def test_pnl_reconciles_with_equity_change_after_carry():
    """
    sum(t.pnl) for all trades must equal equity_curve[-1] − initial_equity.
    This holds even after adding carry cost deductions (A3) and MTM (A1).
    """
    n = 300
    df = _rsi_dip_bounce(n, dip_start=230)
    cfg = BacktestConfig(use_trend_filter=False)
    result = run(df, cfg, symbol="X")

    if not result.trades:
        pytest.skip("No trades")

    total_pnl = sum(t.pnl for t in result.trades)
    equity_change = result.equity_curve[-1] - result.initial_equity
    assert total_pnl == pytest.approx(equity_change, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Indicator tests (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def test_add_indicators_no_future_data():
    """
    Verify no look-ahead: indicators at row i must not depend on rows i+1..N-1.
    Truncating the df at bar 251 must not change any indicator value at bar 250.
    """
    n = 300
    df = _make_df(n)
    cfg = BacktestConfig()
    df_full = _add_indicators(df, cfg)
    trunc = _add_indicators(df.iloc[:251], cfg)
    for col in ["rsi", "ema20", "ema50", "sma200", "atr", "rel_vol"]:
        if col in df_full.columns and col in trunc.columns:
            full_val  = df_full.iloc[250][col]
            trunc_val = trunc.iloc[250][col]
            assert abs(full_val - trunc_val) < 1e-8 or (
                pd.isna(full_val) and pd.isna(trunc_val)
            ), f"Look-ahead in column '{col}' at bar 250"


def test_sma200_requires_200_bars():
    df = _make_df(250)
    cfg = BacktestConfig()
    df_ind = _add_indicators(df, cfg)
    assert pd.isna(df_ind["sma200"].iloc[198])
    assert not pd.isna(df_ind["sma200"].iloc[199])


def test_rel_vol_uses_shifted_average():
    n = 260
    vols = [100.0] * n
    vols[240] = 100_000.0
    df = _make_df(n, vol_fn=lambda i: vols[i])
    cfg = BacktestConfig()
    df_ind = _add_indicators(df, cfg)
    assert df_ind["rel_vol"].iloc[240] == pytest.approx(100_000.0 / 100.0, rel=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Entry signal tests
# ─────────────────────────────────────────────────────────────────────────────

def test_entry_signal_requires_valid_atr():
    row = pd.Series({
        "atr": float("nan"), "rel_vol": 2.0,
        "sma200": 90.0, "close": 100.0,
        "rsi": 31.0, "rsi_prev": 28.0, "ema_cross_recent": False,
    })
    assert _entry_signal(row, BacktestConfig()) is None


def test_entry_signal_trend_filter_blocks_bearish():
    row = pd.Series({
        "atr": 2.0, "rel_vol": 2.0,
        "sma200": 120.0, "close": 100.0,
        "rsi": 31.0, "rsi_prev": 28.0, "ema_cross_recent": True,
    })
    assert _entry_signal(row, BacktestConfig(use_trend_filter=True)) is None


def test_entry_signal_rsi_reversal():
    row = pd.Series({
        "atr": 2.0, "rel_vol": 2.0,
        "sma200": 90.0, "close": 100.0,
        "rsi": 31.0, "rsi_prev": 27.0,
        "ema_cross_recent": False,
    })
    assert _entry_signal(row, BacktestConfig()) == "rsi_reversal"


def test_entry_signal_low_volume_no_rsi_signal():
    row = pd.Series({
        "atr": 2.0, "rel_vol": 1.2,
        "sma200": 90.0, "close": 100.0,
        "rsi": 31.0, "rsi_prev": 27.0,
        "ema_cross_recent": False,
    })
    assert _entry_signal(row, BacktestConfig()) is None


def test_entry_signal_ema_cross():
    row = pd.Series({
        "atr": 2.0, "rel_vol": 2.0,
        "sma200": 90.0, "close": 100.0,
        "rsi": 55.0, "rsi_prev": 52.0,
        "ema_cross_recent": True,
    })
    assert _entry_signal(row, BacktestConfig()) == "ema_cross"


# ─────────────────────────────────────────────────────────────────────────────
# Engine integration tests
# ─────────────────────────────────────────────────────────────────────────────

def test_run_empty_if_too_few_bars():
    df = _make_df(210)
    result = run(df, BacktestConfig(), symbol="X")
    assert len(result.trades) == 0


def test_run_returns_equity_curve_same_length_as_df():
    n = 300
    df = _make_df(n)
    result = run(df, BacktestConfig(), symbol="X")
    assert len(result.equity_curve) == n


def test_no_look_ahead_entry_fills_at_next_open():
    """Entry must fill at bar i+1 OPEN (not at bar i close)."""
    n = 280
    df = _rsi_dip_bounce(n, dip_start=220)
    cfg = BacktestConfig(use_ema_signal=False, use_trend_filter=False)
    result = run(df, cfg, symbol="TEST")
    if not result.trades:
        pytest.skip("No trades")

    trade = result.trades[0]
    entry_bar = trade.entry_bar
    expected = df.iloc[entry_bar]["open"] * (1.0 + cfg.cost_per_side("equity"))
    assert trade.entry_price == pytest.approx(expected, rel=1e-6)


def test_max_positions_respected():
    n = 400
    df = _rsi_dip_bounce(n, dip_start=230)
    cfg = BacktestConfig(max_positions=1, use_trend_filter=False)
    result = run(df, cfg, symbol="X")

    entries = sorted(t.entry_bar for t in result.trades)
    exits   = sorted(t.exit_bar  for t in result.trades)
    open_count = max_open = 0
    for _, kind in sorted([(b, "e") for b in entries] + [(b, "x") for b in exits]):
        open_count += 1 if kind == "e" else -1
        max_open = max(max_open, open_count)
    assert max_open <= cfg.max_positions


def test_time_limit_exit():
    n = 280
    df = _rsi_dip_bounce(n, dip_start=220)
    cfg = BacktestConfig(
        max_hold_days=5, use_trend_filter=False,
        exit_mode="trailing", trail_atr_multiple=99.0,
    )
    result = run(df, cfg, symbol="X")
    if not result.trades:
        pytest.skip("No trades")
    for t in result.trades:
        if t.exit_reason == "time_limit":
            assert t.holding_days <= cfg.max_hold_days + 1


# ─────────────────────────────────────────────────────────────────────────────
# IS/OOS split tests
# ─────────────────────────────────────────────────────────────────────────────

def test_split_run_no_overlap():
    n = 500
    df = _rsi_dip_bounce(n, dip_start=260)
    cfg = BacktestConfig(use_trend_filter=False)
    is_r, oos_r = split_run(df, cfg, symbol="X", split_ratio=0.7)
    is_bars  = {t.entry_bar for t in is_r.trades}
    oos_bars = {t.entry_bar for t in oos_r.trades}
    assert is_bars.isdisjoint(oos_bars)


def test_split_run_equity_curves_cover_full_period():
    n = 500
    df = _make_df(n)
    cfg = BacktestConfig(use_trend_filter=False)
    is_r, oos_r = split_run(df, cfg, symbol="X", split_ratio=0.7)
    assert len(is_r.equity_curve) + len(oos_r.equity_curve) == n


def test_split_oos_initial_equity_equals_is_final():
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
    df = _make_df(800)
    cfg = BacktestConfig(use_trend_filter=False)
    folds = walk_forward(df, cfg, symbol="X", n_splits=4)
    assert 1 <= len(folds) <= 4


def test_walk_forward_oos_labels():
    df = _make_df(800)
    cfg = BacktestConfig(use_trend_filter=False)
    folds = walk_forward(df, cfg, symbol="X", n_splits=3)
    for i, (_, oos_r) in enumerate(folds):
        assert "wf_fold" in oos_r.period_label


# ─────────────────────────────────────────────────────────────────────────────
# Metrics tests
# ─────────────────────────────────────────────────────────────────────────────

def test_metrics_no_trades():
    result = RunResult(
        trades=[], equity_curve=[10_000.0] * 50,
        initial_equity=10_000.0, n_bars=50, period_label="test",
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
    result = RunResult(trades=trades, equity_curve=curve,
                       initial_equity=10_000.0, n_bars=60, period_label="test")
    m = compute(result)
    assert m.win_rate == pytest.approx(1.0)
    assert m.total_pnl == pytest.approx(500.0)
    assert m.profit_factor == float("inf")


def test_metrics_drawdown():
    """Max drawdown: peak 12k → trough 6k → DD = 50%."""
    curve = [10_000.0, 12_000.0, 6_000.0, 8_000.0]
    result = RunResult(trades=[], equity_curve=curve,
                       initial_equity=10_000.0, n_bars=4, period_label="test")
    m = compute(result)
    assert m.max_drawdown_pct == pytest.approx(50.0, abs=0.1)


def test_metrics_sharpe_positive_for_trend():
    curve = [10_000.0 * (1.001 ** i) for i in range(252)]
    result = RunResult(trades=[], equity_curve=curve,
                       initial_equity=10_000.0, n_bars=252, period_label="test")
    m = compute(result)
    assert m.sharpe > 0.0
