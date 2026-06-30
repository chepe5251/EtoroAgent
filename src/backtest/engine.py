"""
Backtest engine — 100% rules-based, zero LLM, zero look-ahead.

NO LOOK-AHEAD GUARANTEE
=======================
Indicators at bar i are computed ONLY from bars 0..i (inclusive).
This is guaranteed structurally:
  - Indicator series use pandas rolling/ewm with no negative shifts.
  - Signals are read from row i; entries fill at row i+1 OPEN.
  - Stop losses fill at the stop price when low_i < stop (intraday, using
    bar i data only — the stop was set from bar i-1 or earlier data).
  - All exit conditions (RSI TP, time limit) fill at bar i+1 OPEN.

CORRESPONDENCE WITH src/tools/technical.py
==========================================
The vectorised series computations here use the same mathematical formulas
as the scalar functions in technical.py.  The only difference is the EMA
seed (pandas ewm starts at first observation; technical.py seeds with SMA
of first `period` values).  After ~2×period bars, they converge.  Given
the >=200 bar warmup, divergence is limited to the first ~100 bars which
are never traded anyway.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    # Entry thresholds
    rsi_oversold: float = 30.0          # doctrine: 30 (not 35)
    rsi_overbought: float = 70.0        # doctrine: 70 (not 65)
    ema_fast: int = 20
    ema_slow: int = 50
    sma_trend: int = 200
    vol_multiplier: float = 1.5         # volume > N× 20-day avg

    # Filters
    use_trend_filter: bool = True        # require close > SMA200 for long entry
    use_rsi_signal: bool = True          # enable RSI reversal entries
    use_ema_signal: bool = True          # enable EMA crossover entries
    ema_cross_lookback: int = 3          # cross must have happened in last N bars

    # Sizing
    initial_equity: float = 10_000.0
    risk_per_trade_pct: float = 1.0      # % of equity to risk per trade
    atr_stop_multiple: float = 1.5       # stop = k×ATR from entry
    max_notional_pct: float = 10.0       # cap single position at 10% notional

    # Exits
    exit_mode: str = "mean_reversion"    # "mean_reversion" | "trailing"
    tp_rsi_level: float = 55.0           # for mean_reversion: exit when RSI >= this
    trail_atr_multiple: float = 1.5      # for trailing: trail by N×ATR
    max_hold_days: int = 20              # hard time limit

    # Costs (conservative realistic defaults for eToro)
    spread_pct: float = 0.05            # 0.05% one-way
    slippage_pct: float = 0.05          # 0.05% one-way
    commission_pct: float = 0.0         # eToro stocks: 0% commission

    # Portfolio constraints
    max_positions: int = 3

    @property
    def cost_per_side(self) -> float:
        return (self.spread_pct + self.slippage_pct + self.commission_pct) / 100.0


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _OpenPosition:
    """Live position during backtest simulation (not persisted)."""
    symbol: str
    entry_bar: int                       # index of entry bar (bar i+1 from signal)
    entry_date: object                   # datetime/str label
    entry_price: float
    stop_price: float
    notional: float                      # $ invested
    asset_class: str                     # "equity" | "crypto"
    signal_type: str                     # "rsi_reversal" | "ema_cross"
    trail_stop: float = 0.0              # current trailing stop (0 = not trailing yet)


@dataclass
class Trade:
    """Completed trade record."""
    symbol: str
    entry_date: object
    exit_date: object
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    notional: float
    pnl: float                           # $ P&L after costs
    pnl_pct: float                       # % P&L on notional
    holding_days: int
    exit_reason: str                     # "stop_loss" | "tp_rsi" | "time_limit" | "trail_stop" | "end_of_data"
    signal_type: str
    asset_class: str


@dataclass
class RunResult:
    """Complete result of a single backtest run (or a IS/OOS slice)."""
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    bar_dates: list[object] = field(default_factory=list)
    initial_equity: float = 10_000.0
    n_bars: int = 0
    period_label: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Indicator series (no look-ahead by construction — all backward-looking)
# ──────────────────────────────────────────────────────────────────────────────

def _add_indicators(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """
    Compute all indicator columns and attach them to a copy of df.

    Each df['indicator'].iloc[i] is computed using ONLY rows 0..i.
    (rolling/ewm use backward windows only; no negative shifts.)

    Corresponds to functions in src/tools/technical.py:
      rsi        ≡ technical.rsi(closes, 14)
      ema20/50   ≡ technical.ema(closes, 20/50)
      atr        ≡ technical.atr(highs, lows, closes, 14)
      rel_vol    ≡ technical.relative_volume(volumes, 20)
      sma200     ≡ SMA used as trend filter (not in technical.py scalar form)
    """
    df = df.copy()

    # ── RSI (Wilder smoothing, alpha=1/14) ──────────────────────────────────
    delta = df["close"].diff()
    gain  = delta.clip(lower=0.0)
    loss  = (-delta).clip(lower=0.0)
    alpha = 1.0 / 14
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=14).mean()
    rs = np.where(
        (avg_gain == 0) & (avg_loss == 0), np.nan,         # flat: undefined
        np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)  # no loss: RSI=100
    )
    df["rsi"] = np.where(np.isinf(rs), 100.0, 100.0 - 100.0 / (1.0 + rs))
    df["rsi_prev"] = df["rsi"].shift(1)

    # ── EMA20 / EMA50 ────────────────────────────────────────────────────────
    df["ema20"] = df["close"].ewm(span=cfg.ema_fast, adjust=False,
                                  min_periods=cfg.ema_fast).mean()
    df["ema50"] = df["close"].ewm(span=cfg.ema_slow, adjust=False,
                                  min_periods=cfg.ema_slow).mean()

    # EMA20 crossing above EMA50 in the last N bars
    ema_bull = (df["ema20"] > df["ema50"]).astype(float)
    cross_up = ((ema_bull == 1) & (ema_bull.shift(1) == 0)).astype(float)
    df["ema_cross_recent"] = cross_up.rolling(cfg.ema_cross_lookback).max().astype(bool)

    # ── SMA200 (trend filter) ────────────────────────────────────────────────
    df["sma200"] = df["close"].rolling(cfg.sma_trend, min_periods=cfg.sma_trend).mean()

    # ── ATR (Wilder smoothing, period=14) ────────────────────────────────────
    hl   = df["high"] - df["low"]
    hpc  = (df["high"] - df["close"].shift(1)).abs()
    lpc  = (df["low"]  - df["close"].shift(1)).abs()
    tr   = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1.0/14, adjust=False, min_periods=14).mean()

    # ── Relative volume (current / avg of PREVIOUS 20 bars) ─────────────────
    # shift(1) ensures we don't include the current bar in the average
    vol_prev_avg = df["volume"].shift(1).rolling(20, min_periods=20).mean()
    df["rel_vol"] = df["volume"] / vol_prev_avg

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Signal logic
# ──────────────────────────────────────────────────────────────────────────────

def _entry_signal(row: pd.Series, cfg: BacktestConfig) -> Optional[str]:
    """
    Return signal type if entry conditions are met for the LONG side, else None.

    Called at the CLOSE of bar i; entry fills at OPEN of bar i+1.

    No data from bar i+1 or later is used here.
    """
    # Need valid ATR for stop sizing
    if not np.isfinite(row.get("atr", np.nan)) or row["atr"] <= 0:
        return None
    # Relative volume confirmation
    if not np.isfinite(row.get("rel_vol", np.nan)):
        return None
    vol_ok = row["rel_vol"] >= cfg.vol_multiplier

    # Trend filter
    trend_ok = True
    if cfg.use_trend_filter:
        sma = row.get("sma200", np.nan)
        if not np.isfinite(sma):
            return None  # no trend data yet — skip
        trend_ok = row["close"] > sma

    if not trend_ok:
        return None

    # ── Signal A: RSI reversal ───────────────────────────────────────────────
    if cfg.use_rsi_signal:
        rsi_curr = row.get("rsi", np.nan)
        rsi_prev = row.get("rsi_prev", np.nan)
        if (np.isfinite(rsi_curr) and np.isfinite(rsi_prev) and
                rsi_prev < cfg.rsi_oversold and rsi_curr >= cfg.rsi_oversold and
                vol_ok):
            return "rsi_reversal"

    # ── Signal B: EMA crossover ──────────────────────────────────────────────
    if cfg.use_ema_signal:
        if row.get("ema_cross_recent", False) and vol_ok:
            return "ema_cross"

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main backtest loop
# ──────────────────────────────────────────────────────────────────────────────

def run(
    df: pd.DataFrame,
    cfg: BacktestConfig,
    symbol: str = "?",
    asset_class: str = "equity",
) -> RunResult:
    """
    Run the full backtest on a single symbol's DataFrame.

    Returns a RunResult with all trades and the equity curve.
    The equity curve has one entry per bar (length == len(df)).
    """
    df_ind = _add_indicators(df, cfg)
    n = len(df_ind)

    equity = cfg.initial_equity
    equity_curve: list[float] = [equity] * n
    positions: list[_OpenPosition] = []
    trades: list[Trade] = []

    # Warmup: need SMA200 + some additional stabilisation
    warmup = cfg.sma_trend + 10

    for i in range(n):
        row = df_ind.iloc[i]

        # ── Update trailing stops in-flight ──────────────────────────────────
        for pos in positions:
            if cfg.exit_mode == "trailing" and pos.trail_stop > 0:
                # Ratchet up the trail if price moved further up
                new_trail = row["close"] - cfg.trail_atr_multiple * row["atr"]
                if new_trail > pos.trail_stop:
                    pos.trail_stop = new_trail

        # ── Check exits (current bar's OHLC, no future data) ─────────────────
        for pos in list(positions):
            exit_price: Optional[float] = None
            exit_reason: str = ""

            # 1. Stop loss — intraday fill at stop price
            if row["low"] <= pos.stop_price:
                exit_price = pos.stop_price * (1 - cfg.cost_per_side)
                exit_reason = "stop_loss"

            # 2. Trailing stop hit
            elif cfg.exit_mode == "trailing" and pos.trail_stop > 0:
                if row["low"] <= pos.trail_stop:
                    exit_price = pos.trail_stop * (1 - cfg.cost_per_side)
                    exit_reason = "trail_stop"

            # 3. Time limit → fill at next bar's open
            elif (i - pos.entry_bar) >= cfg.max_hold_days:
                if i + 1 < n:
                    exit_price = df_ind.iloc[i + 1]["open"] * (1 - cfg.cost_per_side)
                else:
                    exit_price = row["close"] * (1 - cfg.cost_per_side)
                exit_reason = "time_limit"

            # 4. Mean-reversion TP: RSI returns to mid-range → fill at next open
            elif cfg.exit_mode == "mean_reversion":
                rsi_val = row.get("rsi", np.nan)
                if np.isfinite(rsi_val) and rsi_val >= cfg.tp_rsi_level:
                    if i + 1 < n:
                        exit_price = df_ind.iloc[i + 1]["open"] * (1 - cfg.cost_per_side)
                    else:
                        exit_price = row["close"] * (1 - cfg.cost_per_side)
                    exit_reason = "tp_rsi"

            if exit_price is not None:
                pnl_per_unit = exit_price - pos.entry_price
                units = pos.notional / pos.entry_price
                pnl = pnl_per_unit * units
                pnl_pct = pnl / pos.notional * 100.0
                equity += pnl
                trades.append(Trade(
                    symbol=pos.symbol,
                    entry_date=pos.entry_date,
                    exit_date=df_ind.index[i] if hasattr(df_ind.index, "__iter__") else i,
                    entry_bar=pos.entry_bar,
                    exit_bar=i,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    notional=pos.notional,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_days=i - pos.entry_bar,
                    exit_reason=exit_reason,
                    signal_type=pos.signal_type,
                    asset_class=asset_class,
                ))
                positions.remove(pos)

        # ── Activate trailing stop when profitable ────────────────────────────
        if cfg.exit_mode == "trailing":
            for pos in positions:
                if pos.trail_stop == 0:
                    gain = row["close"] - pos.entry_price
                    threshold = 0.5 * row["atr"]  # activate after 0.5×ATR gain
                    if gain >= threshold:
                        pos.trail_stop = row["close"] - cfg.trail_atr_multiple * row["atr"]

        # ── Entry signal (only after warmup, only if slot available) ─────────
        if i >= warmup and i < n - 1 and len(positions) < cfg.max_positions:
            # Check no duplicate symbol already open
            open_symbols = {p.symbol for p in positions}
            if symbol not in open_symbols:
                sig = _entry_signal(row, cfg)
                if sig:
                    next_row = df_ind.iloc[i + 1]
                    entry_price = next_row["open"] * (1 + cfg.cost_per_side)
                    stop_dist = row["atr"] * cfg.atr_stop_multiple
                    if stop_dist > 0:
                        stop_price = entry_price - stop_dist
                        # Risk-based sizing
                        risk_amount = equity * (cfg.risk_per_trade_pct / 100.0)
                        units = risk_amount / stop_dist
                        notional = units * entry_price
                        # Cap notional
                        max_notional = equity * (cfg.max_notional_pct / 100.0)
                        notional = min(notional, max_notional)
                        if notional >= 1.0 and equity > 0:
                            positions.append(_OpenPosition(
                                symbol=symbol,
                                entry_bar=i + 1,
                                entry_date=df_ind.index[i + 1],
                                entry_price=entry_price,
                                stop_price=stop_price,
                                notional=notional,
                                asset_class=asset_class,
                                signal_type=sig,
                            ))

        equity_curve[i] = equity

    # ── Close any positions still open at the last bar ────────────────────────
    for pos in positions:
        last_price = df_ind.iloc[-1]["close"] * (1 - cfg.cost_per_side)
        pnl_per_unit = last_price - pos.entry_price
        units = pos.notional / pos.entry_price
        pnl = pnl_per_unit * units
        pnl_pct = pnl / pos.notional * 100.0
        equity += pnl
        trades.append(Trade(
            symbol=pos.symbol,
            entry_date=pos.entry_date,
            exit_date=df_ind.index[-1],
            entry_bar=pos.entry_bar,
            exit_bar=n - 1,
            entry_price=pos.entry_price,
            exit_price=last_price,
            notional=pos.notional,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=n - 1 - pos.entry_bar,
            exit_reason="end_of_data",
            signal_type=pos.signal_type,
            asset_class=asset_class,
        ))
        equity_curve[-1] = equity

    return RunResult(
        trades=trades,
        equity_curve=equity_curve,
        bar_dates=list(df_ind.index),
        initial_equity=cfg.initial_equity,
        n_bars=n,
        period_label="full",
    )


# ──────────────────────────────────────────────────────────────────────────────
# IS/OOS split
# ──────────────────────────────────────────────────────────────────────────────

def split_run(
    df: pd.DataFrame,
    cfg: BacktestConfig,
    symbol: str = "?",
    split_ratio: float = 0.7,
    asset_class: str = "equity",
) -> tuple[RunResult, RunResult]:
    """
    Run backtest on the full df, then split trades chronologically.

    The split is applied to the ACTIONABLE bars only (after warmup), so both IS
    and OOS have well-warmed indicators.  We do NOT retrain parameters on IS.

    Returns: (is_result, oos_result)
    """
    warmup = cfg.sma_trend + 10
    n = len(df)
    actionable = n - warmup
    if actionable <= 0:
        empty = RunResult(initial_equity=cfg.initial_equity, period_label="insufficient_data")
        return empty, empty

    split_bar = warmup + int(actionable * split_ratio)
    full = run(df, cfg, symbol, asset_class)

    is_trades  = [t for t in full.trades if t.entry_bar < split_bar]
    oos_trades = [t for t in full.trades if t.entry_bar >= split_bar]

    is_result = RunResult(
        trades=is_trades,
        equity_curve=full.equity_curve[:split_bar],
        bar_dates=full.bar_dates[:split_bar],
        initial_equity=cfg.initial_equity,
        n_bars=split_bar,
        period_label="in_sample",
    )
    # OOS equity starts where IS ends
    oos_eq_start = full.equity_curve[split_bar - 1] if split_bar > 0 else cfg.initial_equity
    oos_result = RunResult(
        trades=oos_trades,
        equity_curve=full.equity_curve[split_bar:],
        bar_dates=full.bar_dates[split_bar:],
        initial_equity=oos_eq_start,
        n_bars=n - split_bar,
        period_label="out_of_sample",
    )
    return is_result, oos_result


# ──────────────────────────────────────────────────────────────────────────────
# Walk-forward
# ──────────────────────────────────────────────────────────────────────────────

def walk_forward(
    df: pd.DataFrame,
    cfg: BacktestConfig,
    symbol: str = "?",
    asset_class: str = "equity",
    n_splits: int = 4,
) -> list[tuple[RunResult, RunResult]]:
    """
    Walk-forward: slide an expanding IS window + fixed OOS window.

    Returns list of (is_result, oos_result) for each fold.
    Only the OOS windows from each fold are used for final validation.
    """
    warmup = cfg.sma_trend + 10
    n = len(df)
    actionable = n - warmup
    if actionable < n_splits * 50:          # need at least 50 bars per fold
        return []

    fold_size = actionable // n_splits
    results = []
    for fold in range(n_splits):
        oos_end   = warmup + (fold + 1) * fold_size
        oos_start = oos_end - fold_size // 4   # OOS = last 25% of each fold
        oos_start = max(oos_start, warmup + fold_size)

        if oos_end > n:
            break

        fold_df = df.iloc[:oos_end]
        ratio = oos_start / oos_end
        is_r, oos_r = split_run(fold_df, cfg, symbol, ratio, asset_class)
        oos_r.period_label = f"wf_fold_{fold+1}_oos"
        results.append((is_r, oos_r))

    return results
