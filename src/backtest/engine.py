"""
Backtest engine — 100% rules-based, zero LLM, zero look-ahead.

NO LOOK-AHEAD GUARANTEE
=======================
Indicators at bar i are computed ONLY from bars 0..i (inclusive).
This is guaranteed structurally:
  - Indicator series use pandas rolling/ewm with no negative shifts.
  - Signals are read from row i; entries fill at row i+1 OPEN.
  - Stop losses fill at the stop price (or gap-open) using bar i data only.
  - All exit conditions (RSI TP, time limit) fill at bar i+1 OPEN.
  - MTM (mark-to-market) at bar i uses close[i] of already-open positions only;
    newly-signalled positions (entry_bar = i+1) are excluded from bar i's MTM.

HONESTY FIXES (relative to initial version)
============================================
A1 — Mark-to-market equity curve
    equity_curve[i] = realised_cash + unrealised_pnl_of_open_positions_at_close[i]
    This means draw-down and Sharpe now reflect intra-trade adverse excursions,
    not just closed-trade P&L.  A trade that goes -8% before recovering to +2%
    now shows the -8% trough in the equity curve.

A2 — Gap-through fills
    If a stop/trail bar opens at or below the stop level (e.g., overnight gap),
    the fill is at open × (1−cost), not stop × (1−cost).  Previously, stops always
    filled at the exact stop price regardless of where the bar opened.

A3 — Per-asset-class costs + overnight carry
    Equity and crypto have very different cost structures on eToro:
      Equity: tight spread (~0.07%), low slippage, 0% commission, low carry
      Crypto: wide spread (~1%), higher slippage, significant overnight CFD fees
    Each cost is parametrised separately in BacktestConfig.
    Overnight carry is deducted from P&L at close: notional × carry_pct × hold_days.

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
    use_trend_filter: bool = True        # require an uptrend for long entry
    trend_filter_type: str = "sma200"    # "sma200" (close>SMA200) | "ema50_200" (EMA50>EMA200)
    use_rsi_signal: bool = True          # enable RSI reversal entries
    use_ema_signal: bool = True          # enable EMA crossover entries
    ema_cross_lookback: int = 3          # cross must have happened in last N bars
    use_breakout_signal: bool = False    # enable Donchian breakout entries
    donchian_lookback: int = 20          # breakout = close > highest high of prior N bars
    use_pullback_signal: bool = False    # enable EMA20 pullback-resume entries
    profit_target_pct: float = 0.0       # 0 = disabled; fixed take-profit % (any exit mode)
    use_structure_filter: bool = False   # require HH/HL market structure to be bullish (see market_structure.py)
    structure_swing_k: int = 2           # bars each side to confirm a swing pivot

    # Sizing
    initial_equity: float = 10_000.0
    risk_per_trade_pct: float = 1.0      # % of equity to risk per trade
    atr_stop_multiple: float = 1.5       # stop = k×ATR from entry
    max_notional_pct: float = 10.0       # cap single position at 10% notional
    leverage: float = 1.0                # scales the notional cap only — risk
                                          # per trade (stop-distance sizing) is
                                          # unchanged; leverage just lets a
                                          # position grow bigger before hitting
                                          # the notional cap. Carry cost and P&L
                                          # already scale with notional, so a
                                          # bigger cap here correctly reflects
                                          # a bigger real-money cost/swing.

    # Exits
    exit_mode: str = "mean_reversion"    # "mean_reversion" | "trailing"
    tp_rsi_level: float = 55.0           # for mean_reversion: exit when RSI >= this
    trail_atr_multiple: float = 1.5      # for trailing: trail by N×ATR
    max_hold_days: int = 0               # hard time limit in bars; 0 = disabled (no forced exit)

    # ── Per-asset-class costs (A3) ────────────────────────────────────────────
    # Equity (eToro stocks / ETFs)
    # Source: eToro help-centre fee schedule + typical mid-spread observations.
    #   - Commission: 0% (eToro charges no explicit commission on real stocks)
    #   - Spread:     ~0.07–0.09% on liquid large-caps; use 0.07% conservative
    #   - Slippage:   ~0.03% market-impact for small retail sizes
    #   - Carry:      eToro charges an overnight CFD fee on leveraged positions;
    #                 for non-leveraged stocks the fee is usually 0.  We model a
    #                 small residual (e.g. borrow cost, transaction tax amortised)
    #                 at 0.01%/day ≈ 2.5%/yr.  Adjust to 0 for pure stock accounts.
    equity_spread_pct: float = 0.07
    equity_slippage_pct: float = 0.03
    equity_commission_pct: float = 0.0
    equity_carry_daily_pct: float = 0.01   # %/day of notional

    # Crypto (eToro crypto CFDs)
    # Source: eToro's own fee page — flat 1% per side for Bronze/Silver/Gold
    # tiers (confirmed 2026). Overnight CFD financing fee is charged nightly
    # Mon-Fri, and the Friday-night charge is tripled to cover the Sat+Sun
    # rollover. This weekend 3x multiplier IS modelled explicitly — see
    # _weighted_carry_nights() below, applied at both carry-cost sites.
    #   - Spread:     1.0% flat (Bronze/Silver/Gold; lower for Platinum+/Diamond)
    #   - Slippage:   0.25% (thinner order book than equities)
    #   - Carry:      base nightly rate below; Friday nights charged at 3x.
    crypto_spread_pct: float = 1.0
    crypto_slippage_pct: float = 0.25
    crypto_commission_pct: float = 0.0
    crypto_carry_daily_pct: float = 0.06   # %/day of notional

    # Portfolio constraints
    max_positions: int = 3

    def cost_per_side(self, asset_class: str = "equity") -> float:
        """One-way transaction cost as a fraction (not %). A3."""
        if asset_class == "crypto":
            return (self.crypto_spread_pct + self.crypto_slippage_pct +
                    self.crypto_commission_pct) / 100.0
        return (self.equity_spread_pct + self.equity_slippage_pct +
                self.equity_commission_pct) / 100.0

    def daily_carry(self, asset_class: str = "equity") -> float:
        """Daily carry cost as a fraction of notional (not %). A3."""
        if asset_class == "crypto":
            return self.crypto_carry_daily_pct / 100.0
        return self.equity_carry_daily_pct / 100.0


def _weighted_carry_nights(entry_date: object, exit_date: object, holding_days: int) -> float:
    """Count nights a position is held open, weighting Friday nights at 3x.

    eToro charges overnight CFD financing once per calendar night Mon-Fri,
    and bundles the Sat+Sun rollover into a single Friday-night charge at
    3x the standard rate. A position entered Monday and exited the
    following Monday is held 7 nights, one of which is a Friday — so it
    is charged as if it were 9 nights (4 weekday nights + 1 Friday x3).

    Falls back to a flat 1x/night (`holding_days`) when the index isn't a
    real calendar DatetimeIndex (e.g. synthetic integer-indexed test data),
    since weekday weighting is meaningless without real dates.
    """
    try:
        entry_ts = pd.Timestamp(entry_date)
        exit_ts = pd.Timestamp(exit_date)
    except (TypeError, ValueError):
        return float(holding_days)
    if entry_ts.year < 1980 or exit_ts.year < 1980:
        return float(holding_days)
    entry_ts = entry_ts.normalize()
    exit_ts = exit_ts.normalize()
    if exit_ts <= entry_ts:
        return float(holding_days)
    nights = pd.date_range(entry_ts, exit_ts, freq="D")[:-1]
    weights = np.where(nights.weekday == 4, 3.0, 1.0)
    return float(weights.sum())


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
    pnl: float                           # $ P&L after all costs (incl. carry)
    pnl_pct: float                       # % P&L on notional
    holding_days: int
    exit_reason: str                     # "stop_loss"|"tp_rsi"|"time_limit"|"trail_stop"|"end_of_data"
    signal_type: str
    asset_class: str
    carry_cost: float = 0.0              # $ carry deducted (for transparency)
    is_short: bool = False                # True for short-side trades (e.g. First Red Day)


@dataclass
class RunResult:
    """Complete result of a single backtest run (or a IS/OOS slice)."""
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    bar_dates: list[object] = field(default_factory=list)
    initial_equity: float = 10_000.0
    n_bars: int = 0
    period_label: str = ""


def _structure_trend_series(df: pd.DataFrame, k: int) -> list[bool]:
    """
    Bar-by-bar "is market structure bullish" flag, per market_structure.py's
    HH/HL swing logic (a swing pivot is confirmed k bars after it forms, so
    this carries zero look-ahead: the flag at bar i only reflects swings
    confirmable using bars up to i).
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)
    is_sh = [False] * n
    is_sl = [False] * n
    for i in range(k, n - k):
        window_h = highs[i - k : i + k + 1]
        if highs[i] >= window_h.max() and (window_h == highs[i]).sum() == 1:
            is_sh[i] = True
        window_l = lows[i - k : i + k + 1]
        if lows[i] <= window_l.min() and (window_l == lows[i]).sum() == 1:
            is_sl[i] = True

    swing_highs: list[float] = []
    swing_lows: list[float] = []
    bullish = [False] * n
    trend = False
    for i in range(n):
        confirm_idx = i - k
        if confirm_idx >= 0:
            if is_sh[confirm_idx]:
                swing_highs.append(highs[confirm_idx])
            if is_sl[confirm_idx]:
                swing_lows.append(lows[confirm_idx])
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                hh = swing_highs[-1] > swing_highs[-2]
                hl = swing_lows[-1] > swing_lows[-2]
                trend = hh and hl
        bullish[i] = trend
    return bullish


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
    df["ema200"] = df["close"].ewm(span=cfg.sma_trend, adjust=False,
                                   min_periods=cfg.sma_trend).mean()

    # ── EMA20 pullback-resume (close crosses back above EMA20) ──────────────
    above_ema20 = df["close"] > df["ema20"]
    df["pullback_resume"] = above_ema20 & ~above_ema20.shift(1).fillna(False).astype(bool)

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

    # ── Donchian channel high (trend-following breakout) ────────────────────
    # shift(1) excludes the current bar so "breakout" means close[i] clears
    # the highest high of the PRIOR N bars — no look-ahead.
    df["donchian_high"] = (
        df["high"].shift(1).rolling(cfg.donchian_lookback, min_periods=cfg.donchian_lookback).max()
    )

    # ── Market structure (HH/HL) bullish gate, computed on demand only ──────
    if cfg.use_structure_filter:
        df["structure_bullish"] = _structure_trend_series(df, cfg.structure_swing_k)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Signal logic
# ──────────────────────────────────────────────────────────────────────────────

def _entry_signal(row: pd.Series, cfg: BacktestConfig, asset_class: str = "equity") -> Optional[str]:
    """
    Return signal type if entry conditions are met for the LONG side, else None.

    Called at the CLOSE of bar i; entry fills at OPEN of bar i+1.

    No data from bar i+1 or later is used here.
    """
    # Need valid ATR for stop sizing
    if not np.isfinite(row.get("atr", np.nan)) or row["atr"] <= 0:
        return None
    # Relative volume confirmation — eToro's real API never reports crypto
    # volume (always None), so a permanently-NaN rel_vol would otherwise
    # block every crypto entry forever. Skip the volume gate for crypto;
    # keep it as a hard requirement for equities where real volume exists.
    rel_vol = row.get("rel_vol", np.nan)
    if np.isfinite(rel_vol):
        vol_ok = rel_vol >= cfg.vol_multiplier
    elif asset_class == "crypto":
        vol_ok = True
    else:
        return None

    # Trend filter
    trend_ok = True
    if cfg.use_trend_filter:
        if cfg.trend_filter_type == "ema50_200":
            ema50 = row.get("ema50", np.nan)
            ema200 = row.get("ema200", np.nan)
            if not (np.isfinite(ema50) and np.isfinite(ema200)):
                return None  # no trend data yet — skip
            trend_ok = ema50 > ema200
        else:
            sma = row.get("sma200", np.nan)
            if not np.isfinite(sma):
                return None  # no trend data yet — skip
            trend_ok = row["close"] > sma

    if not trend_ok:
        return None

    # Market-structure (HH/HL) gate — abstain unless swing structure is
    # confirmed bullish, e.g. if EMA-based trend_ok fired but price is
    # trading below the last significant swing low (structure broken/at risk
    # of ChoCh).
    if cfg.use_structure_filter and not row.get("structure_bullish", False):
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

    # ── Signal C: Donchian breakout (trend-following) ────────────────────────
    if cfg.use_breakout_signal:
        donchian_high = row.get("donchian_high", np.nan)
        if np.isfinite(donchian_high) and row["close"] > donchian_high and vol_ok:
            return "breakout"

    # ── Signal D: EMA20 pullback-resume (trend-following) ────────────────────
    if cfg.use_pullback_signal:
        if row.get("pullback_resume", False) and vol_ok:
            return "pullback"

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _gap_fill(
    open_price: float,
    stop_level: float,
    cost: float = 0.0,      # accepted for API symmetry; not used in body
    direction: str = "long",
) -> float:
    """
    Compute raw fill price for a stop, accounting for gap-through (A2).

    For a LONG position:
      - Normal: bar opens above stop → intraday trade-through → fill at stop.
      - Gap-down: bar opens at/below stop → fill at open (cannot fill above open).

    Returns the raw fill price BEFORE deducting transaction cost.
    The caller multiplies by (1 − cost) for a long or (1 + cost) for a short.
    """
    if direction == "long":
        return open_price if open_price <= stop_level else stop_level
    else:  # short (Phase B)
        return open_price if open_price >= stop_level else stop_level


def _unrealised_pnl(
    positions: list[_OpenPosition], close_price: float, current_bar: int
) -> float:
    """
    Sum of unrealised P&L for OPEN positions at close_price (A1 MTM).

    Positions with entry_bar > current_bar have not yet been entered
    (signalled this bar, fill next bar) and are excluded.
    """
    total = 0.0
    for pos in positions:
        if pos.entry_bar <= current_bar:
            units = pos.notional / pos.entry_price
            total += (close_price - pos.entry_price) * units
    return total


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
    equity_curve[i] = realised_cash + unrealised_pnl_of_open_positions  (A1 MTM).
    """
    df_ind = _add_indicators(df, cfg)
    n = len(df_ind)

    realised_equity = cfg.initial_equity   # cash after closed trades
    equity_curve: list[float] = [realised_equity] * n
    positions: list[_OpenPosition] = []
    trades: list[Trade] = []

    warmup = cfg.sma_trend + 10

    for i in range(n):
        row = df_ind.iloc[i]
        cost = cfg.cost_per_side(asset_class)

        # ── Update trailing stops in-flight ──────────────────────────────────
        for pos in positions:
            if cfg.exit_mode == "trailing" and pos.trail_stop > 0:
                atr_val = row["atr"] if np.isfinite(row["atr"]) else 0.0
                new_trail = row["close"] - cfg.trail_atr_multiple * atr_val
                if new_trail > pos.trail_stop:
                    pos.trail_stop = new_trail

        # ── Check exits ───────────────────────────────────────────────────────
        for pos in list(positions):
            exit_price: Optional[float] = None
            exit_reason: str = ""
            pos_cost = cfg.cost_per_side(pos.asset_class)

            # 1. Stop loss — with gap-through check (A2)
            if row["low"] <= pos.stop_price:
                raw = _gap_fill(row["open"], pos.stop_price, "long")
                exit_price = raw * (1.0 - pos_cost)
                exit_reason = "stop_loss"

            # 1b. Fixed profit target (any exit mode, checked after stop loss)
            if exit_price is None and cfg.profit_target_pct > 0:
                target_price = pos.entry_price * (1.0 + cfg.profit_target_pct / 100.0)
                if row["high"] >= target_price:
                    raw = row["open"] if row["open"] >= target_price else target_price
                    exit_price = raw * (1.0 - pos_cost)
                    exit_reason = "profit_target"

            # 2. Trailing stop — with gap-through check (A2)
            if exit_price is None and cfg.exit_mode == "trailing" and pos.trail_stop > 0:
                if row["low"] <= pos.trail_stop:
                    raw = _gap_fill(row["open"], pos.trail_stop, "long")
                    exit_price = raw * (1.0 - pos_cost)
                    exit_reason = "trail_stop"

            # 3. Time limit → fill at next bar's open
            if exit_price is None and cfg.max_hold_days > 0 and (i - pos.entry_bar) >= cfg.max_hold_days:
                if i + 1 < n:
                    exit_price = df_ind.iloc[i + 1]["open"] * (1.0 - pos_cost)
                else:
                    exit_price = row["close"] * (1.0 - pos_cost)
                exit_reason = "time_limit"

            # 4. Mean-reversion TP → fill at next bar's open
            if exit_price is None and cfg.exit_mode == "mean_reversion":
                rsi_val = row.get("rsi", np.nan)
                if np.isfinite(rsi_val) and rsi_val >= cfg.tp_rsi_level:
                    if i + 1 < n:
                        exit_price = df_ind.iloc[i + 1]["open"] * (1.0 - pos_cost)
                    else:
                        exit_price = row["close"] * (1.0 - pos_cost)
                    exit_reason = "tp_rsi"

            # 5. Trend break → fill at next bar's open
            if exit_price is None and cfg.exit_mode == "trend_break":
                ema50_val = row.get("ema50", np.nan)
                if np.isfinite(ema50_val) and row["close"] < ema50_val:
                    if i + 1 < n:
                        exit_price = df_ind.iloc[i + 1]["open"] * (1.0 - pos_cost)
                    else:
                        exit_price = row["close"] * (1.0 - pos_cost)
                    exit_reason = "trend_break"

            if exit_price is not None:
                holding_days = i - pos.entry_bar
                # A3: deduct overnight carry from P&L (weekend nights count 3x)
                carry_nights = _weighted_carry_nights(pos.entry_date, df_ind.index[i], holding_days)
                carry_cost = pos.notional * cfg.daily_carry(pos.asset_class) * carry_nights
                units = pos.notional / pos.entry_price
                raw_pnl = (exit_price - pos.entry_price) * units
                pnl = raw_pnl - carry_cost
                pnl_pct = pnl / pos.notional * 100.0
                realised_equity += pnl
                trades.append(Trade(
                    symbol=pos.symbol,
                    entry_date=pos.entry_date,
                    exit_date=df_ind.index[i],
                    entry_bar=pos.entry_bar,
                    exit_bar=i,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    notional=pos.notional,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_days=holding_days,
                    exit_reason=exit_reason,
                    signal_type=pos.signal_type,
                    asset_class=pos.asset_class,
                    carry_cost=carry_cost,
                ))
                positions.remove(pos)

        # ── Activate trailing stop when profitable ────────────────────────────
        if cfg.exit_mode == "trailing":
            atr_val = row["atr"] if np.isfinite(row["atr"]) else 0.0
            for pos in positions:
                if pos.trail_stop == 0 and pos.entry_bar <= i:
                    gain = row["close"] - pos.entry_price
                    if gain >= 0.5 * atr_val:
                        pos.trail_stop = row["close"] - cfg.trail_atr_multiple * atr_val

        # ── Entry signal ──────────────────────────────────────────────────────
        if i >= warmup and i < n - 1 and len(positions) < cfg.max_positions:
            open_symbols = {p.symbol for p in positions}
            if symbol not in open_symbols:
                sig = _entry_signal(row, cfg, asset_class=asset_class)
                if sig:
                    next_row = df_ind.iloc[i + 1]
                    entry_price = next_row["open"] * (1.0 + cost)
                    stop_dist = row["atr"] * cfg.atr_stop_multiple
                    if stop_dist > 0 and np.isfinite(stop_dist):
                        stop_price = entry_price - stop_dist
                        risk_amount = realised_equity * (cfg.risk_per_trade_pct / 100.0)
                        units = risk_amount / stop_dist
                        notional = units * entry_price
                        max_notional = realised_equity * (cfg.max_notional_pct * cfg.leverage / 100.0)
                        notional = min(notional, max_notional)
                        if notional >= 1.0 and realised_equity > 0:
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

        # ── A1: MTM equity curve ──────────────────────────────────────────────
        # Unrealised P&L uses close[i] of positions that have actually entered
        # (entry_bar <= i). Positions signalled this bar (entry_bar = i+1) excluded.
        unrealised = _unrealised_pnl(positions, row["close"], i)
        equity_curve[i] = realised_equity + unrealised

    # ── Force-close any remaining open positions at end of data ───────────────
    last_row = df_ind.iloc[-1]
    for pos in positions:
        pos_cost = cfg.cost_per_side(pos.asset_class)
        last_price = last_row["close"] * (1.0 - pos_cost)
        holding_days = n - 1 - pos.entry_bar
        carry_nights = _weighted_carry_nights(pos.entry_date, df_ind.index[-1], holding_days)
        carry_cost = pos.notional * cfg.daily_carry(pos.asset_class) * carry_nights
        units = pos.notional / pos.entry_price
        raw_pnl = (last_price - pos.entry_price) * units
        pnl = raw_pnl - carry_cost
        pnl_pct = pnl / pos.notional * 100.0
        realised_equity += pnl
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
            holding_days=holding_days,
            exit_reason="end_of_data",
            signal_type=pos.signal_type,
            asset_class=pos.asset_class,
            carry_cost=carry_cost,
        ))
    # After force-close, all positions closed — realised == MTM
    equity_curve[-1] = realised_equity

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
