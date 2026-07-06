"""
Market Structure (Smart Money Concepts) — BOS/ChoCh backtest.

Long-only, daily-bar implementation of the user's spec:
  - Swing High (SH) / Swing Low (SL): a fractal pivot, higher/lower than
    `swing_k` bars on each side. Confirmed `swing_k` bars after it forms
    (no look-ahead: at bar i we only know about pivots up to bar i-swing_k).
  - Trend state from the last two confirmed swings:
      bullish = higher high (HH) AND higher low (HL)
      bearish = lower high (LH) AND lower low (LL)
      else    = range (no trade)
  - BOS (Break of Structure): close > last swing high while bullish —
    confirms the trend is still alive. We do NOT enter here.
  - Pullback entry: after a BOS, wait for price to pull back down to the
    broken level (old resistance -> new support) and close back above it —
    that candle is the entry trigger (next bar open fill).
  - Stop-loss: at the last confirmed swing low (the HL) that preceded the
    BOS. If price re-closes below that low before entering, the setup is
    invalidated (ChoCh) and we stop waiting.
  - Exit: ChoCh while in a position (close < the swing low the stop is
    based on) exits immediately; the stop also ratchets up to each new,
    higher confirmed swing low (never down); `max_hold_days` is a backstop.

Reuses BacktestConfig (costs/carry), Trade/RunResult and the gap-fill/
carry helpers from engine.py — this strategy is long-only and swing-horizon
like the main engine, unlike First Red Day's short/day-trade mechanics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import (
    BacktestConfig,
    Trade,
    RunResult,
    _OpenPosition,
    _gap_fill,
    _weighted_carry_nights,
)

logger = logging.getLogger(__name__)


@dataclass
class MSConfig:
    swing_k: int = 2               # bars each side required to confirm a pivot
    max_hold_days: int = 20        # backstop, matches the account's swing horizon

    # Sizing (same semantics as BacktestConfig)
    initial_equity: float = 10_000.0
    risk_per_trade_pct: float = 1.0
    max_notional_pct: float = 10.0
    leverage: float = 1.0
    max_positions: int = 3


def _confirmed_swings(df: pd.DataFrame, k: int) -> tuple[list[bool], list[bool]]:
    """Fractal swing high/low flags — True at bar i if high[i]/low[i] is the
    extreme of the window [i-k, i+k]. Only used bar-by-bar once i+k has
    passed in the main loop, so this is not a look-ahead despite being
    precomputed over the whole series."""
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
    return is_sh, is_sl


def run_market_structure(
    df: pd.DataFrame,
    ms_cfg: MSConfig,
    cost_cfg: BacktestConfig,
    symbol: str = "?",
    asset_class: str = "equity",
) -> RunResult:
    df = df.copy()
    n = len(df)
    k = ms_cfg.swing_k
    is_sh, is_sl = _confirmed_swings(df, k)
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    opens = df["open"].to_numpy()

    realised_equity = ms_cfg.initial_equity
    equity_curve: list[float] = [realised_equity] * n
    positions: list[_OpenPosition] = []
    trades: list[Trade] = []
    cost = cost_cfg.cost_per_side(asset_class)

    # Confirmed swing history: list of (bar_index, price)
    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []
    trend = "range"

    awaiting_pullback = False
    bos_level: float = 0.0
    stop_ref_low: float = 0.0   # the HL that will become the stop if we enter

    warmup = 2 * k + 5

    for i in range(warmup, n):
        # ── Reveal any newly-confirmed swing at bar i-k ──────────────────────
        confirm_idx = i - k
        if is_sh[confirm_idx]:
            swing_highs.append((confirm_idx, highs[confirm_idx]))
        if is_sl[confirm_idx]:
            swing_lows.append((confirm_idx, lows[confirm_idx]))

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = swing_highs[-1][1] > swing_highs[-2][1]
            hl = swing_lows[-1][1] > swing_lows[-2][1]
            lh = swing_highs[-1][1] < swing_highs[-2][1]
            ll = swing_lows[-1][1] < swing_lows[-2][1]
            if hh and hl:
                trend = "bullish"
            elif lh and ll:
                trend = "bearish"
            else:
                trend = "range"

        last_sh = swing_highs[-1][1] if swing_highs else None
        last_sl = swing_lows[-1][1] if swing_lows else None

        # ── Manage open position ─────────────────────────────────────────────
        for pos in list(positions):
            exit_price: Optional[float] = None
            exit_reason = ""

            # Ratchet stop up to the newest confirmed swing low (never down)
            if last_sl is not None and last_sl > pos.stop_price:
                pos.stop_price = last_sl

            if lows[i] <= pos.stop_price:
                raw = _gap_fill(opens[i], pos.stop_price, direction="long")
                exit_price = raw * (1.0 - cost)
                exit_reason = "stop_loss"
            elif last_sl is not None and closes[i] < last_sl:
                # ChoCh: structure broken while in position
                if i + 1 < n:
                    exit_price = opens[i + 1] * (1.0 - cost)
                else:
                    exit_price = closes[i] * (1.0 - cost)
                exit_reason = "choch"
            elif (i - pos.entry_bar) >= ms_cfg.max_hold_days:
                if i + 1 < n:
                    exit_price = opens[i + 1] * (1.0 - cost)
                else:
                    exit_price = closes[i] * (1.0 - cost)
                exit_reason = "time_limit"

            if exit_price is not None:
                holding_days = i - pos.entry_bar
                carry_nights = _weighted_carry_nights(pos.entry_date, df.index[i], holding_days)
                carry_cost = pos.notional * cost_cfg.daily_carry(pos.asset_class) * carry_nights
                units = pos.notional / pos.entry_price
                raw_pnl = (exit_price - pos.entry_price) * units
                pnl = raw_pnl - carry_cost
                pnl_pct = pnl / pos.notional * 100.0
                realised_equity += pnl
                trades.append(Trade(
                    symbol=symbol,
                    entry_date=pos.entry_date,
                    exit_date=df.index[i],
                    entry_bar=pos.entry_bar,
                    exit_bar=i,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    notional=pos.notional,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_days=holding_days,
                    exit_reason=exit_reason,
                    signal_type="market_structure",
                    asset_class=asset_class,
                    carry_cost=carry_cost,
                ))
                positions.remove(pos)

        # ── Entry state machine (only when flat) ─────────────────────────────
        if not positions and len(positions) < ms_cfg.max_positions:
            if trend == "bullish" and last_sh is not None and not awaiting_pullback:
                if closes[i] > last_sh:   # BOS
                    awaiting_pullback = True
                    bos_level = last_sh
                    stop_ref_low = last_sl if last_sl is not None else lows[i]

            elif awaiting_pullback:
                if last_sl is not None and closes[i] < last_sl:
                    # Setup invalidated before we ever entered
                    awaiting_pullback = False
                elif lows[i] <= bos_level and closes[i] > bos_level:
                    # Pullback to the broken level + resumption -> enter next open
                    if i + 1 < n:
                        entry_price = opens[i + 1] * (1.0 + cost)
                        stop_price = stop_ref_low
                        stop_dist = entry_price - stop_price
                        if stop_dist > 0 and realised_equity > 0:
                            risk_amount = realised_equity * (ms_cfg.risk_per_trade_pct / 100.0)
                            units = risk_amount / stop_dist
                            notional = units * entry_price
                            max_notional = realised_equity * (ms_cfg.max_notional_pct * ms_cfg.leverage / 100.0)
                            notional = min(notional, max_notional)
                            if notional >= 1.0:
                                positions.append(_OpenPosition(
                                    symbol=symbol,
                                    entry_bar=i + 1,
                                    entry_date=df.index[i + 1],
                                    entry_price=entry_price,
                                    stop_price=stop_price,
                                    notional=notional,
                                    asset_class=asset_class,
                                    signal_type="market_structure",
                                ))
                    awaiting_pullback = False

        # ── MTM equity curve ──────────────────────────────────────────────────
        unrealised = 0.0
        for pos in positions:
            if pos.entry_bar <= i:
                units = pos.notional / pos.entry_price
                unrealised += (closes[i] - pos.entry_price) * units
        equity_curve[i] = realised_equity + unrealised

    # ── Force-close any remaining open position at end of data ──────────────
    last_close = closes[-1]
    for pos in positions:
        pos_cost = cost_cfg.cost_per_side(pos.asset_class)
        last_price = last_close * (1.0 - pos_cost)
        holding_days = n - 1 - pos.entry_bar
        carry_nights = _weighted_carry_nights(pos.entry_date, df.index[-1], holding_days)
        carry_cost = pos.notional * cost_cfg.daily_carry(pos.asset_class) * carry_nights
        units = pos.notional / pos.entry_price
        raw_pnl = (last_price - pos.entry_price) * units
        pnl = raw_pnl - carry_cost
        pnl_pct = pnl / pos.notional * 100.0
        realised_equity += pnl
        trades.append(Trade(
            symbol=symbol,
            entry_date=pos.entry_date,
            exit_date=df.index[-1],
            entry_bar=pos.entry_bar,
            exit_bar=n - 1,
            entry_price=pos.entry_price,
            exit_price=last_price,
            notional=pos.notional,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=holding_days,
            exit_reason="end_of_data",
            signal_type="market_structure",
            asset_class=asset_class,
            carry_cost=carry_cost,
        ))

    return RunResult(
        trades=trades,
        equity_curve=equity_curve,
        bar_dates=list(df.index),
        initial_equity=ms_cfg.initial_equity,
        n_bars=n,
        period_label=symbol,
    )
