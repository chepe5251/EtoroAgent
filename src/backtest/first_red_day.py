"""
First Red Day — short-side day/swing strategy backtest.

Concept (per user's source video):
  1. Find a stock that has moved parabolically over the last N days
     (unusually large, fast cumulative gain).
  2. Don't guess the top. Wait for confirmation: the FIRST day the stock's
     price breaks below the PREVIOUS day's close.
  3. Short at that break, on the theory that trapped late-buyers panic-sell
     into an accelerating drop.

This is a SEPARATE engine from src/backtest/engine.py (which is long-only,
swing-horizon, daily-bar trend-following) because the mechanics are
fundamentally different:
  - Short, not long (stop above entry, profit below entry).
  - Entry triggers and fills on the SAME bar (a resting stop-market order
    triggered by the previous close being broken intraday) rather than
    "signal at close of bar i, fill at open of bar i+1" — this mirrors how
    the long engine already fills its own stop-loss exits off the same
    bar's low, so it's consistent with the codebase's no-look-ahead
    standard, not a new exception to it.
  - Meant to be held only a few days (max_hold_days), not a 5-20 day swing.

HONESTY NOTE ON ASSUMPTIONS
============================
  - Built on DAILY bars, not intraday. The video's entry trigger ("loses the
    previous day's close") is evaluated using the day's low vs. yesterday's
    close, filled via the same gap-through logic already used for long
    stop-losses. This is a reasonable daily-bar approximation of an
    intraday trigger, but a live version watching price tick-by-tick could
    get better fills.
  - Short-side financing fees on eToro are NOT verified against a real fee
    schedule (unlike the long-side crypto/equity costs in engine.py, which
    were researched against eToro's published fee page). `short_borrow_daily_pct`
    below is a placeholder assumption layered on top of the normal carry
    rate — flag this before trusting the $ P&L numbers.
  - This targets small/mid-cap "parabolic mover" stocks, which are NOT what
    your current 119-symbol universe (large blue-chips) contains. Backtests
    run against your existing universe are a mechanics sanity-check, not a
    fair test of the strategy's real edge — real validation needs a feed of
    actual small-cap gappers/runners.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import (
    Trade,
    RunResult,
    _OpenPosition,
    _gap_fill,
    _weighted_carry_nights,
)

logger = logging.getLogger(__name__)


@dataclass
class FRDConfig:
    # Parabolic-run detection
    parabolic_lookback_days: int = 3      # window to measure the run-up
    parabolic_min_return_pct: float = 10.0  # % gain over the window to qualify

    # Liquidity / quality filters — avoid illiquid, hard-to-fill names
    min_price: float = 15.0               # $ minimum share price
    min_dollar_volume: float = 30_000_000.0  # $ minimum 20-day avg daily $ volume

    # Entry / exit
    stop_loss_pct: float = 8.0            # stop ABOVE entry, % of entry price
    profit_target_pct: float = 15.0       # 0 = disabled; take-profit BELOW entry
    max_hold_days: int = 5                # this is a fast trade, not a 5-20d swing

    # Sizing (same semantics as BacktestConfig)
    initial_equity: float = 10_000.0
    risk_per_trade_pct: float = 1.0
    max_notional_pct: float = 10.0
    leverage: float = 1.0
    max_positions: int = 3

    # Costs — equity-only (this strategy targets stocks, not crypto)
    spread_pct: float = 0.07
    slippage_pct: float = 0.03
    commission_pct: float = 0.0
    carry_daily_pct: float = 0.01         # baseline overnight financing
    short_borrow_daily_pct: float = 0.02  # ASSUMPTION — not verified against eToro's
                                          # real short-CFD fee schedule; see module docstring

    def cost_per_side(self) -> float:
        return (self.spread_pct + self.slippage_pct + self.commission_pct) / 100.0

    def daily_carry(self) -> float:
        return (self.carry_daily_pct + self.short_borrow_daily_pct) / 100.0


def run_first_red_day(df: pd.DataFrame, cfg: FRDConfig, symbol: str = "?") -> RunResult:
    """Run the First Red Day short backtest on a single symbol's DataFrame."""
    df = df.copy()
    n = len(df)
    lookback = cfg.parabolic_lookback_days
    df["ret_lookback"] = df["close"] / df["close"].shift(lookback) - 1.0
    df["dollar_vol_20d"] = (df["close"] * df["volume"]).rolling(20, min_periods=20).mean()

    realised_equity = cfg.initial_equity
    equity_curve: list[float] = [realised_equity] * n
    positions: list[_OpenPosition] = []
    trades: list[Trade] = []

    cost = cfg.cost_per_side()
    warmup = lookback + 1

    for i in range(warmup, n):
        row = df.iloc[i]
        prev_close = df["close"].iloc[i - 1]

        # ── Check exits (short: stop above, target below) ───────────────────
        for pos in list(positions):
            exit_price: Optional[float] = None
            exit_reason = ""

            if row["high"] >= pos.stop_price:
                raw = _gap_fill(row["open"], pos.stop_price, direction="short")
                exit_price = raw * (1.0 + cost)   # buy-to-cover costs more
                exit_reason = "stop_loss"
            elif cfg.profit_target_pct > 0:
                target_price = pos.entry_price * (1.0 - cfg.profit_target_pct / 100.0)
                if row["low"] <= target_price:
                    raw = row["open"] if row["open"] <= target_price else target_price
                    exit_price = raw * (1.0 + cost)
                    exit_reason = "profit_target"

            if exit_price is None and (i - pos.entry_bar) >= cfg.max_hold_days:
                if i + 1 < n:
                    exit_price = df["open"].iloc[i + 1] * (1.0 + cost)
                else:
                    exit_price = row["close"] * (1.0 + cost)
                exit_reason = "time_limit"

            if exit_price is not None:
                holding_days = i - pos.entry_bar
                carry_nights = _weighted_carry_nights(pos.entry_date, df.index[i], holding_days)
                carry_cost = pos.notional * cfg.daily_carry() * carry_nights
                units = pos.notional / pos.entry_price
                raw_pnl = (pos.entry_price - exit_price) * units   # short: profit when price falls
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
                    signal_type="first_red_day",
                    asset_class="equity",
                    carry_cost=carry_cost,
                    is_short=True,
                ))
                positions.remove(pos)

        # ── Entry: parabolic run confirmed through yesterday, broken today ──
        if not positions:
            # All gating stats use data through i-1 only — no look-ahead.
            ret_lb = df["ret_lookback"].iloc[i - 1]
            dollar_vol = df["dollar_vol_20d"].iloc[i - 1]
            parabolic = np.isfinite(ret_lb) and ret_lb >= cfg.parabolic_min_return_pct / 100.0
            liquid = (
                prev_close >= cfg.min_price
                and np.isfinite(dollar_vol) and dollar_vol >= cfg.min_dollar_volume
            )
            if parabolic and liquid and row["low"] <= prev_close:
                raw = row["open"] if row["open"] <= prev_close else prev_close
                entry_price = raw * (1.0 - cost)   # short entry: sell, receive slightly less
                stop_price = entry_price * (1.0 + cfg.stop_loss_pct / 100.0)
                stop_dist = stop_price - entry_price
                if stop_dist > 0 and realised_equity > 0:
                    risk_amount = realised_equity * (cfg.risk_per_trade_pct / 100.0)
                    units = risk_amount / stop_dist
                    notional = units * entry_price
                    max_notional = realised_equity * (cfg.max_notional_pct * cfg.leverage / 100.0)
                    notional = min(notional, max_notional)
                    if notional >= 1.0:
                        positions.append(_OpenPosition(
                            symbol=symbol,
                            entry_bar=i,
                            entry_date=df.index[i],
                            entry_price=entry_price,
                            stop_price=stop_price,
                            notional=notional,
                            asset_class="equity",
                            signal_type="first_red_day",
                        ))

        # ── MTM equity curve (short: unrealised gains when price falls) ─────
        unrealised = 0.0
        for pos in positions:
            units = pos.notional / pos.entry_price
            unrealised += (pos.entry_price - row["close"]) * units
        equity_curve[i] = realised_equity + unrealised

    # ── Force-close any remaining open position at end of data ──────────────
    last_row = df.iloc[-1]
    for pos in positions:
        last_price = last_row["close"] * (1.0 + cost)
        holding_days = n - 1 - pos.entry_bar
        carry_nights = _weighted_carry_nights(pos.entry_date, df.index[-1], holding_days)
        carry_cost = pos.notional * cfg.daily_carry() * carry_nights
        units = pos.notional / pos.entry_price
        raw_pnl = (pos.entry_price - last_price) * units
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
            signal_type="first_red_day",
            asset_class="equity",
            carry_cost=carry_cost,
            is_short=True,
        ))

    return RunResult(
        trades=trades,
        equity_curve=equity_curve,
        bar_dates=list(df.index),
        initial_equity=cfg.initial_equity,
        n_bars=n,
        period_label=symbol,
    )
