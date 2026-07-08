"""
Portfolio-level backtest engine — simulates the WHOLE live account, not one
symbol at a time.

Why this exists (vs. engine.py)
================================
engine.py's `run()` backtests every symbol independently, each with its own
fresh `initial_equity`. That's fine for comparing signal/exit rules across
symbols, but it silently assumes unlimited concurrent capital: the aggregate
P&L across N symbols implies you could open all N signals at once with no
shared balance. Real trading doesn't work that way — one account, one equity
curve, and the live bot enforces MAX_OPEN_POSITIONS, MAX_PORTFOLIO_RISK_PCT,
DAILY_LOSS_LIMIT_PCT, and an account-drawdown risk throttle
(src/agents/risk_gate.py, src/core/state.py). None of that is modeled by
engine.py.

This module runs ONE simulation across ALL symbols simultaneously, on a
single shared equity curve, applying the exact same portfolio-level rules
as production:
  - MAX_OPEN_POSITIONS concurrent positions across the whole universe
  - MAX_PORTFOLIO_RISK_PCT cap on summed money-at-risk of open positions
  - DAILY_LOSS_LIMIT_PCT reactive block for the rest of the day
  - Account drawdown hard stop: risk-per-trade drops to reduced_risk_pct once
    equity falls ACCOUNT_DRAWDOWN_HARD_STOP_PCT below its all-time high
  - Time-based exit measured in BARS held (matches engine.py's max_hold_days
    exactly: 20 bars = 20 trading days ≈ 28 calendar days). Production's
    SWING_HARD_EXIT_DAYS currently counts calendar days instead (~14 trading
    days) — a real, documented divergence from the backtest; this engine
    uses the bar-count convention so the two stay comparable, and the fix
    needed on the live side is to make Position.is_past_hard_exit count
    trading days/bars the same way (see src/core/state.py).

Indicator computation and the entry-signal rule are REUSED from engine.py
(_add_indicators, _entry_signal) so the actual trading logic is identical —
only the portfolio bookkeeping around it changes.

Signal timing matches engine.py: a signal detected at day D's close fills at
day D+1's open (mirrors production's scan-before-open / execute-at-open).
Exits (stop-loss, time-limit, trend-break) evaluate and fill same-day, using
day D's own bar — matching the live PositionReviewAgent's once-daily check
that closes immediately rather than waiting for the next print.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.engine import (
    BacktestConfig,
    _add_indicators,
    _entry_signal,
    _gap_fill,
    _weighted_carry_nights,
)
from src.core.sector_map import classify_sector

logger = logging.getLogger(__name__)


def _sector_key(symbol: str) -> str:
    """
    Sector bucket used for the per-sector concurrency cap. Unclassified
    ("Other") symbols get their own unique key (the symbol itself) so the
    cap only constrains sectors we can actually identify — ~77% of this
    universe's names don't match any keyword (foreign-language names,
    bare tickers, etc.), and treating all of those as one giant "Other"
    sector would be an artificial, meaningless restriction.
    """
    sector = classify_sector(symbol)
    return symbol if sector == "Other" else sector


def _conviction_score(row: pd.Series) -> float:
    """
    Deterministic proxy for "conviction" used to prioritize which signals
    get the available position slots when more symbols qualify than there's
    room for. thesis_builder.py currently assigns a fixed confidence (0.75)
    to every thesis, so there's no existing per-signal ranking to reuse —
    this combines the two things the entry rule already conditions on:
    relative volume (how unusual today's participation is) and trend
    strength (how extended EMA50 is above EMA200), rewarding signals with
    stronger confirmation on both.
    """
    rel_vol = row.get("rel_vol", np.nan)
    rel_vol_score = rel_vol if np.isfinite(rel_vol) else 1.0
    ema50 = row.get("ema50", np.nan)
    ema200 = row.get("ema200", np.nan)
    if np.isfinite(ema50) and np.isfinite(ema200) and ema200 != 0:
        trend_strength = max((ema50 - ema200) / ema200, 0.0)
    else:
        trend_strength = 0.0
    return rel_vol_score * (1.0 + trend_strength)


@dataclass
class PortfolioConfig:
    # Signal generation (passed through to a BacktestConfig for _add_indicators/_entry_signal)
    use_trend_filter: bool = True
    trend_filter_type: str = "ema50_200"
    use_rsi_signal: bool = False
    use_ema_signal: bool = False
    use_breakout_signal: bool = True
    use_pullback_signal: bool = True
    donchian_lookback: int = 20
    vol_multiplier: float = 1.5

    # Sizing — mirrors .env / execution_agent.py exactly
    initial_equity: float = 800.0
    risk_per_trade_pct: float = 8.0
    atr_stop_multiple: float = 3.5
    max_notional_pct: float = 13.0
    leverage: float = 3.0

    # Portfolio-level rules — mirrors risk_gate.py / state.py exactly
    max_open_positions: int = 5
    max_portfolio_risk_pct: float = 40.0
    daily_loss_limit_pct: float = 3.0
    account_drawdown_hard_stop_pct: float = 10.0
    reduced_risk_pct: float = 3.0
    max_positions_per_sector: int = 2   # caps correlated exposure across similar names
                                         # ("Other"/unclassified symbols are exempt — see _sector_key)
    max_hold_bars: int = 20              # BARS held (trading days), matching engine.py exactly

    # Costs — same per-side model as engine.py's "equity" asset class
    equity_spread_pct: float = 0.07
    equity_slippage_pct: float = 0.03
    equity_commission_pct: float = 0.0
    equity_carry_daily_pct: float = 0.01

    def cost_per_side(self) -> float:
        return (self.equity_spread_pct / 2 + self.equity_slippage_pct +
                self.equity_commission_pct) / 100.0

    def daily_carry(self) -> float:
        return self.equity_carry_daily_pct / 100.0

    def to_signal_config(self) -> BacktestConfig:
        """Build the BacktestConfig used only for _add_indicators/_entry_signal."""
        return BacktestConfig(
            use_trend_filter=self.use_trend_filter,
            trend_filter_type=self.trend_filter_type,
            use_rsi_signal=self.use_rsi_signal,
            use_ema_signal=self.use_ema_signal,
            use_breakout_signal=self.use_breakout_signal,
            use_pullback_signal=self.use_pullback_signal,
            donchian_lookback=self.donchian_lookback,
            vol_multiplier=self.vol_multiplier,
            atr_stop_multiple=self.atr_stop_multiple,
        )


@dataclass
class PortfolioPosition:
    symbol: str
    entry_date: pd.Timestamp
    entry_bar_index: int        # this symbol's own bar index at fill — time-limit counts
                                 # BARS held (matches engine.py), not calendar days
    entry_price: float          # after entry-side cost
    stop_price: float
    notional: float
    risk_amount: float          # $ at risk if stop is hit — notional * (stop_distance/entry_price)
    risk_pct_used: float        # effective risk-per-trade % active when this was opened


@dataclass
class PortfolioTrade:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    notional: float
    pnl: float
    pnl_pct: float
    holding_days: int
    exit_reason: str
    risk_pct_used: float         # risk-per-trade % actually applied (may be the throttled value)
    risk_amount: float = 0.0     # $ actually at risk at entry (post notional-cap) — the real
                                 # figure Rule 7b checks, which can be less than risk_pct_used
                                 # implies if the notional cap bound before the risk target did


@dataclass
class PortfolioResult:
    trades: list[PortfolioTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    dates: list[pd.Timestamp] = field(default_factory=list)
    initial_equity: float = 800.0
    daily_loss_blocks: int = 0        # how many days new entries were blocked
    drawdown_throttle_days: int = 0   # how many days risk was throttled to reduced_risk_pct


def _prepare_symbol_frames(
    data: dict[str, pd.DataFrame], cfg: BacktestConfig
) -> dict[str, pd.DataFrame]:
    """Compute indicators once per symbol (reuses engine.py's exact logic)."""
    out = {}
    for symbol, df in data.items():
        if df is None or len(df) < cfg.sma_trend + 10:
            continue
        out[symbol] = _add_indicators(df, cfg)
    return out


def run_portfolio(
    data: dict[str, pd.DataFrame],
    cfg: PortfolioConfig,
) -> PortfolioResult:
    """
    Run one shared-equity simulation across every symbol in `data`
    (symbol -> raw OHLCV DataFrame, DatetimeIndex ascending).
    """
    sig_cfg = cfg.to_signal_config()
    frames = _prepare_symbol_frames(data, sig_cfg)

    # Master timeline: every date that appears in ANY symbol's data.
    all_dates = sorted(set().union(*[set(df.index) for df in frames.values()])) if frames else []

    cash = cfg.initial_equity
    open_positions: dict[str, PortfolioPosition] = {}
    trades: list[PortfolioTrade] = []
    equity_curve: list[float] = []
    dates_out: list[pd.Timestamp] = []

    peak_equity = cfg.initial_equity
    daily_loss_blocks = 0
    drawdown_throttle_days = 0
    cost = cfg.cost_per_side()

    # Per-symbol lookup of row-position-by-date for O(1) "is there a next day" checks.
    date_pos = {sym: {d: i for i, d in enumerate(df.index)} for sym, df in frames.items()}

    for date in all_dates:
        # ── 1. Mark today's rows for symbols that traded today ─────────────
        todays_row = {}
        for sym, df in frames.items():
            idx = date_pos[sym].get(date)
            if idx is not None:
                todays_row[sym] = (df, idx)

        # ── 2. Process exits (same-day fill: stop -> time-limit -> trend-break) ──
        realized_pnl_today = 0.0
        for sym in list(open_positions.keys()):
            if sym not in todays_row:
                continue  # symbol didn't trade today (holiday/no data) — hold as-is
            df, i = todays_row[sym]
            row = df.iloc[i]
            pos = open_positions[sym]
            exit_price: Optional[float] = None
            exit_reason = ""

            if row["low"] <= pos.stop_price:
                raw = _gap_fill(row["open"], pos.stop_price, "long")
                exit_price = raw * (1.0 - cost)
                exit_reason = "stop_loss"

            if exit_price is None and cfg.max_hold_bars > 0:
                bars_held = i - pos.entry_bar_index
                if bars_held >= cfg.max_hold_bars:
                    exit_price = row["close"] * (1.0 - cost)
                    exit_reason = "time_limit"

            if exit_price is None:
                ema50_val = row.get("ema50", np.nan)
                if np.isfinite(ema50_val) and row["close"] < ema50_val:
                    exit_price = row["close"] * (1.0 - cost)
                    exit_reason = "trend_break"

            if exit_price is not None:
                holding_days = (date - pos.entry_date).days
                carry_nights = _weighted_carry_nights(pos.entry_date, date, holding_days)
                carry_cost = pos.notional * cfg.daily_carry() * carry_nights
                units = pos.notional / pos.entry_price
                raw_pnl = (exit_price - pos.entry_price) * units
                pnl = raw_pnl - carry_cost
                pnl_pct = pnl / pos.notional * 100.0

                margin = pos.notional / cfg.leverage
                cash += margin + pnl
                realized_pnl_today += pnl

                trades.append(PortfolioTrade(
                    symbol=sym, entry_date=pos.entry_date, exit_date=date,
                    entry_price=pos.entry_price, exit_price=exit_price,
                    notional=pos.notional, pnl=pnl, pnl_pct=pnl_pct,
                    holding_days=holding_days, exit_reason=exit_reason,
                    risk_pct_used=pos.risk_pct_used,
                    risk_amount=pos.risk_amount,
                ))
                del open_positions[sym]

        # ── 3. Mark-to-market equity (cash + margin held + unrealized P&L) ──
        def _equity_now() -> float:
            margin_deployed = 0.0
            unrealized = 0.0
            for s, p in open_positions.items():
                margin_deployed += p.notional / cfg.leverage
                if s in todays_row:
                    df, i = todays_row[s]
                    price = df.iloc[i]["close"]
                else:
                    price = p.entry_price  # no print today — carry forward at entry (conservative)
                units = p.notional / p.entry_price
                unrealized += (price - p.entry_price) * units
            return cash + margin_deployed + unrealized

        equity_now = _equity_now()
        if equity_now > peak_equity:
            peak_equity = equity_now
        drawdown_pct = ((peak_equity - equity_now) / peak_equity * 100.0) if peak_equity > 0 else 0.0

        # ── 4. Daily loss block (Rule 5) — reactive, same-day only ──────────
        loss_pct_today = (-realized_pnl_today / equity_now * 100.0) if equity_now > 0 else 0.0
        blocked_today = loss_pct_today >= cfg.daily_loss_limit_pct
        if blocked_today:
            daily_loss_blocks += 1

        # ── 5. Account drawdown hard stop (throttle risk-per-trade) ─────────
        if drawdown_pct >= cfg.account_drawdown_hard_stop_pct:
            effective_risk_pct = cfg.reduced_risk_pct
            drawdown_throttle_days += 1
        else:
            effective_risk_pct = cfg.risk_per_trade_pct

        # ── 6. Entries — priority-queued by conviction, capped by sector ────
        if not blocked_today:
            open_risk_pct = (
                sum(p.risk_amount for p in open_positions.values()) / equity_now * 100.0
                if equity_now > 0 else 100.0
            )
            open_sector_counts: dict[str, int] = {}
            for p in open_positions.values():
                key = _sector_key(p.symbol)
                open_sector_counts[key] = open_sector_counts.get(key, 0) + 1

            # Pass 1: collect every symbol that signals today, so more-convinced
            # candidates get first claim on the limited slots (queueing) instead
            # of whichever symbol happened to sort first alphabetically.
            candidates = []
            for sym, (df, i) in todays_row.items():
                if sym in open_positions or i + 1 >= len(df):
                    continue
                row = df.iloc[i]
                sig = _entry_signal(row, sig_cfg, asset_class="equity")
                if not sig:
                    continue
                stop_dist = row["atr"] * cfg.atr_stop_multiple
                if not (stop_dist > 0 and np.isfinite(stop_dist)):
                    continue
                candidates.append((sym, df, i, row, stop_dist, _conviction_score(row)))
            candidates.sort(key=lambda c: c[5], reverse=True)

            # Pass 2: fill slots in conviction order, subject to every cap.
            for sym, df, i, row, stop_dist, _score in candidates:
                if len(open_positions) >= cfg.max_open_positions:
                    break

                sector_key = _sector_key(sym)
                if open_sector_counts.get(sector_key, 0) >= cfg.max_positions_per_sector:
                    continue  # sector concentration cap

                fill_date = df.index[i + 1]
                next_open = df.iloc[i + 1]["open"]
                entry_price = next_open * (1.0 + cost)
                risk_amount = equity_now * (effective_risk_pct / 100.0)
                units = risk_amount / stop_dist
                notional = units * entry_price
                max_notional = equity_now * (cfg.max_notional_pct * cfg.leverage / 100.0)
                notional = min(notional, max_notional)
                actual_risk_amount = notional * (stop_dist / entry_price)
                trade_risk_pct = (actual_risk_amount / equity_now * 100.0) if equity_now > 0 else 0.0

                if open_risk_pct + trade_risk_pct > cfg.max_portfolio_risk_pct:
                    continue  # Rule 7b — aggregate portfolio-risk cap

                margin = notional / cfg.leverage
                if margin > cash:
                    continue  # not enough free cash to post margin

                cash -= margin
                open_sector_counts[sector_key] = open_sector_counts.get(sector_key, 0) + 1
                open_positions[sym] = PortfolioPosition(
                    symbol=sym, entry_date=fill_date, entry_bar_index=i + 1,
                    entry_price=entry_price,
                    stop_price=entry_price - stop_dist, notional=notional,
                    risk_amount=actual_risk_amount, risk_pct_used=effective_risk_pct,
                )
                open_risk_pct += trade_risk_pct

        equity_curve.append(_equity_now())
        dates_out.append(date)

    return PortfolioResult(
        trades=trades, equity_curve=equity_curve, dates=dates_out,
        initial_equity=cfg.initial_equity,
        daily_loss_blocks=daily_loss_blocks,
        drawdown_throttle_days=drawdown_throttle_days,
    )


@dataclass
class PortfolioMetrics:
    label: str
    n_trades: int
    win_rate: float
    total_pnl: float
    profit_factor: float
    max_drawdown_pct: float
    start_equity: float
    end_equity: float
    exit_reasons: dict


def _max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak * 100.0
            max_dd = max(max_dd, dd)
    return max_dd


def summarize(result: PortfolioResult, trades: list[PortfolioTrade], label: str) -> PortfolioMetrics:
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0
    exit_reasons: dict = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return PortfolioMetrics(
        label=label,
        n_trades=len(trades),
        win_rate=(len(wins) / len(trades) * 100.0) if trades else 0.0,
        total_pnl=sum(t.pnl for t in trades),
        profit_factor=pf,
        max_drawdown_pct=_max_drawdown(result.equity_curve),
        start_equity=result.initial_equity,
        end_equity=result.equity_curve[-1] if result.equity_curve else result.initial_equity,
        exit_reasons=exit_reasons,
    )


def split_is_oos(
    result: PortfolioResult, split_ratio: float = 0.7
) -> tuple[PortfolioMetrics, PortfolioMetrics]:
    """
    Split by calendar time (not trade count): trades that OPENED before the
    split date are in-sample, the rest out-of-sample. The equity curve is one
    continuous simulation (the account never resets), so IS/OOS here means
    "which period this trade's entry falls in", not two separate runs.
    """
    if not result.dates:
        empty = summarize(result, [], "empty")
        return empty, empty

    start, end = result.dates[0], result.dates[-1]
    split_date = start + (end - start) * split_ratio

    is_trades = [t for t in result.trades if t.entry_date < split_date]
    oos_trades = [t for t in result.trades if t.entry_date >= split_date]

    is_idx = sum(1 for d in result.dates if d < split_date)
    is_curve = result.equity_curve[:is_idx] or [result.initial_equity]
    oos_curve = result.equity_curve[is_idx:] or [is_curve[-1]]

    is_result = PortfolioResult(equity_curve=is_curve, dates=result.dates[:is_idx],
                                 initial_equity=result.initial_equity)
    oos_result = PortfolioResult(equity_curve=oos_curve, dates=result.dates[is_idx:],
                                  initial_equity=is_curve[-1])

    return (
        summarize(is_result, is_trades, "IN-SAMPLE"),
        summarize(oos_result, oos_trades, "OUT-OF-SAMPLE"),
    )
