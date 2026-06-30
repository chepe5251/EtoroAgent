"""
Backtest performance metrics.

All metrics are computed from a RunResult produced by engine.py.
No I/O, no network, no LLM.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtest.engine import RunResult, Trade


@dataclass
class PeriodMetrics:
    period_label: str
    n_trades: int
    win_rate: float           # fraction 0..1
    avg_pnl_pct: float        # mean % P&L per trade
    total_pnl: float          # $ sum
    profit_factor: float      # gross_win / gross_loss (inf if no losses)
    max_drawdown_pct: float   # peak-to-trough as % of peak equity
    sharpe: float             # annualised Sharpe (assume 252 trading days)
    avg_hold_days: float
    n_bars: int
    initial_equity: float
    final_equity: float
    exit_reason_counts: dict[str, int]

    def as_dict(self) -> dict:
        return {
            "period": self.period_label,
            "n_trades": self.n_trades,
            "win_rate_pct": round(self.win_rate * 100, 1),
            "avg_pnl_pct": round(self.avg_pnl_pct, 2),
            "total_pnl_usd": round(self.total_pnl, 2),
            "profit_factor": round(self.profit_factor, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe": round(self.sharpe, 3),
            "avg_hold_days": round(self.avg_hold_days, 1),
            "n_bars": self.n_bars,
            "initial_equity": round(self.initial_equity, 2),
            "final_equity": round(self.final_equity, 2),
            "return_pct": round(
                (self.final_equity / self.initial_equity - 1) * 100 if self.initial_equity else 0,
                2,
            ),
            "exit_reason_counts": self.exit_reason_counts,
        }

    def __str__(self) -> str:
        d = self.as_dict()
        lines = [
            f"  [{d['period'].upper()}]  {d['n_bars']} bars",
            f"  Trades:      {d['n_trades']}",
            f"  Win rate:    {d['win_rate_pct']}%",
            f"  Avg P&L:     {d['avg_pnl_pct']:+.2f}% / trade",
            f"  Total P&L:   ${d['total_pnl_usd']:+,.2f}  ({d['return_pct']:+.1f}%)",
            f"  Profit F:    {d['profit_factor']:.2f}",
            f"  Max DD:      {d['max_drawdown_pct']:.2f}%",
            f"  Sharpe:      {d['sharpe']:.3f}",
            f"  Avg hold:    {d['avg_hold_days']:.1f} days",
            f"  Exits:       {d['exit_reason_counts']}",
        ]
        return "\n".join(lines)


def _max_drawdown(equity_curve: list[float]) -> float:
    """Peak-to-trough drawdown as % of peak (positive number)."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _daily_returns(equity_curve: list[float]) -> list[float]:
    """Bar-to-bar % returns from equity curve."""
    returns = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]
        if prev > 0:
            returns.append((equity_curve[i] - prev) / prev)
        else:
            returns.append(0.0)
    return returns


def _sharpe(daily_returns: list[float], periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio (0 risk-free rate)."""
    n = len(daily_returns)
    if n < 2:
        return 0.0
    mean = sum(daily_returns) / n
    var = sum((r - mean) ** 2 for r in daily_returns) / (n - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(periods_per_year)


def compute(result: "RunResult") -> PeriodMetrics:
    """Compute all metrics for a single RunResult period."""
    trades: list["Trade"] = result.trades

    if not trades:
        return PeriodMetrics(
            period_label=result.period_label,
            n_trades=0,
            win_rate=0.0,
            avg_pnl_pct=0.0,
            total_pnl=0.0,
            profit_factor=0.0,
            max_drawdown_pct=_max_drawdown(result.equity_curve),
            sharpe=_sharpe(_daily_returns(result.equity_curve)),
            avg_hold_days=0.0,
            n_bars=result.n_bars,
            initial_equity=result.initial_equity,
            final_equity=result.equity_curve[-1] if result.equity_curve else result.initial_equity,
            exit_reason_counts={},
        )

    wins = [t for t in trades if t.pnl >= 0]
    losses = [t for t in trades if t.pnl < 0]
    win_rate = len(wins) / len(trades)
    avg_pnl_pct = sum(t.pnl_pct for t in trades) / len(trades)
    total_pnl = sum(t.pnl for t in trades)
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    avg_hold = sum(t.holding_days for t in trades) / len(trades)

    daily = _daily_returns(result.equity_curve)
    sharpe = _sharpe(daily)

    exit_counts: dict[str, int] = {}
    for t in trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    return PeriodMetrics(
        period_label=result.period_label,
        n_trades=len(trades),
        win_rate=win_rate,
        avg_pnl_pct=avg_pnl_pct,
        total_pnl=total_pnl,
        profit_factor=profit_factor,
        max_drawdown_pct=_max_drawdown(result.equity_curve),
        sharpe=sharpe,
        avg_hold_days=avg_hold,
        n_bars=result.n_bars,
        initial_equity=result.initial_equity,
        final_equity=result.equity_curve[-1] if result.equity_curve else result.initial_equity,
        exit_reason_counts=exit_counts,
    )
