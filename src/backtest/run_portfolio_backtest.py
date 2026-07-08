"""
CLI entry point for the PORTFOLIO backtest — the whole account simulated as
one shared equity curve, with the same concurrency/risk caps as production.

Usage (requires cached candles — fetch them first via run_backtest.py --fetch):
  python -m src.backtest.run_portfolio_backtest --symbols AAPL MSFT TSLA
  python -m src.backtest.run_portfolio_backtest --symbols $(cat universe.txt) --min-bars 60

See src/backtest/portfolio_engine.py's module docstring for what this models
that engine.py's per-symbol backtest does not (shared capital, position/risk
caps, daily loss limit, account drawdown throttle, calendar-day time exit).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.backtest import data as bt_data
from src.backtest.portfolio_engine import PortfolioConfig, run_portfolio, split_is_oos, summarize

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("backtest.portfolio")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="etoroBot PORTFOLIO backtest (shared equity, real risk caps)")
    p.add_argument("--symbols", nargs="+", required=True, help="Symbols to include in the portfolio")
    p.add_argument("--interval", default="D1", choices=["D1"], help="Only D1 is supported")
    p.add_argument("--min-bars", type=int, default=250,
                   help="Minimum cached bars required to include a symbol (default 250)")
    p.add_argument("--equity", type=float, default=800.0, help="Starting account equity")
    p.add_argument("--risk-pct", type=float, default=8.0, help="Risk %% per trade (default 8.0)")
    p.add_argument("--atr-stop-multiple", type=float, default=3.5)
    p.add_argument("--max-notional-pct", type=float, default=13.0)
    p.add_argument("--leverage", type=float, default=3.0)
    p.add_argument("--max-open-positions", type=int, default=5)
    p.add_argument("--max-positions-per-sector", type=int, default=2,
                   help="Cap concurrent positions in the same (heuristically classified) sector")
    p.add_argument("--max-portfolio-risk-pct", type=float, default=40.0)
    p.add_argument("--daily-loss-limit-pct", type=float, default=3.0)
    p.add_argument("--account-drawdown-hard-stop-pct", type=float, default=10.0)
    p.add_argument("--reduced-risk-pct", type=float, default=3.0)
    p.add_argument("--max-hold-bars", type=int, default=20, help="BARS held (trading days), matches engine.py exactly")
    p.add_argument("--split", type=float, default=0.7, help="IS/OOS split ratio by calendar time")
    p.add_argument("--verbose", action="store_true")
    return p


def _print_metrics(m) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {m.label}")
    print(f"  Trades:       {m.n_trades}")
    print(f"  Win rate:     {m.win_rate:.1f}%")
    print(f"  Total P&L:    ${m.total_pnl:+,.2f}")
    print(f"  Profit Factor:{m.profit_factor:.2f}" if m.profit_factor != float("inf") else "  Profit Factor: inf (no losses)")
    print(f"  Max Drawdown: {m.max_drawdown_pct:.2f}%  (portfolio-level, not per-symbol)")
    print(f"  Start equity: ${m.start_equity:,.2f}  ->  End equity: ${m.end_equity:,.2f}")
    print(f"  Exits:        {m.exit_reasons}")
    print(f"{'═' * 60}")


def main() -> None:
    args = _build_parser().parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.min_bars is not None:
        bt_data._MIN_BARS_FOR_BACKTEST = args.min_bars

    symbols = [s.upper() for s in args.symbols]
    data = {}
    for symbol in symbols:
        df = bt_data.load_dataframe(symbol, args.interval)
        if df is not None:
            data[symbol] = df
    print(f"Loaded {len(data)}/{len(symbols)} symbols with sufficient cached history "
          f"(>= {bt_data._MIN_BARS_FOR_BACKTEST} bars)")

    if not data:
        print("No symbols with sufficient data — nothing to backtest.")
        sys.exit(1)

    cfg = PortfolioConfig(
        initial_equity=args.equity,
        risk_per_trade_pct=args.risk_pct,
        atr_stop_multiple=args.atr_stop_multiple,
        max_notional_pct=args.max_notional_pct,
        leverage=args.leverage,
        max_open_positions=args.max_open_positions,
        max_positions_per_sector=args.max_positions_per_sector,
        max_portfolio_risk_pct=args.max_portfolio_risk_pct,
        daily_loss_limit_pct=args.daily_loss_limit_pct,
        account_drawdown_hard_stop_pct=args.account_drawdown_hard_stop_pct,
        reduced_risk_pct=args.reduced_risk_pct,
        max_hold_bars=args.max_hold_bars,
    )

    print(f"\n{'#' * 60}")
    print("  etoroBot PORTFOLIO Backtester — shared equity, real risk caps")
    print(f"  Symbols            : {len(data)}")
    print(f"  Equity              : ${cfg.initial_equity:,.0f}")
    print(f"  Risk/trade          : {cfg.risk_per_trade_pct}%")
    print(f"  ATR stop            : {cfg.atr_stop_multiple}x")
    print(f"  Leverage            : {cfg.leverage}x")
    print(f"  Max open positions  : {cfg.max_open_positions}")
    print(f"  Max portfolio risk  : {cfg.max_portfolio_risk_pct}%")
    print(f"  Daily loss limit    : {cfg.daily_loss_limit_pct}%")
    print(f"  Drawdown hard stop  : {cfg.account_drawdown_hard_stop_pct}% -> risk {cfg.reduced_risk_pct}%")
    print(f"  Max hold            : {cfg.max_hold_bars} BARS (trading days)")
    print(f"{'#' * 60}")

    result = run_portfolio(data, cfg)

    full_metrics = summarize(result, result.trades, "FULL PERIOD")
    _print_metrics(full_metrics)
    print(f"  Daily-loss blocks triggered: {result.daily_loss_blocks}")
    print(f"  Days with drawdown-throttle active: {result.drawdown_throttle_days}")

    is_metrics, oos_metrics = split_is_oos(result, args.split)
    _print_metrics(is_metrics)
    _print_metrics(oos_metrics)

    print("\nDone.")


if __name__ == "__main__":
    main()
