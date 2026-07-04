"""
CLI entry point for the backtester.

Usage (requires cached candles — see --fetch flag):
  python -m src.backtest.run_backtest                  # default universe
  python -m src.backtest.run_backtest --symbols AAPL TSLA BTC
  python -m src.backtest.run_backtest --fetch          # re-download before backtesting
  python -m src.backtest.run_backtest --walk-forward   # walk-forward mode (4 folds)
  python -m src.backtest.run_backtest --exit trailing  # compare exit modes

Fetching requires ETORO_USER and ETORO_PASS in .env (or env vars).
Running from cached CSVs does NOT require credentials.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.backtest import data as bt_data
from src.backtest import engine as bt_engine
from src.backtest import metrics as bt_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger("backtest.run")

# ── Default universe (mix of equity + crypto) ────────────────────────────────
_DEFAULT_SYMBOLS = [
    # Equities
    "AAPL", "MSFT", "TSLA", "NVDA", "AMZN",
    # Crypto (available on eToro as CFDs)
    "BTC", "ETH", "SOL",
]

_CRYPTO = {"BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "POL", "LINK"}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="etoroBot backtest runner (100%% rules, no LLM)"
    )
    p.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS,
                   help="Symbols to backtest")
    p.add_argument("--fetch", action="store_true",
                   help="Re-download candles (needs ETORO credentials)")
    p.add_argument("--force-fetch", action="store_true",
                   help="Force re-download even if cache exists")
    p.add_argument("--years", type=int, default=5,
                   help="Years of history to fetch (default 5)")
    p.add_argument("--months", type=int, default=None,
                   help="Filter data to only use the last N months for the backtest")
    p.add_argument("--split", type=float, default=0.7,
                   help="IS/OOS split ratio (default 0.7)")
    p.add_argument("--walk-forward", action="store_true",
                   help="Run walk-forward mode (4 folds)")
    p.add_argument("--exit", choices=["mean_reversion", "trailing"],
                   default="mean_reversion",
                   help="Exit mode (default: mean_reversion)")
    p.add_argument("--no-trend-filter", action="store_true",
                   help="Disable SMA200 trend filter (compare variant)")
    p.add_argument("--risk-pct", type=float, default=1.0,
                   help="Risk %% per trade of equity (default 1.0)")
    p.add_argument("--equity", type=float, default=10_000.0,
                   help="Starting equity in USD (default 10000)")
    p.add_argument("--verbose", action="store_true")
    return p


async def _maybe_fetch(symbols: list[str], args: argparse.Namespace) -> None:
    """Download candles if --fetch or --force-fetch is set."""
    if not (args.fetch or args.force_fetch):
        return

    try:
        from src.core.etoro_client import EtoroClient
    except ImportError as e:
        logger.error("Cannot import EtoroClient: %s", e)
        sys.exit(1)

    if not os.getenv("ETORO_PUBLIC_API_KEY") or not os.getenv("ETORO_USER_KEY"):
        logger.error("ETORO_PUBLIC_API_KEY and ETORO_USER_KEY must be set in .env to fetch data")
        sys.exit(1)

    async with EtoroClient() as client:
        logger.info("Fetching candles for: %s", symbols)
        await bt_data.fetch_all(
            symbols, client,
            years=args.years,
            force=args.force_fetch,
        )


def _print_symbol_report(
    symbol: str,
    is_m: bt_metrics.PeriodMetrics,
    oos_m: bt_metrics.PeriodMetrics,
) -> None:
    sep = "─" * 56
    print(f"\n{'═' * 56}")
    print(f"  {symbol}")
    print(sep)
    print(is_m)
    print(sep)
    print(oos_m)


def _print_aggregate(
    label: str,
    all_metrics: list[bt_metrics.PeriodMetrics],
) -> None:
    if not all_metrics:
        return
    n_trades = sum(m.n_trades for m in all_metrics)
    if n_trades == 0:
        print(f"\n[{label} AGGREGATE] — 0 trades across all symbols")
        return

    all_trades_pnl = []
    total_pnl = 0.0
    wins = 0
    gross_win = 0.0
    gross_loss = 0.0
    exit_totals: dict[str, int] = {}

    for m in all_metrics:
        total_pnl += m.total_pnl
        wins += int(m.win_rate * m.n_trades + 0.5)
        gross_win += m.profit_factor * abs(sum(
            t for t in [] if t < 0  # placeholder — already in PeriodMetrics
        ))
        for reason, cnt in m.exit_reason_counts.items():
            exit_totals[reason] = exit_totals.get(reason, 0) + cnt

    win_rate_agg = wins / n_trades if n_trades else 0
    avg_pnl = sum(m.avg_pnl_pct * m.n_trades for m in all_metrics) / n_trades

    pf_list = [m.profit_factor for m in all_metrics if m.n_trades > 0 and not (m.profit_factor == float("inf"))]
    pf_avg = sum(pf_list) / len(pf_list) if pf_list else 0.0

    print(f"\n{'═' * 56}")
    print(f"  AGGREGATE — {label} ({len(all_metrics)} symbols, {n_trades} trades)")
    print(f"  Win rate:   {win_rate_agg * 100:.1f}%")
    print(f"  Avg P&L:    {avg_pnl:+.2f}% / trade")
    print(f"  Total P&L:  ${total_pnl:+,.2f}")
    print(f"  Avg PF:     {pf_avg:.2f}")
    print(f"  Exits:      {exit_totals}")
    print(f"{'═' * 56}")


def main() -> None:
    args = _build_parser().parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    symbols = [s.upper() for s in args.symbols]

    asyncio.run(_maybe_fetch(symbols, args))

    cfg = bt_engine.BacktestConfig(
        initial_equity=args.equity,
        risk_per_trade_pct=args.risk_pct,
        use_trend_filter=not args.no_trend_filter,
        exit_mode=args.exit,
    )

    print(f"\n{'#' * 56}")
    print(f"  etoroBot Backtester")
    print(f"  Exit mode  : {cfg.exit_mode}")
    print(f"  Trend filt : {'ON (SMA200)' if cfg.use_trend_filter else 'OFF'}")
    print(f"  Risk/trade : {cfg.risk_per_trade_pct}%")
    print(f"  Equity     : ${cfg.initial_equity:,.0f}")
    print(f"  IS split   : {args.split:.0%} / {1-args.split:.0%}")
    if args.months:
        print(f"  Months     : Last {args.months} months")
    print(f"  Symbols    : {' '.join(symbols)}")
    print(f"{'#' * 56}")

    is_metrics_all: list[bt_metrics.PeriodMetrics] = []
    oos_metrics_all: list[bt_metrics.PeriodMetrics] = []

    for symbol in symbols:
        asset_class = "crypto" if symbol in _CRYPTO else "equity"
        df = bt_data.load_dataframe(symbol)
        if df is None:
            print(f"\n[{symbol}] SKIP — no cached data (run with --fetch)")
            continue

        if args.months:
            import pandas as pd
            cutoff = df.index[-1] - pd.DateOffset(months=args.months)
            df = df[df.index >= cutoff]
            if len(df) < 20: # arbitrary minimum
                print(f"\n[{symbol}] SKIP — too few bars after --months filter")
                continue

        if args.walk_forward:
            folds = bt_engine.walk_forward(df, cfg, symbol, asset_class=asset_class)
            if not folds:
                print(f"\n[{symbol}] SKIP — too few bars for walk-forward")
                continue
            print(f"\n{'═' * 56}")
            print(f"  {symbol} — Walk-Forward ({len(folds)} folds)")
            for is_r, oos_r in folds:
                is_m = bt_metrics.compute(is_r)
                oos_m = bt_metrics.compute(oos_r)
                print(f"\n  Fold: {oos_r.period_label}")
                print(f"  IS  : {is_r.n_bars} bars, {is_m.n_trades} trades, "
                      f"PF={is_m.profit_factor:.2f}, WR={is_m.win_rate*100:.0f}%")
                print(f"  OOS : {oos_r.n_bars} bars, {oos_m.n_trades} trades, "
                      f"PF={oos_m.profit_factor:.2f}, WR={oos_m.win_rate*100:.0f}%")
                oos_metrics_all.append(oos_m)
        else:
            is_r, oos_r = bt_engine.split_run(df, cfg, symbol, args.split, asset_class)
            is_m = bt_metrics.compute(is_r)
            oos_m = bt_metrics.compute(oos_r)
            _print_symbol_report(symbol, is_m, oos_m)
            is_metrics_all.append(is_m)
            oos_metrics_all.append(oos_m)

    if not args.walk_forward:
        _print_aggregate("IN-SAMPLE", is_metrics_all)
        _print_aggregate("OUT-OF-SAMPLE", oos_metrics_all)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
