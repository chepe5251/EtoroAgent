"""
CLI entry point for the Market Structure (BOS/ChoCh) backtest.

Usage:
  python -m src.backtest.run_market_structure --symbols AAPL MSFT TSLA
  python -m src.backtest.run_market_structure --fetch --years 5
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
from src.backtest import metrics as bt_metrics
from src.backtest.engine import BacktestConfig
from src.backtest.market_structure import MSConfig, run_market_structure

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("backtest.ms")

_DEFAULT_SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN"]
_CRYPTO = {"BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "POL", "LINK"}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Market Structure (BOS/ChoCh) backtest")
    p.add_argument("--symbols", nargs="+", default=_DEFAULT_SYMBOLS)
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--force-fetch", action="store_true")
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--risk-pct", type=float, default=1.0)
    p.add_argument("--swing-k", type=int, default=2)
    p.add_argument("--max-hold-days", type=int, default=20)
    return p


async def _maybe_fetch(symbols: list[str], args: argparse.Namespace) -> None:
    if not (args.fetch or args.force_fetch):
        return
    from src.core.etoro_client import EtoroClient
    if not os.getenv("ETORO_PUBLIC_API_KEY") or not os.getenv("ETORO_USER_KEY"):
        logger.error("ETORO_PUBLIC_API_KEY and ETORO_USER_KEY must be set in .env to fetch data")
        sys.exit(1)
    async with EtoroClient() as client:
        await bt_data.fetch_all(symbols, client, years=args.years, force=args.force_fetch)


def main() -> None:
    args = _build_parser().parse_args()
    symbols = [s.upper() for s in args.symbols]
    asyncio.run(_maybe_fetch(symbols, args))

    ms_cfg = MSConfig(
        initial_equity=args.equity,
        leverage=args.leverage,
        risk_per_trade_pct=args.risk_pct,
        swing_k=args.swing_k,
        max_hold_days=args.max_hold_days,
    )
    cost_cfg = BacktestConfig(leverage=args.leverage)

    print(f"\n{'#' * 56}")
    print("  Market Structure (BOS/ChoCh) — long-only backtest")
    print(f"  Swing k    : {ms_cfg.swing_k} bars each side")
    print(f"  Max hold   : {ms_cfg.max_hold_days}d (backstop)")
    print(f"  Equity     : ${ms_cfg.initial_equity:,.0f}  Leverage: {ms_cfg.leverage}x")
    print(f"{'#' * 56}")

    all_metrics = []
    all_trades = []
    for symbol in symbols:
        asset_class = "crypto" if symbol in _CRYPTO else "equity"
        df = bt_data.load_dataframe(symbol)
        if df is None:
            print(f"\n[{symbol}] SKIP — no cached data (run with --fetch)")
            continue
        result = run_market_structure(df, ms_cfg, cost_cfg, symbol, asset_class)
        m = bt_metrics.compute(result)
        print(f"\n{'═' * 56}\n  {symbol}\n{'─' * 56}")
        print(m)
        all_metrics.append(m)
        all_trades.extend(result.trades)

    if all_trades:
        n_trades = len(all_trades)
        total_pnl = sum(t.pnl for t in all_trades)
        wins = [t for t in all_trades if t.pnl > 0]
        losses = [t for t in all_trades if t.pnl <= 0]
        win_rate = len(wins) / n_trades * 100
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        avg_hold = sum(t.holding_days for t in all_trades) / n_trades
        exit_totals: dict[str, int] = {}
        for t in all_trades:
            exit_totals[t.exit_reason] = exit_totals.get(t.exit_reason, 0) + 1
        print(f"\n{'═' * 56}")
        print(f"  AGGREGATE ({len(all_metrics)} symbols, {n_trades} trades)")
        print(f"  Win rate:   {win_rate:.1f}%")
        print(f"  Profit F:   {pf:.2f}")
        print(f"  Avg hold:   {avg_hold:.1f} days")
        print(f"  Total P&L:  ${total_pnl:+,.2f}")
        print(f"  Exits:      {exit_totals}")
        print(f"{'═' * 56}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
