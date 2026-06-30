"""
Backtest data layer — candle download via EtoroClient + disk cache.

Workflow:
  1. Call fetch_symbol() (async, needs EtoroClient).
  2. Candles are cached as CSV in data/candles/<SYMBOL>.csv.
  3. Subsequent calls load from cache (no network).

The backtest engine then calls load_dataframe() (sync, offline).
"""
from __future__ import annotations

import asyncio
import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.core.etoro_client import EtoroClient

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "candles"
_TRADING_DAYS_PER_YEAR = 252
_MIN_BARS_FOR_BACKTEST = 250   # need warmup + at least some IS data


def _cache_path(symbol: str) -> Path:
    return _CACHE_DIR / f"{symbol.upper()}.csv"


def _normalise_candle(c: dict) -> dict | None:
    """Normalise any candle dict variant into {date, open, high, low, close, volume}."""
    try:
        # Date field: try several keys
        date = (
            c.get("date") or c.get("timestamp") or
            c.get("openTime") or c.get("time") or ""
        )
        o = float(c.get("open",   c.get("o", 0)))
        h = float(c.get("high",   c.get("h", 0)))
        lo = float(c.get("low",   c.get("l", 0)))
        cl = float(c.get("close", c.get("c", 0)))
        vol = float(c.get("volume", c.get("v", 0)))
        if cl <= 0:
            return None
        return {"date": str(date), "open": o, "high": h, "low": lo, "close": cl, "volume": vol}
    except (TypeError, ValueError):
        return None


def load_cached_candles(symbol: str) -> list[dict] | None:
    """Load candles from CSV cache. Returns None if no cache file exists."""
    path = _cache_path(symbol)
    if not path.exists():
        return None
    try:
        rows = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append({
                    "date":   row["date"],
                    "open":   float(row["open"]),
                    "high":   float(row["high"]),
                    "low":    float(row["low"]),
                    "close":  float(row["close"]),
                    "volume": float(row["volume"]),
                })
        return rows or None
    except Exception as exc:
        logger.warning("Cache load failed for %s: %s", symbol, exc)
        return None


def save_candles(symbol: str, candles: list[dict]) -> None:
    """Save normalised candle list to CSV cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol)
    try:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["date", "open", "high", "low", "close", "volume"]
            )
            writer.writeheader()
            writer.writerows(candles)
        logger.info("Cached %d candles for %s → %s", len(candles), symbol, path.name)
    except Exception as exc:
        logger.warning("Cache save failed for %s: %s", symbol, exc)


async def fetch_symbol(
    symbol: str,
    client: "EtoroClient",
    years: int = 5,
    force: bool = False,
) -> list[dict]:
    """
    Download D1 candles for symbol via EtoroClient; cache to disk.
    Returns the cached list if fresh (unless force=True).
    Returns [] if the download fails (caller should skip the symbol).
    """
    if not force:
        cached = load_cached_candles(symbol)
        if cached:
            logger.info(
                "Using cached %d bars for %s", len(cached), symbol
            )
            return cached

    count = years * _TRADING_DAYS_PER_YEAR + 60   # +60 warm-up buffer
    try:
        raw = await client.get_candles(symbol, interval="D1", count=count)
        candles = [_normalise_candle(c) for c in (raw or [])]
        candles = [c for c in candles if c is not None]
        if not candles:
            logger.warning("%s: API returned 0 usable candles", symbol)
            return []
        save_candles(symbol, candles)
        return candles
    except Exception as exc:
        logger.warning("Download failed for %s: %s — skipping", symbol, exc)
        return []


async def fetch_all(
    symbols: list[str],
    client: "EtoroClient",
    years: int = 5,
    force: bool = False,
    inter_request_delay: float = 0.5,
) -> dict[str, list[dict]]:
    """Fetch candles for a list of symbols, returning {symbol: candles}."""
    results: dict[str, list[dict]] = {}
    for symbol in symbols:
        results[symbol] = await fetch_symbol(symbol, client, years, force)
        await asyncio.sleep(inter_request_delay)
    return results


def load_dataframe(symbol: str) -> pd.DataFrame | None:
    """
    Load cached candles into a pandas DataFrame.

    Returns None if no cache exists or fewer than _MIN_BARS_FOR_BACKTEST rows.

    Index: DatetimeIndex (date column parsed as datetime).
    Columns: open, high, low, close, volume.
    """
    candles = load_cached_candles(symbol)
    if not candles:
        logger.warning("%s: no cache — run fetch first", symbol)
        return None

    df = pd.DataFrame(candles)

    # Parse date column into index
    if "date" in df.columns and df["date"].iloc[0]:
        try:
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date").sort_index()
        except Exception:
            # Fallback: use integer index
            df = df.drop(columns=["date"], errors="ignore")
    else:
        df = df.drop(columns=["date"], errors="ignore")

    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    n = len(df)
    if n < _MIN_BARS_FOR_BACKTEST:
        logger.warning(
            "%s: only %d bars (need >=%d) — skipping", symbol, n, _MIN_BARS_FOR_BACKTEST
        )
        return None

    logger.info("%s: loaded %d bars", symbol, n)
    return df
