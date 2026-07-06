"""
Backtest data layer — candle download via EtoroClient + disk cache.

Workflow:
  1. Call fetch_symbol() (async, needs EtoroClient).
  2. Candles are cached as CSV in data/candles/<SYMBOL>.csv.
  3. Subsequent calls load from cache (no network).

The backtest engine then calls load_dataframe() (sync, offline).

─────────────────────────────────────────────────────────────────
NOTE — eToro API field names (manual verification required):
  The exact JSON field names returned by
  GET /market-data/instruments/history/candles
  are not confirmed from the public portal at writing time.
  _ETORO_FIELD_MAP below lists the candidates tried in order.
  After running --fetch in demo mode, inspect the raw response
  (set LOG_LEVEL=DEBUG or add a temporary print()) and update
  _ETORO_FIELD_MAP to put the real names first.

NOTE — Pagination (manual verification required):
  The eToro API documents `limit` (max 1000) but not a date-range
  or cursor parameter.  fetch_symbol() requests the maximum 1000
  bars per call.  For 5 years of daily data (~1260 trading days),
  _PAGE_LIMIT=1000 covers ~4 trading years.  If the API supports
  a `from` / `startDate` / `cursor` param, update _fetch_page()
  and set _SUPPORTS_PAGINATION = True.

NOTE — Split/dividend adjustment:
  Confirm in demo whether the returned close prices are adjusted
  for corporate actions (splits, dividends).  Multi-year equity
  backtests on unadjusted prices produce false return signals
  (e.g., a 2-for-1 split looks like a −50% crash).
  eToro typically delivers adjusted prices for most markets, but
  verify for each region before trusting long-horizon signals.
─────────────────────────────────────────────────────────────────
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
_MIN_BARS_FOR_BACKTEST = 250
_PAGE_LIMIT = 1000          # eToro API max per request — confirmed NOT paginable:
                            # asc and desc requests both return the identical most
                            # recent 1000-bar window, just reordered. This hard-caps
                            # how much history is obtainable for intraday intervals.
_MAX_GAP_DAYS = 10          # log warning if gap between bars exceeds this (D1 only)

# Bars/calendar-day for each interval — eToro quotes stock CFDs continuously
# (24h/day, confirmed via live H4/H1 fetch — NOT limited to exchange hours).
_BARS_PER_DAY = {"D1": 1, "H4": 6, "H1": 24}

# ── eToro response field mapping ─────────────────────────────────────────────
# Candidates tried in order (first match wins).
# NOTE: update after verifying actual field names in demo mode.
_ETORO_FIELD_MAP = {
    # date / timestamp variants
    "date":   ["date", "time", "timestamp", "fromDate", "openTime", "barTime", "Date"],
    # price variants
    "open":   ["open", "openPrice", "Open"],
    "high":   ["high", "highPrice", "High"],
    "low":    ["low",  "lowPrice",  "Low"],
    "close":  ["close", "closePrice", "Close", "last"],
    # volume variants
    "volume": ["volume", "volumeValue", "totalVolume", "Volume", "vol"],
}


def _cache_path(symbol: str, interval: str = "D1") -> Path:
    suffix = "" if interval == "D1" else f"_{interval}"
    return _CACHE_DIR / f"{symbol.upper()}{suffix}.csv"


def _pick(d: dict, candidates: list[str]):
    """Return the first value found in d whose key matches any candidate."""
    for k in candidates:
        if k in d:
            return d[k]
    return None


def _normalise_candle(c: dict) -> dict | None:
    """
    Normalise any eToro candle dict into {date, open, high, low, close, volume}.

    Tries multiple field-name variants per _ETORO_FIELD_MAP.
    Returns None if close <= 0 or mandatory fields are missing.
    """
    try:
        date = _pick(c, _ETORO_FIELD_MAP["date"]) or ""
        o    = _pick(c, _ETORO_FIELD_MAP["open"])
        h    = _pick(c, _ETORO_FIELD_MAP["high"])
        lo   = _pick(c, _ETORO_FIELD_MAP["low"])
        cl   = _pick(c, _ETORO_FIELD_MAP["close"])
        vol  = _pick(c, _ETORO_FIELD_MAP["volume"])

        if cl is None:
            return None
        cl_f = float(cl)
        if cl_f <= 0:
            return None

        return {
            "date":   str(date),
            "open":   float(o)   if o   is not None else cl_f,
            "high":   float(h)   if h   is not None else cl_f,
            "low":    float(lo)  if lo  is not None else cl_f,
            "close":  cl_f,
            "volume": float(vol) if vol is not None else 0.0,
        }
    except (TypeError, ValueError) as exc:
        logger.debug("_normalise_candle: skipping malformed bar: %s", exc)
        return None


def load_cached_candles(symbol: str, interval: str = "D1") -> list[dict] | None:
    """Load candles from CSV cache. Returns None if no cache file exists."""
    path = _cache_path(symbol, interval)
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


def save_candles(symbol: str, candles: list[dict], interval: str = "D1") -> None:
    """Save normalised candle list to CSV cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol, interval)
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


async def _fetch_page(
    symbol: str,
    client: "EtoroClient",
    count: int,
    interval: str = "D1",
    direction: str = "asc",
) -> list[dict]:
    """Fetch a single page of candles (up to _PAGE_LIMIT) from EtoroClient."""
    try:
        raw = await client.get_candles(
            symbol, interval=interval, count=min(count, _PAGE_LIMIT), direction=direction
        )
        return raw or []
    except Exception as exc:
        logger.warning("Page fetch failed for %s: %s", symbol, exc)
        return []


async def fetch_symbol(
    symbol: str,
    client: "EtoroClient",
    years: int = 5,
    force: bool = False,
    interval: str = "D1",
) -> list[dict]:
    """
    Download candles for `symbol` via EtoroClient; cache to disk (per-interval file).

    Returns the cached list if one exists (unless force=True).
    Returns [] on download failure (caller should skip the symbol).

    CONFIRMED (live-tested): eToro caps every candles request at 1000 bars
    and does NOT support pagination — asc and desc requests for the same
    symbol/interval return the identical most recent 1000-bar window, just
    reordered. For D1 that's ~4 years; for H4 ~8 months; for H1 ~6 weeks.
    Requesting more `years` than that ceiling silently returns less data —
    a warning is logged when fewer bars than _MIN_BARS_FOR_BACKTEST arrive.
    """
    if not force:
        cached = load_cached_candles(symbol, interval)
        if cached:
            logger.info("Using cached %d %s bars for %s", len(cached), interval, symbol)
            return cached

    bars_per_day = _BARS_PER_DAY.get(interval, 1)
    target = int(years * 365 * bars_per_day) + 60   # +60 warmup buffer

    raw = await _fetch_page(symbol, client, count=min(target, _PAGE_LIMIT), interval=interval)

    candles = [_normalise_candle(c) for c in raw]
    candles = [c for c in candles if c is not None]

    if not candles:
        logger.warning("%s: API returned 0 usable candles", symbol)
        return []

    candles = _dedup_and_sort(candles)

    if len(candles) < _MIN_BARS_FOR_BACKTEST:
        logger.warning(
            "%s (%s): only %d bars received (need >=%d) — eToro's 1000-bar-per-"
            "request cap with no pagination limits how far back %s data goes.",
            symbol, interval, len(candles), _MIN_BARS_FOR_BACKTEST, interval,
        )

    save_candles(symbol, candles, interval)
    return candles


def _dedup_and_sort(candles: list[dict]) -> list[dict]:
    """
    Deduplicate by date string and sort chronologically (ascending).

    Keeps the last occurrence when dates clash (assumes the API returns
    newer data later in the list, which is true for direction=asc).
    """
    seen: dict[str, dict] = {}
    for c in candles:
        seen[c["date"]] = c          # later entry wins
    return sorted(seen.values(), key=lambda c: c["date"])


async def fetch_all(
    symbols: list[str],
    client: "EtoroClient",
    years: int = 5,
    force: bool = False,
    inter_request_delay: float = 0.5,
    interval: str = "D1",
) -> dict[str, list[dict]]:
    """Fetch candles for a list of symbols. Returns {symbol: candles}."""
    results: dict[str, list[dict]] = {}
    for symbol in symbols:
        results[symbol] = await fetch_symbol(symbol, client, years, force, interval)
        await asyncio.sleep(inter_request_delay)
    return results


def _detect_gaps(df: pd.DataFrame, symbol: str) -> None:
    """
    Log a warning for any gap between consecutive bars larger than _MAX_GAP_DAYS.

    Large gaps can indicate: missing data, trading halts, corporate actions, or
    a feed error (e.g., split not accounted for in the raw price series).
    If many gaps are found, the backtest signals may be unreliable — verify the
    raw data in demo before running IS/OOS analysis.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return
    diffs = df.index.to_series().diff().dt.days.dropna()
    large = diffs[diffs > _MAX_GAP_DAYS]
    if large.empty:
        return
    logger.warning(
        "%s: %d calendar gap(s) > %d days detected — may indicate missing data "
        "or unadjusted corporate action. Dates: %s",
        symbol, len(large), _MAX_GAP_DAYS,
        list(large.index.strftime("%Y-%m-%d")),
    )


def load_dataframe(symbol: str, interval: str = "D1") -> pd.DataFrame | None:
    """
    Load cached candles into a pandas DataFrame.

    Returns None if no cache exists or fewer than _MIN_BARS_FOR_BACKTEST rows.

    Index: DatetimeIndex (UTC, ascending).
    Columns: open, high, low, close, volume (all float).

    Post-processing:
      - Deduplicates by date (keeps last).
      - Sorts chronologically.
      - Detects and logs large calendar gaps (potential data issues, D1 only).
    """
    candles = load_cached_candles(symbol, interval)
    if not candles:
        logger.warning("%s: no %s cache — run fetch first", symbol, interval)
        return None

    df = pd.DataFrame(candles)

    # Parse date → DatetimeIndex
    if "date" in df.columns and not df["date"].isna().all():
        try:
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
            df = df.dropna(subset=["date"])
            # Dedup by date before setting as index
            df = df.drop_duplicates(subset=["date"], keep="last")
            df = df.set_index("date").sort_index()
        except Exception as exc:
            logger.warning("%s: could not parse date column: %s", symbol, exc)
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

    if interval == "D1":
        _detect_gaps(df, symbol)

    logger.info("%s: loaded %d bars", symbol, n)
    return df
