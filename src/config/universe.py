"""
Trading universe — static fallback lists by region.

IMPORTANT: These are best-guess tickers. eToro's internal instrument names
may differ (e.g. "MC" vs "LVMH", "VOW3" vs "VWAGY").
Run `python src/config/discovery.py` to build the authoritative cache.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_FILE = Path(__file__).parent.parent.parent / "universe_cache.json"
_CACHE_TTL_DAYS = 7

# ── Static fallback lists ─────────────────────────────────────────────────────

US_STOCKS: list[str] = [
    # Big Tech / FAANG+
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
    # Financials
    "JPM", "BAC", "GS", "V", "MA", "AXP", "MS",
    # Healthcare
    "JNJ", "UNH", "LLY", "ABBV", "PFE", "MRK", "AMGN", "TMO", "DHR",
    # Consumer
    "WMT", "COST", "HD", "NKE", "SBUX", "MCD", "PM", "PG",
    # Energy
    "XOM", "CVX", "COP",
    # Industrial
    "GE", "HON", "DE", "CAT", "BA",
    # Semiconductors
    "AMD", "INTC", "QCOM", "AMAT", "ADI", "TXN",
    # Software / Cloud / AI
    "CRM", "NOW", "PANW", "ISRG", "BKNG", "NFLX", "DIS",
    # Diversified
    "BRK.B", "ACN", "SPGI", "BLK", "ABT",
]

EU_STOCKS: list[str] = [
    # Germany
    "SAP", "SIE", "ALV", "BAYN", "BMW", "ADS", "DB1",
    # France
    "MC", "OR", "TTE", "BNP", "AIR",
    # Netherlands
    "ASML", "PHIA", "INGA",
    # UK (often tradeable on eToro without ".L" suffix)
    "SHEL", "BP", "HSBA", "AZN", "GSK", "ULVR", "RIO",
    # Italy
    "ENEL", "ENI",
    # Spain
    "IBE", "ITX",
]

ASIA_STOCKS: list[str] = [
    # Taiwan ADR
    "TSM",
    # China ADRs
    "BABA", "JD", "BIDU", "NIO", "XPEV", "LI",
    # Japan ADRs
    "TM", "SONY", "HMC", "MUFG",
    # South Korea
    "KB",
    # SE Asia
    "SE",
    # India ADRs
    "INFY", "WIT", "HDB",
]

CRYPTO: list[str] = [
    "BTC", "ETH", "BNB", "XRP", "SOL", "DOGE", "ADA", "AVAX",
    "DOT", "LINK", "LTC", "UNI", "ATOM", "XLM", "NEAR",
    "BCH", "APT", "ARB", "OP", "MATIC",
]

REGION_SYMBOLS: dict[str, list[str]] = {
    "US": US_STOCKS,
    "EU": EU_STOCKS,
    "ASIA": ASIA_STOCKS,
    "CRYPTO": CRYPTO,
}

# ── Cache helpers ─────────────────────────────────────────────────────────────


def load_cache() -> dict[str, Any] | None:
    """Load universe cache if it exists and is fresh."""
    if not _CACHE_FILE.exists():
        return None
    try:
        data = json.loads(_CACHE_FILE.read_text())
        saved_at = datetime.fromisoformat(data["saved_at"])
        age_days = (datetime.now(timezone.utc) - saved_at).days
        if age_days > _CACHE_TTL_DAYS:
            logger.info("Universe cache is %d days old — consider re-running discovery.py", age_days)
        return data
    except Exception as exc:
        logger.warning("Could not read universe cache: %s", exc)
        return None


def get_symbols(region: str, use_cache: bool = True) -> list[str]:
    """
    Get tradeable symbols for a region.
    Uses discovery cache when available, falls back to static list.
    """
    if use_cache:
        cache = load_cache()
        if cache and region in cache.get("regions", {}):
            symbols = cache["regions"][region]
            logger.debug("Universe: loaded %d %s symbols from cache", len(symbols), region)
            return symbols

    symbols = REGION_SYMBOLS.get(region, [])
    logger.debug("Universe: using %d static %s symbols", len(symbols), region)
    return symbols


def get_all_symbols(use_cache: bool = True) -> list[str]:
    """All symbols across all regions (deduplicated)."""
    seen: set[str] = set()
    result: list[str] = []
    for region in REGION_SYMBOLS:
        for sym in get_symbols(region, use_cache=use_cache):
            if sym not in seen:
                seen.add(sym)
                result.append(sym)
    return result


def get_instrument_id(symbol: str, use_cache: bool = True) -> str | None:
    """Look up the eToro instrument ID from cache."""
    if not use_cache:
        return None
    cache = load_cache()
    if not cache:
        return None
    return cache.get("instrument_ids", {}).get(symbol)
