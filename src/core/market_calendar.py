"""
Market calendar — trading windows per region with real holiday support.
Uses pandas_market_calendars when available; falls back to weekday-only check.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pandas_market_calendars as mcal
    import pandas as pd
    _HAS_MCAL = True
except ImportError:
    _HAS_MCAL = False
    # FAIL-CLOSED: when the library is absent we do NOT assume any equity market is open.
    # This prevents trading on holidays.  The bot will skip equity regions until the
    # library is installed.  Crypto (24/7) is unaffected.
    logger.critical(
        "pandas_market_calendars is not installed. "
        "Equity market open/close detection is DISABLED. "
        "Install it with: pip install pandas-market-calendars"
    )

# ── Region config ─────────────────────────────────────────────────────────────
# calendar: pandas_market_calendars name, None = 24/7
# fallbacks: tried in order if primary calendar is not found in mcal

_REGION_CONFIGS: dict[str, dict] = {
    "US": {
        "calendar": "NYSE",
        "fallback": ["NASDAQ"],
        "tz": "America/New_York",
        "open_h": 9, "open_m": 30,
        "close_h": 16, "close_m": 0,
    },
    "EU": {
        "calendar": "XFRA",
        "fallback": ["Euronext", "LSE"],
        "tz": "Europe/Berlin",
        "open_h": 9, "open_m": 0,
        "close_h": 17, "close_m": 30,
    },
    "ASIA": {
        # "ASIA" is a symbol-list label, not a trading venue: every symbol in
        # ASIA_STOCKS (BABA, JD, BIDU, NIO, TSM, TM, SONY, HMC, MUFG, KB, SE,
        # INFY, WIT, HDB...) is a US-listed ADR trading on NYSE/NASDAQ hours,
        # confirmed via real hourly volume (peaks 13:00-20:00 UTC = US session,
        # near-zero around 00:00 UTC = Tokyo open). Use the US calendar/hours.
        "calendar": "NYSE",
        "fallback": ["NASDAQ"],
        "tz": "America/New_York",
        "open_h": 9, "open_m": 30,
        "close_h": 16, "close_m": 0,
    },
    "CRYPTO": {
        "calendar": None,
        "fallback": [],
        "tz": "UTC",
        "open_h": 0, "open_m": 0,
        "close_h": 23, "close_m": 59,
    },
}

_NEAR_CLOSE_MINUTES = 30  # warn when within N minutes of close


@dataclass
class MarketWindow:
    region: str
    is_open: bool
    is_trading_day: bool
    is_near_close: bool      # within _NEAR_CLOSE_MINUTES of close
    open_utc: Optional[datetime]   # today's open in UTC
    close_utc: Optional[datetime]  # today's close in UTC


# ── Calendar resolution ───────────────────────────────────────────────────────

def _get_mcal(region: str):
    """Return a pandas_market_calendars calendar object for the region."""
    if not _HAS_MCAL:
        return None
    config = _REGION_CONFIGS[region]
    names = [config["calendar"]] + config["fallback"]
    for name in names:
        if name is None:
            continue
        try:
            cal = mcal.get_calendar(name)
            return cal
        except Exception:
            continue
    logger.warning("No mcal calendar found for region %s — using weekday fallback", region)
    return None


def is_trading_day(region: str, d: date | None = None) -> bool:
    """Return True if `d` (UTC date, default today) is a trading day for the region."""
    if d is None:
        d = datetime.now(timezone.utc).date()
    if region == "CRYPTO":
        return True

    cal = _get_mcal(region)
    if cal is not None:
        try:
            schedule = cal.schedule(
                start_date=d.isoformat(),
                end_date=d.isoformat(),
            )
            return not schedule.empty
        except Exception as exc:
            logger.debug("mcal.schedule failed for %s on %s: %s", region, d, exc)
            # If the calendar query itself fails, fall back to weekday-only
            return d.weekday() < 5

    # Library not installed: fail-closed for equity regions.
    # Return False so the bot skips equity cycles rather than trading on holidays.
    logger.warning(
        "is_trading_day(%s, %s): pandas_market_calendars unavailable — assuming CLOSED",
        region, d,
    )
    return False


def get_market_status(region: str, dt: datetime | None = None) -> MarketWindow:
    """
    Return current market status for a region.
    `dt` defaults to now (UTC-aware).
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    config = _REGION_CONFIGS[region]
    tz = ZoneInfo(config["tz"])
    local_dt = dt.astimezone(tz)
    today = local_dt.date()

    if region == "CRYPTO":
        return MarketWindow(
            region="CRYPTO",
            is_open=True,
            is_trading_day=True,
            is_near_close=False,
            open_utc=None,
            close_utc=None,
        )

    is_day = is_trading_day(region, today)

    open_local = local_dt.replace(
        hour=config["open_h"], minute=config["open_m"], second=0, microsecond=0
    )
    close_local = local_dt.replace(
        hour=config["close_h"], minute=config["close_m"], second=0, microsecond=0
    )
    open_utc = open_local.astimezone(ZoneInfo("UTC"))
    close_utc = close_local.astimezone(ZoneInfo("UTC"))

    is_open = is_day and open_local <= local_dt < close_local
    is_near_close = (
        is_open
        and (close_local - local_dt).total_seconds() <= _NEAR_CLOSE_MINUTES * 60
    )

    return MarketWindow(
        region=region,
        is_open=is_open,
        is_trading_day=is_day,
        is_near_close=is_near_close,
        open_utc=open_utc if is_day else None,
        close_utc=close_utc if is_day else None,
    )


def get_open_regions(dt: datetime | None = None) -> list[str]:
    """Return list of regions whose market is currently open."""
    return [
        region
        for region in _REGION_CONFIGS
        if get_market_status(region, dt).is_open
    ]


def market_open_cron(region: str) -> dict:
    """
    Return APScheduler CronTrigger kwargs to fire at market open for a region.
    """
    config = _REGION_CONFIGS[region]
    if region == "CRYPTO":
        # Fire every 6 hours
        return {"hour": "0,6,12,18", "minute": "0"}
    return {
        "hour": config["open_h"],
        "minute": config["open_m"] + 5,  # 5 min after open
        "timezone": config["tz"],
    }
