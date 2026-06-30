"""
Tests for market_calendar — trading day detection and market status.
"""
import sys
from datetime import datetime, date, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.market_calendar import is_trading_day, get_market_status, get_open_regions, MarketWindow


# ── is_trading_day ─────────────────────────────────────────────────────────────

def test_nyse_open_tuesday():
    # 2024-01-02 is a Tuesday (and NOT a holiday for NYSE)
    assert is_trading_day("US", date(2024, 1, 2)) is True


def test_nyse_closed_saturday():
    # 2024-01-06 is a Saturday
    assert is_trading_day("US", date(2024, 1, 6)) is False


def test_nyse_closed_sunday():
    assert is_trading_day("US", date(2024, 1, 7)) is False


def test_nyse_closed_new_years():
    # NYSE is closed January 1st
    assert is_trading_day("US", date(2024, 1, 1)) is False


def test_nyse_closed_independence_day_observed():
    # July 4, 2024 is a Thursday — NYSE is closed for Independence Day
    assert is_trading_day("US", date(2024, 7, 4)) is False


def test_eu_closed_weekend():
    assert is_trading_day("EU", date(2024, 6, 1)) is False  # Saturday


def test_eu_open_weekday():
    assert is_trading_day("EU", date(2024, 6, 3)) is True   # Monday


def test_crypto_always_open():
    # Crypto is 24/7 regardless of date
    assert is_trading_day("CRYPTO", date(2024, 1, 1)) is True   # New Year
    assert is_trading_day("CRYPTO", date(2024, 7, 4)) is True   # Independence Day
    assert is_trading_day("CRYPTO", date(2024, 12, 25)) is True  # Christmas


# ── get_market_status ──────────────────────────────────────────────────────────

def test_us_market_open_during_session():
    # 13:00 UTC = 9:00 ET (before open)
    # 14:30 UTC = 10:30 ET (market open)
    dt = datetime(2024, 6, 3, 14, 30, tzinfo=timezone.utc)  # Monday, 10:30 ET
    status = get_market_status("US", dt)
    assert status.is_trading_day is True
    assert status.is_open is True
    assert status.is_near_close is False


def test_us_market_closed_before_open():
    # 12:00 UTC = 8:00 ET — before 9:30 ET open
    dt = datetime(2024, 6, 3, 12, 0, tzinfo=timezone.utc)
    status = get_market_status("US", dt)
    assert status.is_open is False


def test_us_market_closed_after_close():
    # 21:00 UTC = 17:00 ET — after 16:00 ET close
    dt = datetime(2024, 6, 3, 21, 0, tzinfo=timezone.utc)
    status = get_market_status("US", dt)
    assert status.is_open is False


def test_us_market_near_close():
    # 19:40 UTC = 15:40 ET — 20 min before 16:00 close
    dt = datetime(2024, 6, 3, 19, 40, tzinfo=timezone.utc)
    status = get_market_status("US", dt)
    assert status.is_open is True
    assert status.is_near_close is True


def test_crypto_always_open_status():
    # Any datetime returns is_open=True for CRYPTO
    dt = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    status = get_market_status("CRYPTO", dt)
    assert status.is_open is True
    assert status.is_trading_day is True
    assert status.is_near_close is False


def test_market_status_returns_market_window():
    dt = datetime(2024, 6, 3, 14, 30, tzinfo=timezone.utc)
    status = get_market_status("US", dt)
    assert isinstance(status, MarketWindow)
    assert status.region == "US"


def test_unknown_region_raises():
    with pytest.raises((ValueError, KeyError)):
        get_market_status("MARS")


# ── get_open_regions ──────────────────────────────────────────────────────────

def test_crypto_always_in_open_regions():
    # At any datetime, CRYPTO should appear in open regions
    dt = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    regions = get_open_regions(dt)
    assert "CRYPTO" in regions


def test_us_in_open_regions_during_session():
    # Monday 14:30 UTC = 10:30 ET
    dt = datetime(2024, 6, 3, 14, 30, tzinfo=timezone.utc)
    regions = get_open_regions(dt)
    assert "US" in regions


def test_us_not_in_open_regions_after_close():
    # Monday 22:00 UTC = 18:00 ET
    dt = datetime(2024, 6, 3, 22, 0, tzinfo=timezone.utc)
    regions = get_open_regions(dt)
    assert "US" not in regions
