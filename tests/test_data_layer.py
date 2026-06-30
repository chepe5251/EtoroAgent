"""
Tests for the backtest data layer (Fase 0d).

Rules:
- ZERO network calls. EtoroClient.get_candles is mocked.
- Tests verify: correct endpoint params, field-name parsing, pagination
  dedup/concat, symbol→instrumentId resolution, and gap detection.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.data import (
    _ETORO_FIELD_MAP,
    _MIN_BARS_FOR_BACKTEST,
    _dedup_and_sort,
    _normalise_candle,
    fetch_symbol,
    load_dataframe,
    save_candles,
    _detect_gaps,
)


# ─────────────────────────────────────────────────────────────────────────────
# _normalise_candle — field-name variants
# ─────────────────────────────────────────────────────────────────────────────

def test_normalise_candle_verbose_keys():
    """Standard verbose field names → correct normalisation."""
    raw = {
        "date": "2024-01-15", "open": 150.0, "high": 155.0,
        "low": 149.0, "close": 153.0, "volume": 1_000_000,
    }
    c = _normalise_candle(raw)
    assert c is not None
    assert c["close"] == 153.0
    assert c["high"] == 155.0
    assert c["date"] == "2024-01-15"


def test_normalise_candle_etoro_price_variants():
    """eToro may return openPrice / highPrice / closePrice / lowPrice variants."""
    raw = {
        "time": "2024-01-15T00:00:00Z",
        "openPrice": 100.0, "highPrice": 105.0,
        "lowPrice": 99.0, "closePrice": 103.0, "volumeValue": 500_000,
    }
    c = _normalise_candle(raw)
    assert c is not None
    assert c["close"] == 103.0
    assert c["volume"] == 500_000
    assert "2024-01-15" in c["date"]


def test_normalise_candle_timestamp_key():
    """Timestamp key used as date."""
    raw = {
        "timestamp": "2024-03-01",
        "open": 200.0, "high": 210.0, "low": 195.0,
        "close": 205.0, "volume": 0,
    }
    c = _normalise_candle(raw)
    assert c is not None
    assert c["date"] == "2024-03-01"


def test_normalise_candle_rejects_zero_close():
    """Zero or negative close must be rejected (returns None)."""
    raw = {"date": "2024-01-01", "open": 0, "high": 0, "low": 0, "close": 0, "volume": 0}
    assert _normalise_candle(raw) is None


def test_normalise_candle_missing_close_returns_none():
    """Bar without any recognised close key → None."""
    raw = {"date": "2024-01-01", "open": 100.0, "high": 101.0, "low": 99.0, "volume": 1}
    assert _normalise_candle(raw) is None


def test_normalise_candle_missing_optional_fields_defaults_to_close():
    """When open/high/low/volume are absent, they default to close."""
    raw = {"date": "2024-01-01", "close": 100.0}
    c = _normalise_candle(raw)
    assert c is not None
    assert c["open"] == 100.0
    assert c["high"] == 100.0
    assert c["low"] == 100.0
    assert c["volume"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# _dedup_and_sort
# ─────────────────────────────────────────────────────────────────────────────

def test_dedup_and_sort_removes_duplicates():
    """When two candles share the same date, only the last one is kept."""
    candles = [
        {"date": "2024-01-01", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 0},
        {"date": "2024-01-02", "open": 2, "high": 2, "low": 2, "close": 2, "volume": 0},
        {"date": "2024-01-01", "open": 9, "high": 9, "low": 9, "close": 9, "volume": 0},
    ]
    result = _dedup_and_sort(candles)
    assert len(result) == 2
    dates = [c["date"] for c in result]
    assert dates == sorted(dates)
    # Last occurrence of 2024-01-01 wins
    jan1 = next(c for c in result if c["date"] == "2024-01-01")
    assert jan1["close"] == 9.0


def test_dedup_and_sort_ascending():
    """Output must be sorted chronologically ascending."""
    candles = [
        {"date": "2024-01-03", "open": 3, "high": 3, "low": 3, "close": 3, "volume": 0},
        {"date": "2024-01-01", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 0},
        {"date": "2024-01-02", "open": 2, "high": 2, "low": 2, "close": 2, "volume": 0},
    ]
    result = _dedup_and_sort(candles)
    assert [c["date"] for c in result] == ["2024-01-01", "2024-01-02", "2024-01-03"]


# ─────────────────────────────────────────────────────────────────────────────
# fetch_symbol — mocked EtoroClient
# ─────────────────────────────────────────────────────────────────────────────

def _make_raw_candles(n: int, start_date: str = "2020-01-01") -> list[dict]:
    """Build n synthetic eToro-format candles using real eToro field names."""
    import datetime
    base = datetime.date.fromisoformat(start_date)
    result = []
    for i in range(n):
        d = (base + datetime.timedelta(days=i)).isoformat()
        result.append({
            "time": d,               # eToro uses "time" (per our _ETORO_FIELD_MAP)
            "openPrice":  100.0 + i,
            "highPrice":  101.0 + i,
            "lowPrice":    99.0 + i,
            "closePrice": 100.5 + i,
            "volumeValue": 1_000_000,
        })
    return result


def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.get_candles = AsyncMock()
    client.get_instrument_id = AsyncMock(return_value="12345")
    return client


def test_fetch_symbol_parses_etoro_format(mock_client, tmp_path):
    """fetch_symbol must parse eToro-style field names and return valid candles."""
    raw = _make_raw_candles(300)
    mock_client.get_candles.return_value = raw

    with patch("src.backtest.data._CACHE_DIR", tmp_path):
        result = run_async(fetch_symbol("AAPL", mock_client, years=1, force=True))

    assert len(result) == 300
    assert all(c["close"] > 0 for c in result)
    assert result[0]["date"] < result[-1]["date"]   # ascending


def test_fetch_symbol_calls_get_candles_with_correct_params(mock_client, tmp_path):
    """
    fetch_symbol must call client.get_candles with interval='D1' and direction='asc'.
    The instrument ID resolution (symbol→id) must happen inside EtoroClient.get_candles;
    we verify the outer call signature here.
    """
    raw = _make_raw_candles(300)
    mock_client.get_candles.return_value = raw

    with patch("src.backtest.data._CACHE_DIR", tmp_path):
        run_async(fetch_symbol("TSLA", mock_client, years=1, force=True))

    mock_client.get_candles.assert_called_once()
    call_kwargs = mock_client.get_candles.call_args
    # Symbol must be passed
    assert call_kwargs.args[0] == "TSLA" or call_kwargs.kwargs.get("symbol") == "TSLA"
    # Interval must be D1
    assert "D1" in str(call_kwargs)


def test_fetch_symbol_returns_cached_without_network(mock_client, tmp_path):
    """If a cache CSV exists, fetch_symbol must NOT call get_candles."""
    # Pre-populate cache
    raw = _make_raw_candles(300)
    candles = [_normalise_candle(c) for c in raw]
    with patch("src.backtest.data._CACHE_DIR", tmp_path):
        save_candles("NVDA", [c for c in candles if c])
        result = run_async(fetch_symbol("NVDA", mock_client, years=1, force=False))

    mock_client.get_candles.assert_not_called()
    assert len(result) == 300


def test_fetch_symbol_deduplicates_overlapping_pages(mock_client, tmp_path):
    """
    Simulate a 2-page response where page 2 overlaps with page 1 on 50 bars.
    fetch_symbol must deduplicate and return the correct total.
    """
    page1 = _make_raw_candles(300, "2021-01-01")   # bars 0..299
    page2 = _make_raw_candles(300, "2021-10-28")   # bars 300..599, overlaps ~50 bars with page1

    # First call returns page1, second call returns page2
    mock_client.get_candles.side_effect = [page1, page2]

    with patch("src.backtest.data._CACHE_DIR", tmp_path):
        with patch("src.backtest.data._PAGE_LIMIT", 300):
            result = run_async(fetch_symbol("BTC", mock_client, years=2, force=True))

    # Result should have no duplicate dates and be sorted
    dates = [c["date"] for c in result]
    assert len(dates) == len(set(dates)), "Duplicate dates found after merge"
    assert dates == sorted(dates), "Dates not in ascending order"


def test_fetch_symbol_returns_empty_on_api_failure(mock_client, tmp_path):
    """If get_candles raises, fetch_symbol must return [] (not propagate)."""
    mock_client.get_candles.side_effect = RuntimeError("API error")

    with patch("src.backtest.data._CACHE_DIR", tmp_path):
        result = run_async(fetch_symbol("ETH", mock_client, force=True))

    assert result == []


def test_fetch_symbol_warns_if_too_few_bars(mock_client, tmp_path, caplog):
    """When API returns fewer bars than _MIN_BARS_FOR_BACKTEST, a warning is logged."""
    raw = _make_raw_candles(50)   # well below minimum
    mock_client.get_candles.return_value = raw

    import logging
    with patch("src.backtest.data._CACHE_DIR", tmp_path):
        with caplog.at_level(logging.WARNING, logger="src.backtest.data"):
            result = run_async(fetch_symbol("SOL", mock_client, force=True))

    assert len(result) == 50
    assert any("only" in msg.lower() and "bars" in msg.lower() for msg in caplog.messages), (
        f"Expected a 'too few bars' warning, got: {caplog.messages}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# EtoroClient.get_candles — endpoint and param verification
# ─────────────────────────────────────────────────────────────────────────────

def test_get_candles_uses_correct_endpoint():
    """
    EtoroClient.get_candles must call _request with
    path='/market-data/instruments/history/candles'.
    """
    from src.core.etoro_client import EtoroClient

    with patch.dict("os.environ", {
        "ETORO_PUBLIC_API_KEY": "test-key",
        "ETORO_USER_KEY": "test-user-key",
        "ETORO_MODE": "demo",
    }):
        client = EtoroClient()
        client._client = MagicMock()

        captured_path = {}

        async def fake_request(method, path, **kwargs):
            captured_path["path"] = path
            captured_path["params"] = kwargs.get("params", {})
            return {"data": []}

        client._request = fake_request

        # Mock universe cache to return an instrumentId without network
        with patch("src.core.etoro_client._cache_id", return_value="99"):
            run_async(client.get_candles("AAPL", interval="D1", count=100))

    assert captured_path.get("path") == "/market-data/instruments/history/candles", (
        f"Wrong endpoint: {captured_path.get('path')}"
    )


def test_get_candles_translates_D1_to_OneDay():
    """Interval 'D1' must be translated to 'OneDay' in the request params."""
    from src.core.etoro_client import EtoroClient

    with patch.dict("os.environ", {
        "ETORO_PUBLIC_API_KEY": "k", "ETORO_USER_KEY": "u", "ETORO_MODE": "demo"
    }):
        client = EtoroClient()

        captured = {}

        async def fake_request(method, path, **kwargs):
            captured.update(kwargs.get("params", {}))
            return {"data": []}

        client._request = fake_request

        with patch("src.core.etoro_client._cache_id", return_value="42"):
            run_async(client.get_candles("MSFT", interval="D1", count=100))

    assert captured.get("interval") == "OneDay", (
        f"Expected interval='OneDay', got {captured.get('interval')!r}"
    )


def test_get_candles_uses_instrument_id_not_ticker():
    """
    The request params must include 'instrumentId' (numeric), not the raw ticker.
    """
    from src.core.etoro_client import EtoroClient

    with patch.dict("os.environ", {
        "ETORO_PUBLIC_API_KEY": "k", "ETORO_USER_KEY": "u", "ETORO_MODE": "demo"
    }):
        client = EtoroClient()

        captured = {}

        async def fake_request(method, path, **kwargs):
            captured.update(kwargs.get("params", {}))
            return []

        client._request = fake_request

        # Simulate: cache returns id "777" for "TSLA"
        with patch("src.core.etoro_client._cache_id", return_value="777"):
            run_async(client.get_candles("TSLA", interval="D1", count=50))

    assert "instrumentId" in captured, "instrumentId param missing from request"
    assert captured["instrumentId"] == "777"
    assert "TSLA" not in str(captured.get("instrumentId", "")), (
        "Request must use numeric instrumentId, not raw ticker"
    )


def test_get_candles_returns_empty_when_instrument_id_not_found():
    """If instrumentId cannot be resolved (cache miss + API miss), return []."""
    from src.core.etoro_client import EtoroClient

    with patch.dict("os.environ", {
        "ETORO_PUBLIC_API_KEY": "k", "ETORO_USER_KEY": "u", "ETORO_MODE": "demo"
    }):
        client = EtoroClient()

        async def fake_request(*a, **kw):
            return {"data": []}

        client._request = fake_request
        client.get_instrument_id = AsyncMock(return_value=None)

        with patch("src.core.etoro_client._cache_id", return_value=None):
            result = run_async(client.get_candles("UNKNOWN_SYM", interval="D1"))

    assert result == []


def test_get_candles_limit_capped_at_1000():
    """count > 1000 must be capped to 1000 in the request (API max)."""
    from src.core.etoro_client import EtoroClient

    with patch.dict("os.environ", {
        "ETORO_PUBLIC_API_KEY": "k", "ETORO_USER_KEY": "u", "ETORO_MODE": "demo"
    }):
        client = EtoroClient()

        captured = {}

        async def fake_request(method, path, **kwargs):
            captured.update(kwargs.get("params", {}))
            return []

        client._request = fake_request

        with patch("src.core.etoro_client._cache_id", return_value="1"):
            run_async(client.get_candles("AAPL", interval="D1", count=9999))

    assert captured.get("limit") <= 1000, (
        f"limit should be capped at 1000, got {captured.get('limit')}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# load_dataframe — dedup, sort, gap detection
# ─────────────────────────────────────────────────────────────────────────────

def test_load_dataframe_deduplicates_cache(tmp_path):
    """load_dataframe must deduplicate dates in the CSV."""
    candles = [
        {"date": "2020-01-01", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 0},
        {"date": "2020-01-02", "open": 2, "high": 2, "low": 2, "close": 2, "volume": 0},
        {"date": "2020-01-01", "open": 9, "high": 9, "low": 9, "close": 9, "volume": 0},
    ] * 100   # repeat to exceed _MIN_BARS_FOR_BACKTEST
    with patch("src.backtest.data._CACHE_DIR", tmp_path):
        save_candles("X", candles)
        df = load_dataframe("X")

    if df is None:
        pytest.skip("Not enough bars after dedup")
    assert df.index.duplicated().sum() == 0


def test_load_dataframe_sorts_ascending(tmp_path):
    """Dates in DataFrame must be in ascending order."""
    # Build 300 candles in reverse order
    import datetime
    base = datetime.date(2021, 1, 1)
    candles = []
    for i in range(300, 0, -1):
        d = (base + datetime.timedelta(days=i)).isoformat()
        candles.append({"date": d, "open": 100, "high": 101, "low": 99,
                        "close": 100, "volume": 1000})
    with patch("src.backtest.data._CACHE_DIR", tmp_path):
        save_candles("Y", candles)
        df = load_dataframe("Y")

    assert df is not None
    assert df.index.is_monotonic_increasing


def test_load_dataframe_warns_on_large_gap(tmp_path, caplog):
    """A gap larger than _MAX_GAP_DAYS must produce a warning log."""
    import datetime
    import logging
    base = datetime.date(2021, 1, 1)
    candles = []
    for i in range(300):
        # Introduce a 30-day gap at bar 150
        days = i if i < 150 else i + 30
        d = (base + datetime.timedelta(days=days)).isoformat()
        candles.append({"date": d, "open": 100, "high": 101, "low": 99,
                        "close": 100, "volume": 1000})
    with patch("src.backtest.data._CACHE_DIR", tmp_path):
        save_candles("Z", candles)
        with caplog.at_level(logging.WARNING, logger="src.backtest.data"):
            df = load_dataframe("Z")

    assert df is not None
    gap_warnings = [m for m in caplog.messages if "gap" in m.lower()]
    assert gap_warnings, f"Expected gap warning, got: {caplog.messages}"


def test_load_dataframe_returns_none_when_below_minimum(tmp_path):
    """Returns None when fewer than _MIN_BARS_FOR_BACKTEST bars are in the cache."""
    candles = [
        {"date": f"2021-01-{i:02d}", "open": 100, "high": 101, "low": 99,
         "close": 100, "volume": 1000}
        for i in range(1, 30)   # only 29 bars
    ]
    with patch("src.backtest.data._CACHE_DIR", tmp_path):
        save_candles("W", candles)
        df = load_dataframe("W")

    assert df is None
