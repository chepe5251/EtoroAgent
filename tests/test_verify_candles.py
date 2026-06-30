"""
Mocked tests for verify_candles.py.

All assertions are against pure analysis functions; no network calls.
Async functions that call EtoroClient are tested via AsyncMock.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

# Force DEMO before the module-level import in verify_candles.py triggers
os.environ.setdefault("ETORO_PUBLIC_API_KEY", "test-key")
os.environ.setdefault("ETORO_USER_KEY", "test-user-key")
os.environ["ETORO_MODE"] = "demo"

from src.backtest.verify_candles import (
    INTERVAL_CANDIDATES,
    SPLIT_THRESHOLD,
    _MIN_BARS,
    _probe_intervals,
    _try_parse_date,
    analyze_fields,
    compute_depth,
    detect_splits,
    suggest_field_map,
)


# ────────────────────────────────────────────────────────────────────────────────
# analyze_fields
# ────────────────────────────────────────────────────────────────────────────────

def test_analyze_fields_all_standard_keys():
    candle = {
        "date": "2024-01-01", "open": 100.0, "high": 110.0,
        "low": 95.0, "close": 105.0, "volume": 5000.0,
    }
    r = analyze_fields(candle)
    assert r["matched"]["date"]   == "date"
    assert r["matched"]["open"]   == "open"
    assert r["matched"]["close"]  == "close"
    assert r["matched"]["volume"] == "volume"
    assert r["missing"]   == []
    assert r["unmatched"] == []


def test_analyze_fields_etoro_vendor_keys():
    candle = {
        "time": "2024-01-01T00:00:00Z",
        "openPrice": 100.0, "highPrice": 110.0,
        "lowPrice": 95.0,   "closePrice": 105.0,
        "volumeValue": 5000.0,
    }
    r = analyze_fields(candle)
    assert r["matched"]["date"]   == "time"
    assert r["matched"]["open"]   == "openPrice"
    assert r["matched"]["close"]  == "closePrice"
    assert r["matched"]["volume"] == "volumeValue"
    assert r["missing"]   == []
    assert r["unmatched"] == []


def test_analyze_fields_entirely_unknown_keys():
    candle = {"o": 100.0, "h": 110.0, "l": 95.0, "c": 105.0, "v": 5000.0, "t": "2024-01-01"}
    r = analyze_fields(candle)
    assert set(r["missing"]) == {"date", "open", "high", "low", "close", "volume"}
    assert set(r["unmatched"]) == {"o", "h", "l", "c", "v", "t"}


def test_analyze_fields_extra_unknown_key():
    candle = {
        "date": "2024-01-01", "open": 100.0, "high": 110.0,
        "low": 95.0, "close": 105.0, "volume": 5000.0,
        "extra_field": "ignored",
    }
    r = analyze_fields(candle)
    assert r["missing"]   == []
    assert r["unmatched"] == ["extra_field"]


def test_analyze_fields_timestamp_variant():
    candle = {
        "timestamp": "2024-01-01T00:00:00Z",
        "close": 105.0, "open": 100.0, "high": 110.0, "low": 95.0, "volume": 1000.0,
    }
    r = analyze_fields(candle)
    assert r["matched"]["date"] == "timestamp"
    assert r["missing"] == []


# ────────────────────────────────────────────────────────────────────────────────
# _try_parse_date
# ────────────────────────────────────────────────────────────────────────────────

def test_try_parse_date_iso_with_z():
    d = _try_parse_date("2024-01-15T00:00:00Z")
    assert d is not None
    assert d.year == 2024 and d.month == 1 and d.day == 15


def test_try_parse_date_plain():
    d = _try_parse_date("2022-06-30")
    assert d is not None
    assert d.year == 2022 and d.month == 6


def test_try_parse_date_invalid():
    d = _try_parse_date("not-a-date")
    assert d is None


# ────────────────────────────────────────────────────────────────────────────────
# compute_depth
# ────────────────────────────────────────────────────────────────────────────────

def test_compute_depth_empty():
    d = compute_depth([])
    assert d["count"] == 0
    assert d["too_few"] is True
    assert d["first_date"] is None


def test_compute_depth_two_candles_too_few():
    candles = [
        {"date": "2020-01-01", "close": 100},
        {"date": "2023-01-01", "close": 150},
    ]
    d = compute_depth(candles)
    assert d["count"] == 2
    assert d["first_date"] == "2020-01-01"
    assert d["last_date"]  == "2023-01-01"
    assert d["too_few"] is True
    assert d["years_coverage"] is not None
    assert 2.9 < d["years_coverage"] < 3.1


def test_compute_depth_sufficient_count():
    candles = [
        {"date": f"2024-{(i // 28 + 1):02d}-{(i % 28 + 1):02d}", "close": 100.0 + i}
        for i in range(260)
    ]
    d = compute_depth(candles)
    assert d["count"] == 260
    assert d["too_few"] is False


def test_compute_depth_time_field_variant():
    candles = [
        {"time": "2019-01-02T00:00:00Z", "close": 100},
        {"time": "2024-06-30T00:00:00Z", "close": 200},
    ]
    d = compute_depth(candles)
    # "time" is the second candidate for "date" in _ETORO_FIELD_MAP
    assert d["first_date"] == "2019-01-02T00:00:00Z"
    assert d["last_date"]  == "2024-06-30T00:00:00Z"
    assert d["years_coverage"] is not None
    assert d["years_coverage"] > 5.0


def test_compute_depth_no_date_field():
    candles = [{"close": 100}, {"close": 105}]
    d = compute_depth(candles)
    assert d["count"] == 2
    assert d["first_date"] is None
    assert d["years_coverage"] is None


# ────────────────────────────────────────────────────────────────────────────────
# detect_splits
# ────────────────────────────────────────────────────────────────────────────────

def test_detect_splits_no_split():
    candles = [
        {"close": 100.0, "date": "2024-01-01"},
        {"close": 102.0, "date": "2024-01-02"},
        {"close": 101.5, "date": "2024-01-03"},
    ]
    assert detect_splits(candles) == []


def test_detect_splits_two_for_one_split():
    candles = [
        {"close": 200.0, "date": "2024-01-01"},
        {"close": 100.0, "date": "2024-01-02"},  # −50 %
        {"close": 101.0, "date": "2024-01-03"},
    ]
    jumps = detect_splits(candles)
    assert len(jumps) == 1
    assert jumps[0]["date"] == "2024-01-02"
    assert abs(jumps[0]["pct_change"] - 0.5) < 1e-9


def test_detect_splits_reverse_split():
    candles = [
        {"close": 100.0, "date": "2024-01-01"},
        {"close": 300.0, "date": "2024-01-02"},  # +200 %
    ]
    jumps = detect_splits(candles)
    assert len(jumps) == 1
    assert jumps[0]["pct_change"] > 0.25


def test_detect_splits_borderline_below_threshold():
    candles = [
        {"close": 100.0, "date": "2024-01-01"},
        {"close": 124.9, "date": "2024-01-02"},  # 24.9 % — under threshold
    ]
    assert detect_splits(candles) == []


def test_detect_splits_borderline_above_threshold():
    candles = [
        {"close": 100.0, "date": "2024-01-01"},
        {"close": 125.1, "date": "2024-01-02"},  # 25.1 % — over threshold
    ]
    jumps = detect_splits(candles)
    assert len(jumps) == 1


def test_detect_splits_multiple_events():
    candles = [
        {"close": 200.0, "date": "2024-01-01"},
        {"close": 100.0, "date": "2024-01-02"},  # split 1
        {"close": 101.0, "date": "2024-01-03"},
        {"close":  50.0, "date": "2024-01-04"},  # split 2
    ]
    jumps = detect_splits(candles)
    assert len(jumps) == 2


def test_detect_splits_works_with_closeprice_field():
    # _normalise_candle() maps "closePrice" → close, so detect_splits works
    candles = [
        {"closePrice": 200.0, "date": "2024-01-01"},
        {"closePrice": 100.0, "date": "2024-01-02"},
    ]
    jumps = detect_splits(candles)
    assert len(jumps) == 1


# ────────────────────────────────────────────────────────────────────────────────
# suggest_field_map
# ────────────────────────────────────────────────────────────────────────────────

def test_suggest_field_map_promotes_matched_key():
    matched = {
        "date": "time", "open": "openPrice", "high": "highPrice",
        "low": "lowPrice", "close": "closePrice", "volume": "volumeValue",
    }
    sample = {
        "time": "2024-01-01T00:00:00Z", "openPrice": 100.0, "highPrice": 110.0,
        "lowPrice": 95.0, "closePrice": 105.0, "volumeValue": 5000.0,
    }
    suggested, is_unambiguous = suggest_field_map(matched, [], sample)
    assert suggested["date"][0]   == "time"
    assert suggested["close"][0]  == "closePrice"
    assert suggested["volume"][0] == "volumeValue"
    assert is_unambiguous is True


def test_suggest_field_map_unambiguous_all_matched():
    matched = {f: f for f in ["date", "open", "high", "low", "close", "volume"]}
    sample  = {f: 1.0 for f in matched}
    sample["date"] = "2024-01-01"
    _, is_unambiguous = suggest_field_map(matched, [], sample)
    assert is_unambiguous is True


def test_suggest_field_map_numeric_unknown_marks_ambiguous():
    matched = {
        "date": "date", "open": "open", "high": "high",
        "low": "low", "close": "close", "volume": "volume",
    }
    sample = {k: 1.0 for k in matched}
    sample["date"] = "2024-01-01"
    sample["somePrice"] = 104.5  # unknown numeric key → ambiguous
    _, is_unambiguous = suggest_field_map(matched, ["somePrice"], sample)
    assert is_unambiguous is False


def test_suggest_field_map_missing_field_marks_ambiguous():
    matched = {f: f for f in ["date", "open", "high", "low", "close"]}  # volume missing
    sample  = {f: 1.0 for f in matched}
    sample["date"] = "2024-01-01"
    _, is_unambiguous = suggest_field_map(matched, [], sample)
    assert is_unambiguous is False


# ────────────────────────────────────────────────────────────────────────────────
# _probe_intervals (async, mocked)
# ────────────────────────────────────────────────────────────────────────────────

SAMPLE_CANDLE = {
    "date": "2024-01-01", "open": 100.0, "high": 110.0,
    "low": 95.0, "close": 105.0, "volume": 1000.0,
}


@pytest.mark.asyncio
async def test_probe_intervals_calls_get_candles_for_each_candidate():
    mock_client = AsyncMock()
    mock_client.get_candles = AsyncMock(return_value=[SAMPLE_CANDLE] * 5)

    results = await _probe_intervals(mock_client, "AAPL")

    assert mock_client.get_candles.call_count == len(INTERVAL_CANDIDATES)
    for iv in INTERVAL_CANDIDATES:
        assert iv in results
        ok, count = results[iv]
        assert ok is True
        assert count == 5


@pytest.mark.asyncio
async def test_probe_intervals_captures_per_interval_exception():
    mock_client = AsyncMock()
    mock_client.get_candles = AsyncMock(side_effect=Exception("HTTP 400 Bad Request"))

    results = await _probe_intervals(mock_client, "AAPL")

    assert len(results) == len(INTERVAL_CANDIDATES)
    for iv, (ok, info) in results.items():
        assert ok is False
        assert "400" in info or "Bad Request" in info


@pytest.mark.asyncio
async def test_probe_intervals_mixed_results():
    call_count = [0]

    async def fake_get_candles(symbol, interval, count, **kwargs):
        call_count[0] += 1
        if interval == "OneDay":
            return [SAMPLE_CANDLE] * 5
        raise Exception(f"Unsupported interval: {interval}")

    mock_client = AsyncMock()
    mock_client.get_candles = fake_get_candles

    results = await _probe_intervals(mock_client, "AAPL")

    ok_day, cnt_day = results["OneDay"]
    assert ok_day is True
    assert cnt_day == 5

    for iv in INTERVAL_CANDIDATES:
        if iv != "OneDay":
            ok, _ = results[iv]
            assert ok is False
