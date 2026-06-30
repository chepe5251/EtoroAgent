"""
Tests for technical indicator calculations.
Run with: pytest tests/test_technical.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.technical import atr, bollinger_bands, ema, macd, rsi, relative_volume, compute_all


def _linear(start: float, step: float, n: int) -> list[float]:
    return [start + i * step for i in range(n)]


def test_ema_basic():
    result = ema([1.0, 2.0, 3.0, 4.0, 5.0], period=3)
    assert len(result) > 0
    # EMA with k=0.5 and seed=2: next = 3*0.5+2*0.5=2.5, then 4*0.5+2.5*0.5=3.25...
    assert result[0] == pytest.approx(2.0)


def test_ema_insufficient_data():
    assert ema([1.0, 2.0], period=3) == []


def test_rsi_returns_none_on_insufficient_data():
    assert rsi([1.0, 2.0], period=14) is None


def test_rsi_constant_series():
    # No change → 0 gains → RSI should be defined (0 losses → RS=inf → RSI=100)
    closes = [100.0] * 20
    result = rsi(closes, 14)
    assert result == pytest.approx(100.0)


def test_rsi_range():
    closes = _linear(100.0, 1.0, 50)
    result = rsi(closes, 14)
    assert result is not None
    assert 0 <= result <= 100


def test_macd_insufficient_data():
    assert macd([1.0] * 10) is None


def test_macd_keys():
    closes = _linear(100.0, 0.5, 60)
    result = macd(closes)
    assert result is not None
    assert "macd_line" in result
    assert "signal_line" in result
    assert "histogram" in result


def test_bollinger_bands_keys():
    closes = [100.0 + i * 0.1 for i in range(30)]
    result = bollinger_bands(closes, 20, 2.0)
    assert result is not None
    assert result["upper"] > result["middle"] > result["lower"]


def test_bollinger_bands_insufficient():
    assert bollinger_bands([1.0] * 5, 20) is None


def test_atr_basic():
    h = [105.0] * 20
    l = [95.0] * 20
    c = [100.0] * 20
    result = atr(h, l, c, 14)
    assert result is not None
    assert result > 0


def test_atr_insufficient():
    assert atr([1.0], [1.0], [1.0], 14) is None


def test_relative_volume():
    vols = [100.0] * 20 + [200.0]
    result = relative_volume(vols, 20)
    assert result == pytest.approx(2.0)


def test_compute_all_keys():
    candles = [
        {"open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000}
        for _ in range(100)
    ]
    result = compute_all(candles)
    assert "rsi_14" in result
    assert "macd" in result
    assert "ema_20" in result
    assert "ema_50" in result
    assert "bollinger" in result
    assert "atr_14" in result
    assert "relative_volume" in result
    assert "last_close" in result
