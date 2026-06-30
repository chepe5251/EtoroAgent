"""
Pure-function technical indicator calculations.
All inputs are plain Python lists (oldest → newest).
"""
import math
from typing import Optional


def ema(values: list[float], period: int) -> list[float]:
    """Exponential Moving Average."""
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    result: list[float] = []
    # seed with SMA of first `period` values
    seed = sum(values[:period]) / period
    result.append(seed)
    for v in values[period:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """RSI — returns the latest value or None if not enough data."""
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_gain == 0 and avg_loss == 0:
        # No price movement at all — RSI is undefined (matches pandas-ta NaN behaviour)
        return None
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Optional[dict]:
    """MACD — returns {macd_line, signal_line, histogram} or None."""
    if len(closes) < slow + signal_period:
        return None
    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)
    # align by taking the last min(len) values
    n = min(len(fast_ema), len(slow_ema))
    macd_line = [fast_ema[-n + i] - slow_ema[-n + i] for i in range(n)]
    if len(macd_line) < signal_period:
        return None
    signal_line = ema(macd_line, signal_period)
    if not signal_line:
        return None
    latest_macd = macd_line[-1]
    latest_signal = signal_line[-1]
    return {
        "macd_line": latest_macd,
        "signal_line": latest_signal,
        "histogram": latest_macd - latest_signal,
    }


def bollinger_bands(
    closes: list[float], period: int = 20, std_dev: float = 2.0
) -> Optional[dict]:
    """Bollinger Bands — returns {upper, middle, lower} or None."""
    if len(closes) < period:
        return None
    window = closes[-period:]
    middle = sum(window) / period
    variance = sum((x - middle) ** 2 for x in window) / period
    std = math.sqrt(variance)
    return {
        "upper": middle + std_dev * std,
        "middle": middle,
        "lower": middle - std_dev * std,
        "bandwidth": (std_dev * 2 * std) / middle if middle else 0,
    }


def atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> Optional[float]:
    """Average True Range — returns latest ATR value or None."""
    if len(closes) < period + 1:
        return None
    true_ranges = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)
    if len(true_ranges) < period:
        return None
    # Wilder smoothing
    atr_val = sum(true_ranges[:period]) / period
    for tr in true_ranges[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period
    return atr_val


def relative_volume(volumes: list[float], period: int = 20) -> Optional[float]:
    """Current volume relative to the N-period average."""
    if len(volumes) < period + 1:
        return None
    avg = sum(volumes[-period - 1 : -1]) / period
    if avg == 0:
        return None
    return volumes[-1] / avg


def compute_all(candles: list[dict]) -> dict:
    """
    Compute all indicators from a list of OHLCV candle dicts.
    Each candle: {open, high, low, close, volume}
    Returns a flat indicator dict.
    """
    opens = [c["open"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]
    volumes = [c.get("volume", 0) for c in candles]

    ema20_series = ema(closes, 20)
    ema50_series = ema(closes, 50)

    return {
        "rsi_14": rsi(closes, 14),
        "macd": macd(closes, 12, 26, 9),
        "ema_20": ema20_series[-1] if ema20_series else None,
        "ema_50": ema50_series[-1] if ema50_series else None,
        "bollinger": bollinger_bands(closes, 20, 2.0),
        "atr_14": atr(highs, lows, closes, 14),
        "relative_volume": relative_volume(volumes, 20),
        "last_close": closes[-1] if closes else None,
        "last_volume": volumes[-1] if volumes else None,
    }
