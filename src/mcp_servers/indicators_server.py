"""
MCP server — technical indicator tools.
Fetches candles from eToro internally and computes all indicators.
Pure read-only — never touches orders.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

from src.core.etoro_client import EtoroClient
from src.tools.technical import compute_all, rsi, macd, ema, bollinger_bands, atr

mcp = FastMCP("indicators")

_client: EtoroClient | None = None


async def _get_client() -> EtoroClient:
    global _client
    if _client is None:
        import httpx
        _client = EtoroClient()
        _client._client = httpx.AsyncClient(timeout=30.0)
    return _client


def _normalise(candles: list[dict]) -> list[dict]:
    out = []
    for c in candles:
        out.append({
            "open":   float(c.get("open",   c.get("Open",   0))),
            "high":   float(c.get("high",   c.get("High",   0))),
            "low":    float(c.get("low",    c.get("Low",    0))),
            "close":  float(c.get("close",  c.get("Close",  0))),
            "volume": float(c.get("volume", c.get("Volume", 0))),
            "time":   c.get("time", c.get("Time", c.get("timestamp", ""))),
        })
    return out


@mcp.tool()
async def indicators_full_analysis(
    symbol: str,
    interval: str = "M15",
    count: int = 100,
) -> dict:
    """
    Fetch recent candles for a symbol and compute all technical indicators
    in one call. This is the primary tool for technical analysis.

    Args:
        symbol: Ticker, e.g. 'BTC', 'ETH', 'AAPL'
        interval: Candle timeframe (default M15)
        count: Number of candles (default 100; use 200 for EMA50/MACD accuracy)

    Returns:
        Dict with:
          - symbol: str
          - interval: str
          - candle_count: int
          - last_close: float
          - rsi_14: float | null  (< 30 = oversold, > 70 = overbought)
          - macd: {macd_line, signal_line, histogram} | null
          - ema_20: float | null
          - ema_50: float | null
          - bollinger: {upper, middle, lower, bandwidth} | null
          - atr_14: float | null  (use for stop-loss sizing)
          - relative_volume: float | null  (> 1.5 = elevated volume)
    """
    client = await _get_client()
    raw = await client.get_candles(symbol, interval, count)
    candles = _normalise(raw)
    if not candles:
        return {"error": f"No candles returned for {symbol}"}

    indicators = compute_all(candles)
    return {
        "symbol": symbol,
        "interval": interval,
        "candle_count": len(candles),
        **indicators,
    }


@mcp.tool()
async def indicators_compute_from_data(candles: list[dict]) -> dict:
    """
    Compute technical indicators from raw candle data you already have.
    Use this if you fetched candles via etoro_get_candles and want to
    compute indicators without a second API call.

    Args:
        candles: List of dicts with keys: open, high, low, close, volume

    Returns:
        Same structure as indicators_full_analysis (without symbol/interval)
    """
    normalised = _normalise(candles)
    if not normalised:
        return {"error": "Empty candle list"}
    return {"candle_count": len(normalised), **compute_all(normalised)}


if __name__ == "__main__":
    mcp.run()
