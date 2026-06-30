"""
MCP server — eToro read-only tools.
Exposes candles, rates, portfolio, balance.
NO execution tools — the LLM can never move money through this server.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Make sure src/ is importable when running as a subprocess
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

from src.core.etoro_client import EtoroClient

mcp = FastMCP("etoro-readonly")

# Each MCP server subprocess has its own client instance
_client: EtoroClient | None = None
_client_ctx = None


async def _get_client() -> EtoroClient:
    global _client, _client_ctx
    if _client is None:
        _client = EtoroClient()
        _client_ctx = _client._client  # type: ignore[attr-defined]
        import httpx
        _client._client = httpx.AsyncClient(timeout=30.0)
    return _client


@mcp.tool()
async def etoro_get_candles(symbol: str, interval: str = "M15", count: int = 100) -> list:
    """
    Fetch OHLCV (Open/High/Low/Close/Volume) candlestick data for a symbol.

    Args:
        symbol: Ticker symbol, e.g. 'BTC', 'ETH', 'AAPL', 'TSLA'
        interval: Candle timeframe — M1, M5, M15, M30, H1, H4, D1 (default M15)
        count: Number of candles to return (default 100, max 500)

    Returns:
        List of candle dicts with keys: open, high, low, close, volume, time
    """
    client = await _get_client()
    return await client.get_candles(symbol, interval, count)


@mcp.tool()
async def etoro_get_rates(symbols: list[str]) -> dict:
    """
    Get current bid/ask/last rates for one or more symbols.

    Args:
        symbols: List of ticker symbols, e.g. ['BTC', 'ETH']

    Returns:
        Dict mapping symbol → {bid, ask, close/last, instrumentName}
    """
    client = await _get_client()
    return await client.get_rates(symbols)


@mcp.tool()
async def etoro_get_portfolio() -> list:
    """
    Get all currently open positions in the account (demo or real).

    Returns:
        List of position dicts with keys: symbol, positionId, amount,
        openRate, currentRate, isPurchase (True=long), stopLoss, etc.
    """
    client = await _get_client()
    return await client.get_portfolio()


@mcp.tool()
async def etoro_get_balance() -> dict:
    """
    Get the current account balance and available funds.

    Returns:
        Dict with keys: availableToTrade, balance/equity, currency
    """
    client = await _get_client()
    return await client.get_balance()


if __name__ == "__main__":
    mcp.run()
