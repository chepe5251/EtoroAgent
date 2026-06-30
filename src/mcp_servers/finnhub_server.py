"""
MCP server — Finnhub news sentiment tools (free tier).
Docs: https://finnhub.io/docs/api
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv()

mcp = FastMCP("finnhub")

_BASE = "https://finnhub.io/api/v1"


def _api_key() -> str:
    key = os.getenv("FINNHUB_API_KEY", "")
    if not key:
        raise RuntimeError("FINNHUB_API_KEY not set")
    return key


@mcp.tool()
async def finnhub_get_news_sentiment(symbol: str) -> dict:
    """
    Get news sentiment score for a stock/ETF symbol from Finnhub.
    Only works for equities (AAPL, TSLA, etc.) — not crypto.

    Args:
        symbol: Stock ticker, e.g. 'AAPL', 'TSLA'

    Returns:
        Dict with:
          - symbol: str
          - buzz: {articlesInLastWeek, buzz, weeklyAverage}
          - sentiment: {bearishPercent, bullishPercent}
          - companyNewsScore: float  (-1 to 1 equivalent concept)
          - sectorAverageBullishPercent: float
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{_BASE}/news-sentiment",
            params={"symbol": symbol, "token": _api_key()},
        )
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def finnhub_get_company_news(symbol: str, hours_back: int = 24) -> list[dict]:
    """
    Get recent news articles for a stock symbol.

    Args:
        symbol: Stock ticker, e.g. 'AAPL', 'TSLA'
        hours_back: How many hours back to look (default 24, max 72)

    Returns:
        List of articles with: headline, summary, source, url, datetime, sentiment
    """
    now = datetime.now(timezone.utc)
    from_dt = now - timedelta(hours=max(1, min(hours_back, 72)))
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{_BASE}/company-news",
            params={
                "symbol": symbol,
                "from": from_dt.strftime("%Y-%m-%d"),
                "to": now.strftime("%Y-%m-%d"),
                "token": _api_key(),
            },
        )
        resp.raise_for_status()
        articles = resp.json()
        if not isinstance(articles, list):
            return []
        # Return most recent 10, with only relevant fields
        result = []
        for a in articles[:10]:
            result.append({
                "headline": a.get("headline", ""),
                "summary": (a.get("summary", "") or "")[:200],
                "source": a.get("source", ""),
                "datetime": a.get("datetime", 0),
                "url": a.get("url", ""),
            })
        return result


@mcp.tool()
async def finnhub_get_quote(symbol: str) -> dict:
    """
    Get current stock quote from Finnhub (useful to cross-check eToro rates).

    Args:
        symbol: Stock ticker, e.g. 'AAPL', 'TSLA'

    Returns:
        {c: current, h: high, l: low, o: open, pc: prev_close, t: timestamp}
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{_BASE}/quote",
            params={"symbol": symbol, "token": _api_key()},
        )
        resp.raise_for_status()
        return resp.json()


if __name__ == "__main__":
    mcp.run()
