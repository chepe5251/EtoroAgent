"""
MCP server — CryptoPanic news & sentiment tools.
Docs: https://cryptopanic.com/api/developer/
Free tier: 100 req/day
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv()

mcp = FastMCP("cryptopanic")

_BASE = "https://cryptopanic.com/api/v1"

_VALID_FILTERS = frozenset(
    ["rising", "hot", "bullish", "bearish", "important", "saved", "lol"]
)
_VALID_KINDS = frozenset(["news", "media", "all"])


def _api_key() -> str:
    key = os.getenv("CRYPTOPANIC_API_KEY", "")
    if not key:
        raise RuntimeError("CRYPTOPANIC_API_KEY not set")
    return key


@mcp.tool()
async def cryptopanic_get_news(
    currencies: list[str],
    filter: str = "hot",
    kind: str = "news",
) -> list[dict]:
    """
    Get recent crypto news from CryptoPanic, filtered by sentiment/popularity.
    Only for crypto assets (BTC, ETH, SOL, etc.).

    Args:
        currencies: List of crypto symbols, e.g. ['BTC', 'ETH']
        filter: One of 'rising', 'hot', 'bullish', 'bearish', 'important'
                (default 'hot' — most engaged stories right now)
        kind: 'news' | 'media' | 'all' (default 'news')

    Returns:
        List of news items with: title, published_at, source, url,
        votes (positive, negative, important), sentiment
    """
    filter = filter if filter in _VALID_FILTERS else "hot"
    kind = kind if kind in _VALID_KINDS else "news"

    params = {
        "auth_token": _api_key(),
        "currencies": ",".join(c.upper() for c in currencies),
        "filter": filter,
        "kind": kind,
        "public": "true",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(f"{_BASE}/posts/", params=params)
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results", [])
    out = []
    for item in results[:15]:
        votes = item.get("votes", {})
        source = item.get("source", {})
        out.append({
            "title": item.get("title", ""),
            "published_at": item.get("published_at", ""),
            "source": source.get("title", "") if isinstance(source, dict) else "",
            "url": item.get("url", ""),
            "positive_votes": votes.get("positive", 0),
            "negative_votes": votes.get("negative", 0),
            "important_votes": votes.get("important", 0),
            "panic": votes.get("panic", 0),
        })
    return out


@mcp.tool()
async def cryptopanic_get_sentiment_summary(currencies: list[str]) -> dict:
    """
    Get a quick bullish/bearish breakdown for crypto symbols by comparing
    the count of bullish vs bearish tagged posts in the last 24h.

    Args:
        currencies: List of crypto symbols, e.g. ['BTC', 'ETH']

    Returns:
        Dict mapping symbol → {bullish_count, bearish_count, net_sentiment}
        where net_sentiment > 0 means more bullish posts
    """
    async with httpx.AsyncClient(timeout=20.0) as client:
        summary = {}
        for currency in currencies:
            try:
                bullish_resp = await client.get(
                    f"{_BASE}/posts/",
                    params={
                        "auth_token": _api_key(),
                        "currencies": currency.upper(),
                        "filter": "bullish",
                        "kind": "news",
                        "public": "true",
                    },
                )
                bullish_resp.raise_for_status()
                bullish_count = len(bullish_resp.json().get("results", []))

                bearish_resp = await client.get(
                    f"{_BASE}/posts/",
                    params={
                        "auth_token": _api_key(),
                        "currencies": currency.upper(),
                        "filter": "bearish",
                        "kind": "news",
                        "public": "true",
                    },
                )
                bearish_resp.raise_for_status()
                bearish_count = len(bearish_resp.json().get("results", []))

                total = bullish_count + bearish_count
                net = (bullish_count - bearish_count) / total if total > 0 else 0
                summary[currency.upper()] = {
                    "bullish_count": bullish_count,
                    "bearish_count": bearish_count,
                    "net_sentiment": round(net, 3),
                }
            except Exception as exc:
                summary[currency.upper()] = {"error": str(exc)}

    return summary


if __name__ == "__main__":
    mcp.run()
