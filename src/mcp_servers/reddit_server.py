"""
MCP server — Reddit sentiment tools via OAuth2 REST API (no PRAW dependency).
Searches r/wallstreetbets, r/CryptoCurrency, r/stocks for recent high-score posts.
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

mcp = FastMCP("reddit")

_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_API_BASE = "https://oauth.reddit.com"

_CRYPTO_SUBS = ["CryptoCurrency", "CryptoMarkets", "Bitcoin", "ethereum"]
_EQUITY_SUBS = ["wallstreetbets", "stocks", "investing"]
_CRYPTO_TICKERS = {"BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "MATIC"}


def _creds() -> tuple[str, str, str]:
    cid = os.getenv("REDDIT_CLIENT_ID", "")
    secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    ua = os.getenv("REDDIT_USER_AGENT", "etoroAgent/1.0")
    if not cid or not secret:
        raise RuntimeError("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set")
    return cid, secret, ua


async def _get_token() -> str:
    cid, secret, ua = _creds()
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            _TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(cid, secret),
            headers={"User-Agent": ua},
        )
        resp.raise_for_status()
        return resp.json()["access_token"]


@mcp.tool()
async def reddit_get_subreddit_sentiment(
    symbol: str,
    hours_back: int = 6,
    min_score: int = 50,
    limit: int = 25,
) -> dict:
    """
    Search relevant subreddits for recent posts mentioning a symbol
    and return a sentiment snapshot.

    Automatically picks crypto subreddits (r/CryptoCurrency, r/Bitcoin, etc.)
    for crypto assets and stock subreddits (r/wallstreetbets, r/stocks, etc.)
    for equities.

    Args:
        symbol: Ticker, e.g. 'BTC', 'TSLA'
        hours_back: How far back to look in hours (default 6)
        min_score: Minimum Reddit score (upvotes) to include (default 50)
        limit: Max posts to fetch per subreddit (default 25)

    Returns:
        {
          symbol: str,
          posts_found: int,
          avg_score: float,
          top_posts: list[{title, score, subreddit, created_utc}],
          sentiment_hint: "bullish" | "bearish" | "neutral" | "mixed"
        }
    """
    _, _, ua = _creds()
    token = await _get_token()
    subreddits = _CRYPTO_SUBS if symbol.upper() in _CRYPTO_TICKERS else _EQUITY_SUBS
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).timestamp()

    headers = {"Authorization": f"Bearer {token}", "User-Agent": ua}
    posts: list[dict] = []

    async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
        for sub in subreddits:
            try:
                resp = await client.get(
                    f"{_API_BASE}/r/{sub}/search",
                    params={
                        "q": symbol,
                        "sort": "new",
                        "limit": limit,
                        "t": "day",
                        "restrict_sr": True,
                    },
                )
                resp.raise_for_status()
                for child in resp.json().get("data", {}).get("children", []):
                    d = child.get("data", {})
                    score = d.get("score", 0)
                    created = d.get("created_utc", 0)
                    if score >= min_score and created >= cutoff:
                        title = d.get("title", "").lower()
                        posts.append({
                            "title": d.get("title", ""),
                            "score": score,
                            "subreddit": sub,
                            "created_utc": created,
                            "_bull": any(w in title for w in ["buy", "bull", "moon", "long", "calls", "pump"]),
                            "_bear": any(w in title for w in ["sell", "bear", "short", "puts", "dump", "crash"]),
                        })
            except Exception:
                continue

    if not posts:
        return {
            "symbol": symbol,
            "posts_found": 0,
            "avg_score": 0,
            "top_posts": [],
            "sentiment_hint": "neutral",
        }

    posts.sort(key=lambda p: p["score"], reverse=True)
    avg_score = sum(p["score"] for p in posts) / len(posts)
    bull = sum(1 for p in posts if p["_bull"])
    bear = sum(1 for p in posts if p["_bear"])

    if bull > bear * 1.5:
        hint = "bullish"
    elif bear > bull * 1.5:
        hint = "bearish"
    elif bull > 0 and bear > 0:
        hint = "mixed"
    else:
        hint = "neutral"

    return {
        "symbol": symbol,
        "posts_found": len(posts),
        "avg_score": round(avg_score, 1),
        "top_posts": [
            {"title": p["title"], "score": p["score"], "subreddit": p["subreddit"]}
            for p in posts[:5]
        ],
        "sentiment_hint": hint,
        "bullish_posts": bull,
        "bearish_posts": bear,
    }


if __name__ == "__main__":
    mcp.run()
