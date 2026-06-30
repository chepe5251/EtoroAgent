"""
Sentiment aggregation helpers.
The actual LLM scoring is done inside SentimentAgent;
these helpers handle fetching and pre-processing.
"""
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

NEWS_API_BASE = "https://newsapi.org/v2"
REDDIT_BASE = "https://oauth.reddit.com"
REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"

CRYPTO_SUBREDDITS = ["CryptoCurrency", "CryptoMarkets"]
STOCK_SUBREDDITS = ["wallstreetbets", "stocks", "investing"]


async def fetch_news(
    symbol: str,
    api_key: str,
    *,
    page_size: int = 10,
    hours_back: int = 6,
) -> list[dict]:
    """Fetch recent news headlines for a symbol from NewsAPI."""
    if not api_key:
        return []
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            resp = await client.get(
                f"{NEWS_API_BASE}/everything",
                params={
                    "q": symbol,
                    "sortBy": "publishedAt",
                    "pageSize": page_size,
                    "language": "en",
                    "apiKey": api_key,
                },
            )
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            cutoff = datetime.now(timezone.utc).timestamp() - hours_back * 3600
            result = []
            for a in articles:
                published = a.get("publishedAt", "")
                try:
                    ts = datetime.fromisoformat(
                        published.replace("Z", "+00:00")
                    ).timestamp()
                except ValueError:
                    ts = 0
                if ts >= cutoff:
                    result.append(
                        {
                            "title": a.get("title", ""),
                            "description": a.get("description", ""),
                            "source": a.get("source", {}).get("name", ""),
                        }
                    )
            return result
        except Exception as exc:
            logger.warning("NewsAPI fetch failed for %s: %s", symbol, exc)
            return []


async def _get_reddit_token(client_id: str, client_secret: str, user_agent: str) -> Optional[str]:
    """Obtain an OAuth2 bearer token from Reddit."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.post(
                REDDIT_TOKEN_URL,
                data={"grant_type": "client_credentials"},
                auth=(client_id, client_secret),
                headers={"User-Agent": user_agent},
            )
            resp.raise_for_status()
            return resp.json().get("access_token")
        except Exception as exc:
            logger.warning("Reddit token error: %s", exc)
            return None


async def fetch_reddit_posts(
    symbol: str,
    client_id: str,
    client_secret: str,
    user_agent: str,
    *,
    hours_back: int = 6,
    min_score: int = 50,
    limit: int = 25,
) -> list[dict]:
    """Fetch recent high-score Reddit posts mentioning a symbol."""
    if not client_id or not client_secret:
        return []

    token = await _get_reddit_token(client_id, client_secret, user_agent)
    if not token:
        return []

    # pick subreddits based on whether symbol looks like crypto or stock
    crypto_tickers = {"BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE"}
    if symbol.upper() in crypto_tickers:
        subreddits = CRYPTO_SUBREDDITS
    else:
        subreddits = STOCK_SUBREDDITS

    cutoff = datetime.now(timezone.utc).timestamp() - hours_back * 3600
    posts: list[dict] = []

    async with httpx.AsyncClient(
        timeout=20.0,
        headers={"Authorization": f"Bearer {token}", "User-Agent": user_agent},
    ) as client:
        for sub in subreddits:
            try:
                resp = await client.get(
                    f"{REDDIT_BASE}/r/{sub}/search",
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
                    if d.get("score", 0) >= min_score and d.get("created_utc", 0) >= cutoff:
                        posts.append(
                            {
                                "title": d.get("title", ""),
                                "selftext": (d.get("selftext", "") or "")[:300],
                                "score": d.get("score", 0),
                                "subreddit": sub,
                            }
                        )
            except Exception as exc:
                logger.warning("Reddit fetch failed for r/%s: %s", sub, exc)

    return posts


def aggregate_sentiment(
    news_score: Optional[float],
    reddit_score: Optional[float],
    news_weight: float = 0.4,
    reddit_weight: float = 0.6,
) -> float:
    """Weighted average of news and reddit sentiment scores."""
    if news_score is None and reddit_score is None:
        return 0.0
    if news_score is None:
        return reddit_score
    if reddit_score is None:
        return news_score
    return news_score * news_weight + reddit_score * reddit_weight
