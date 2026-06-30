import json
import logging
import os
from typing import Optional

import litellm
from dotenv import load_dotenv

from src.core.state import ProjectState
from src.tools.sentiment import aggregate_sentiment, fetch_news, fetch_reddit_posts

load_dotenv()
logger = logging.getLogger(__name__)

_SCORE_PROMPT = """\
You are a financial sentiment analyst. Score the overall sentiment of the \
following news articles/social media posts about {symbol} on a scale from \
-1.0 (very bearish) to +1.0 (very bullish). Return ONLY a JSON object with \
a single key "score" and a float value.

Posts:
{posts}
"""


class SentimentAgent:
    """Fetches news + Reddit posts and scores sentiment per symbol using an LLM."""

    def __init__(self, state: ProjectState):
        self.state = state
        self.news_api_key = os.getenv("NEWS_API_KEY", "")
        self.reddit_client_id = os.getenv("REDDIT_CLIENT_ID", "")
        self.reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        self.reddit_user_agent = os.getenv("REDDIT_USER_AGENT", "etoroAgent/1.0")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o")
        self.llm_base_url: Optional[str] = os.getenv("LLM_BASE_URL") or None
        self.llm_api_key: Optional[str] = os.getenv("LLM_API_KEY") or None

    async def run(self):
        symbols = list(self.state.market_data.keys())
        if not symbols:
            logger.warning("SentimentAgent: no symbols in state, skipping")
            return

        logger.info("SentimentAgent: scoring sentiment for %s", symbols)
        for symbol in symbols:
            try:
                news_score = await self._score_news(symbol)
                reddit_score = await self._score_reddit(symbol)
                final = aggregate_sentiment(news_score, reddit_score)
                self.state.sentiment[symbol] = final
                logger.info(
                    "SentimentAgent: %s — news=%.2f reddit=%.2f aggregated=%.2f",
                    symbol,
                    news_score if news_score is not None else float("nan"),
                    reddit_score if reddit_score is not None else float("nan"),
                    final,
                )
            except Exception as exc:
                logger.error("SentimentAgent error for %s: %s", symbol, exc, exc_info=True)
                self.state.sentiment.setdefault(symbol, 0.0)

    async def _score_news(self, symbol: str) -> Optional[float]:
        articles = await fetch_news(symbol, self.news_api_key)
        if not articles:
            return None
        posts_text = "\n".join(
            f"- [{a['source']}] {a['title']}: {a['description']}" for a in articles
        )
        return await self._llm_score(symbol, posts_text)

    async def _score_reddit(self, symbol: str) -> Optional[float]:
        posts = await fetch_reddit_posts(
            symbol,
            self.reddit_client_id,
            self.reddit_client_secret,
            self.reddit_user_agent,
        )
        if not posts:
            return None
        posts_text = "\n".join(
            f"- [r/{p['subreddit']} score={p['score']}] {p['title']} {p['selftext']}"
            for p in posts
        )
        return await self._llm_score(symbol, posts_text)

    async def _llm_score(self, symbol: str, posts_text: str) -> Optional[float]:
        prompt = _SCORE_PROMPT.format(symbol=symbol, posts=posts_text[:3000])
        kwargs: dict = {"model": self.llm_model, "messages": [{"role": "user", "content": prompt}]}
        if self.llm_base_url:
            kwargs["base_url"] = self.llm_base_url
        if self.llm_api_key:
            kwargs["api_key"] = self.llm_api_key
        try:
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content.strip()
            # strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content)
            score = float(parsed.get("score", 0.0))
            return max(-1.0, min(1.0, score))
        except Exception as exc:
            logger.warning("LLM sentiment scoring failed for %s: %s", symbol, exc)
            return None
