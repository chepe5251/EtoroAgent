import asyncio
import logging
import os
import time
import uuid
from collections import deque
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_URL = "https://public-api.etoro.com/api/v1"


class RateLimiter:
    """Token-bucket rate limiter tracked per sliding window."""

    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            # purge old timestamps outside the window
            while self._timestamps and now - self._timestamps[0] >= self.period:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.max_calls:
                sleep_for = self.period - (now - self._timestamps[0])
                if sleep_for > 0:
                    logger.debug("Rate limit reached, sleeping %.2fs", sleep_for)
                    await asyncio.sleep(sleep_for)
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self.period:
                    self._timestamps.popleft()
            self._timestamps.append(time.monotonic())


class EtoroClient:
    """Async HTTP client for the eToro Public API."""

    def __init__(self):
        self.api_key = os.environ["ETORO_PUBLIC_API_KEY"]
        self.user_key = os.environ["ETORO_USER_KEY"]
        self.mode = os.getenv("ETORO_MODE", "demo").lower()  # demo | real

        # shared limiter: 60 req/60s; market data has its own 120 req/60s
        self._shared_limiter = RateLimiter(max_calls=60, period=60.0)
        self._market_limiter = RateLimiter(max_calls=120, period=60.0)

        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *_):
        if self._client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "x-user-key": self.user_key,
            "x-request-id": str(uuid.uuid4()),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _execution_prefix(self) -> str:
        """Returns the path segment for execution endpoints based on mode."""
        return "demo" if self.mode == "demo" else "real"

    async def _request(
        self,
        method: str,
        path: str,
        *,
        limiter: Optional[RateLimiter] = None,
        json: Any = None,
        params: Any = None,
        retries: int = 4,
    ) -> Any:
        if limiter is None:
            limiter = self._shared_limiter
        await limiter.acquire()

        url = f"{BASE_URL}{path}"
        last_exc: Exception = RuntimeError("No attempts made")

        for attempt in range(retries):
            try:
                response = await self._client.request(
                    method, url, headers=self._headers(), json=json, params=params
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    wait = 2 ** attempt
                    logger.warning(
                        "HTTP %s from %s — retrying in %ds (attempt %d/%d)",
                        response.status_code, path, wait, attempt + 1, retries,
                    )
                    await asyncio.sleep(wait)
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning("HTTP error %s — retrying in %ds", exc, wait)
                await asyncio.sleep(wait)
            except httpx.RequestError as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning("Request error %s — retrying in %ds", exc, wait)
                await asyncio.sleep(wait)

        raise last_exc

    # ------------------------------------------------------------------
    # Market data endpoints
    # ------------------------------------------------------------------

    async def get_candles(
        self, symbol: str, interval: str = "M15", count: int = 100
    ) -> list[dict]:
        """
        Fetch OHLCV candles for a symbol.
        interval examples: M1, M5, M15, M30, H1, H4, D1
        """
        data = await self._request(
            "GET",
            f"/instruments/{symbol}/candles",
            limiter=self._market_limiter,
            params={"resolution": interval, "count": count},
        )
        # Normalise to list of {time, open, high, low, close, volume}
        candles = data.get("data", data) if isinstance(data, dict) else data
        return candles

    async def get_rates(self, symbols: list[str]) -> dict[str, dict]:
        """Return current bid/ask rates for a list of symbols."""
        data = await self._request(
            "GET",
            "/rates",
            limiter=self._market_limiter,
            params={"instruments": ",".join(symbols)},
        )
        rates = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(rates, list):
            return {item["instrumentName"]: item for item in rates}
        return rates

    async def get_instrument_id(self, symbol: str) -> Optional[str]:
        """Resolve a symbol ticker to eToro instrument ID."""
        data = await self._request(
            "GET",
            "/instruments",
            limiter=self._market_limiter,
            params={"filter": symbol},
        )
        instruments = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(instruments, list) and instruments:
            return str(instruments[0].get("instrumentId", instruments[0].get("id")))
        return None

    # ------------------------------------------------------------------
    # Account endpoints (mode-aware)
    # ------------------------------------------------------------------

    async def get_balance(self) -> dict:
        """Return available balance and equity."""
        data = await self._request("GET", f"/{self._execution_prefix()}/balance")
        return data.get("data", data) if isinstance(data, dict) else data

    async def get_portfolio(self) -> list[dict]:
        """Return current open positions from the account."""
        data = await self._request("GET", f"/{self._execution_prefix()}/portfolio")
        portfolio = data.get("data", data) if isinstance(data, dict) else data
        return portfolio if isinstance(portfolio, list) else []

    # ------------------------------------------------------------------
    # Execution endpoints (mode-aware)
    # ------------------------------------------------------------------

    async def open_position(
        self,
        instrument_id: str,
        amount_usd: float,
        is_buy: bool,
        stop_loss_pct: float,
        trailing_stop: bool = True,
    ) -> dict:
        """Open a new position. Returns the created position dict."""
        payload = {
            "instrumentId": instrument_id,
            "amount": amount_usd,
            "isBuy": is_buy,
            "stopLoss": round(stop_loss_pct, 4),
            "trailingStop": trailing_stop,
        }
        data = await self._request(
            "POST",
            f"/{self._execution_prefix()}/positions",
            json=payload,
        )
        return data.get("data", data) if isinstance(data, dict) else data

    async def close_position(self, position_id: str, instrument_id: str) -> dict:
        """Close an open position."""
        data = await self._request(
            "DELETE",
            f"/{self._execution_prefix()}/positions/{position_id}",
            params={"instrumentId": instrument_id},
        )
        return data.get("data", data) if isinstance(data, dict) else data
