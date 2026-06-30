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
            while self._timestamps and now - self._timestamps[0] >= self.period:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.max_calls:
                sleep_for = self.period - (now - self._timestamps[0])
                if sleep_for > 0:
                    logger.debug("Rate limit reached, sleeping %.2fs", sleep_for)
                # Release lock while sleeping so other coroutines can check
                self._timestamps.popleft()
                self._timestamps.appendleft(time.monotonic() - self.period + sleep_for)
            self._timestamps.append(time.monotonic())


class EtoroClient:
    """Async HTTP client for the eToro Public API."""

    def __init__(self):
        self.api_key = os.environ["ETORO_PUBLIC_API_KEY"]
        self.user_key = os.environ["ETORO_USER_KEY"]
        self.mode = os.getenv("ETORO_MODE", "demo").lower()  # demo | real

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
        """Base headers without x-request-id (generated per-request for idempotency)."""
        return {
            "x-api-key": self.api_key,
            "x-user-key": self.user_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _execution_prefix(self) -> str:
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
        # One stable idempotency key for the entire lifetime of this logical request.
        # If the server supports x-request-id deduplication, duplicate retries are
        # recognized and de-duplicated.  This does NOT make retries safe for servers
        # that don't implement it — writes are therefore not retried at all (see below).
        request_id = str(uuid.uuid4())
        is_write = method.upper() in ("POST", "PUT", "PATCH", "DELETE")
        last_exc: Exception = RuntimeError("No attempts made")

        for attempt in range(retries):
            try:
                headers = {**self._headers(), "x-request-id": request_id}
                response = await self._client.request(
                    method, url, headers=headers, json=json, params=params
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    if is_write:
                        # Never blindly retry a write: the server may already have
                        # processed the request. Let the caller decide (reconcile
                        # against get_portfolio() before assuming it failed).
                        response.raise_for_status()
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
                if is_write:
                    raise  # Don't retry; risk of duplicate execution
                wait = 2 ** attempt
                logger.warning("HTTP error — retrying in %ds: %s", wait, exc)
                await asyncio.sleep(wait)
            except httpx.RequestError as exc:
                last_exc = exc
                if is_write:
                    # Network error on write: broker may have processed it.
                    # Caller must reconcile against get_portfolio().
                    raise
                wait = 2 ** attempt
                logger.warning("Request error — retrying in %ds: %s", wait, exc)
                await asyncio.sleep(wait)

        raise last_exc

    # ------------------------------------------------------------------
    # Market data endpoints
    # ------------------------------------------------------------------

    async def get_candles(
        self, symbol: str, interval: str = "M15", count: int = 100
    ) -> list[dict]:
        data = await self._request(
            "GET",
            f"/instruments/{symbol}/candles",
            limiter=self._market_limiter,
            params={"resolution": interval, "count": count},
        )
        candles = data.get("data", data) if isinstance(data, dict) else data
        return candles

    async def get_rates(self, symbols: list[str]) -> dict[str, dict]:
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
        data = await self._request("GET", f"/{self._execution_prefix()}/balance")
        return data.get("data", data) if isinstance(data, dict) else data

    async def get_portfolio(self) -> list[dict]:
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
        """Open a new position.

        NOTE: The `stopLoss` field semantics must be verified against the eToro
        Public API docs before using in `real` mode.  Many broker APIs expect a
        rate (absolute price) in `stopLossRate`, not a percentage.  Validate
        this in demo mode by checking the actual stop placement in the eToro UI
        after opening a test position.
        """
        payload = {
            "instrumentId": instrument_id,
            "amount": amount_usd,
            "isBuy": is_buy,
            "stopLoss": round(stop_loss_pct, 4),  # verify: % vs absolute rate vs $amount
            "trailingStop": trailing_stop,
        }
        data = await self._request(
            "POST",
            f"/{self._execution_prefix()}/positions",
            json=payload,
        )
        return data.get("data", data) if isinstance(data, dict) else data

    async def close_position(self, position_id: str, instrument_id: str) -> dict:
        data = await self._request(
            "DELETE",
            f"/{self._execution_prefix()}/positions/{position_id}",
            params={"instrumentId": instrument_id},
        )
        return data.get("data", data) if isinstance(data, dict) else data

    async def update_stop_loss(
        self, position_id: str, instrument_id: str, new_stop: float
    ) -> dict:
        """Update the stop loss on an open position.

        NOTE: This endpoint and payload format must be verified against the eToro
        Public API docs.  The exact field name and expected unit (%, rate, amount)
        may differ.  Test in demo mode and confirm in the UI before enabling in real.
        """
        payload = {
            "instrumentId": instrument_id,
            "stopLoss": round(new_stop, 4),
        }
        data = await self._request(
            "PUT",
            f"/{self._execution_prefix()}/positions/{position_id}",
            json=payload,
        )
        return data.get("data", data) if isinstance(data, dict) else data
