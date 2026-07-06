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

BASE_URL = "https://public-api.etoro.com"

# Module-level alias so tests can patch via 'src.core.etoro_client._cache_id'
from src.config.universe import get_instrument_id as _cache_id  # noqa: E402

# ── Interval translation table ─────────────────────────────────────────────────
# Confirmed against https://api-portal.etoro.com/api-reference/openapi.json
# (candlesResponse endpoint `interval` enum).
_INTERVAL_MAP: dict[str, str] = {
    "D1":  "OneDay",
    "W1":  "OneWeek",
    "H1":  "OneHour",
    "H4":  "FourHours",
    "M60": "OneHour",
    "M15": "FifteenMinutes",
    "M5":  "FiveMinutes",
    "M1":  "OneMinute",
    "OneDay":          "OneDay",
    "OneWeek":         "OneWeek",
    "OneHour":         "OneHour",
    "FourHours":       "FourHours",
    "FifteenMinutes":  "FifteenMinutes",
    "FiveMinutes":     "FiveMinutes",
    "OneMinute":       "OneMinute",
}


class RateLimiter:
    """Token-bucket rate limiter tracked per sliding window."""

    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self):
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self.period:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(time.monotonic())
                    return
                sleep_for = self.period - (now - self._timestamps[0])
            if sleep_for > 0:
                logger.debug("Rate limit reached, sleeping %.2fs", sleep_for)
                await asyncio.sleep(sleep_for)


class OrderRejected(RuntimeError):
    """Raised when eToro rejects or cancels an order (non-zero errorCode)."""


class EtoroClient:
    """Async HTTP client for the eToro Public API.

    Endpoints verified against https://api-portal.etoro.com/api-reference/openapi.json
    (eToro Api v1.279.0). Demo vs. real is NOT a URL prefix uniformly — each
    endpoint family has its own convention (some insert "demo" mid-path, some
    have no distinct demo path at all). See per-method comments.
    """

    def __init__(self):
        self.api_key = os.environ["ETORO_PUBLIC_API_KEY"]
        self.user_key = os.environ["ETORO_USER_KEY"]
        self.mode = os.getenv("ETORO_MODE", "demo").lower()  # demo | real
        self.is_demo = self.mode != "real"

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
        self,
        symbol: str,
        interval: str = "D1",
        count: int = 100,
        direction: str = "asc",
    ) -> list[dict]:
        """
        Fetch OHLCV candles for a symbol.

        Endpoint: GET /api/v1/market-data/instruments/{instrumentId}/history/candles/{direction}/{interval}/{candlesCount}
        (path parameters, not query — candlesCount is capped at 1000 server-side).

        The `symbol` string is resolved to a numeric instrumentId via the universe
        cache first; if not cached, a live API call is made.
        Returns an empty list (and logs a warning) if instrumentId cannot be resolved.
        """
        etoro_interval = _INTERVAL_MAP.get(interval, interval)

        instrument_id: Optional[str] = _cache_id(symbol)
        if instrument_id is None:
            instrument_id = await self.get_instrument_id(symbol)
        if instrument_id is None:
            logger.warning(
                "get_candles: cannot resolve instrumentId for %s — returning []", symbol
            )
            return []

        candles_count = min(count, 1000)
        data = await self._request(
            "GET",
            f"/api/v1/market-data/instruments/{instrument_id}/history/candles/"
            f"{direction}/{etoro_interval}/{candles_count}",
            limiter=self._market_limiter,
        )

        # Response shape: {"interval": ..., "candles": [{"instrumentId": ..., "candles": [...]}]}
        groups = data.get("candles", []) if isinstance(data, dict) else []
        if not groups:
            logger.debug("get_candles: %s returned no candle groups", symbol)
            return []
        candles = groups[0].get("candles", [])
        logger.debug("get_candles: %s returned %d raw candles", symbol, len(candles))
        return candles

    async def get_rates(self, symbols: list[str]) -> dict[str, dict]:
        """Fetch live rates. Endpoint: GET /api/v1/market-data/instruments/rates
        (takes numeric instrumentIds, not symbols — resolved via the universe cache)."""
        id_by_symbol: dict[str, str] = {}
        for symbol in symbols:
            instrument_id = _cache_id(symbol)
            if instrument_id is None:
                instrument_id = await self.get_instrument_id(symbol)
            if instrument_id is not None:
                id_by_symbol[symbol] = instrument_id

        if not id_by_symbol:
            return {}

        data = await self._request(
            "GET",
            "/api/v1/market-data/instruments/rates",
            limiter=self._market_limiter,
            params={"instrumentIds": list(id_by_symbol.values())},
        )
        rates = data.get("rates", []) if isinstance(data, dict) else []
        rate_by_id = {str(item["instrumentID"]): item for item in rates}
        return {
            symbol: rate_by_id[instrument_id]
            for symbol, instrument_id in id_by_symbol.items()
            if instrument_id in rate_by_id
        }

    async def get_instrument_id(self, symbol: str) -> Optional[str]:
        """Endpoint: GET /api/v1/instruments/{symbol} — exact-symbol lookup."""
        try:
            data = await self._request(
                "GET",
                f"/api/v1/instruments/{symbol}",
                limiter=self._market_limiter,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        if isinstance(data, dict) and "instrumentId" in data:
            return str(data["instrumentId"])
        return None

    # ------------------------------------------------------------------
    # Account endpoints
    # ------------------------------------------------------------------

    async def _get_client_portfolio(self) -> dict:
        """Endpoint: GET /api/v1/trading/info/portfolio (real) or
        GET /api/v1/trading/info/demo/portfolio (demo). Shared by get_balance()
        and get_portfolio() since both fields live in the same response."""
        path = (
            "/api/v1/trading/info/demo/portfolio"
            if self.is_demo
            else "/api/v1/trading/info/portfolio"
        )
        data = await self._request("GET", path)
        return data.get("clientPortfolio", {}) if isinstance(data, dict) else {}

    async def get_balance(self) -> float:
        """Tradable USD balance.

        NOTE: GET /api/v1/balances requires the 'etoro-public:money.balance:read'
        scope, which is not grantable through the standard API Key Management UI
        (confirmed 403 InsufficientPermissions even with a fresh Read+Write real
        key). The portfolio endpoint's `credit` field ("Available trading balance
        in USD") is accessible with normal trade scopes and is used instead.
        """
        client_portfolio = await self._get_client_portfolio()
        return float(client_portfolio.get("credit", 0))

    async def get_portfolio(self) -> list[dict]:
        """Returns positions normalised to {positionId, instrumentId, isBuy, openRate,
        stopLossRate, takeProfitRate, amount, leverage} — lower-camel keys, matching
        what the rest of the codebase expects."""
        client_portfolio = await self._get_client_portfolio()
        positions = client_portfolio.get("positions", [])
        return [
            {
                "positionId": str(p.get("positionID")),
                "instrumentId": str(p.get("instrumentID")),
                "isBuy": p.get("isBuy"),
                "openRate": p.get("openRate"),
                "currentRate": p.get("currentRate") or p.get("rate"),
                "stopLossRate": p.get("stopLossRate"),
                "takeProfitRate": p.get("takeProfitRate"),
                "amount": p.get("amount"),
                "leverage": p.get("leverage"),
            }
            for p in positions
        ]

    # ------------------------------------------------------------------
    # Execution endpoints
    # ------------------------------------------------------------------

    async def open_position(
        self,
        instrument_id: str,
        amount_usd: float,
        is_buy: bool,
        stop_loss_pct: float,
        current_price: float,
        trailing_stop: bool = True,
        leverage: int = 1,
    ) -> dict:
        """Open a new position via the unified order endpoint (POST
        /api/v2/trading/execution/orders, or .../demo/orders in demo mode).

        That endpoint is asynchronous: it only returns an orderId, not the fill.
        We poll GET .../orders:lookup?orderId=... until the order reaches a
        terminal state, then read the resulting position's id and fill price.

        stop_loss_pct is a % distance from current_price — converted here to the
        absolute stopLossRate the API requires (there is no percentage field).

        amount_usd is the MARGIN committed from balance (not notional exposure) —
        real market exposure = amount_usd × leverage. Callers sizing by notional
        must pass amount_usd = notional / leverage.
        """
        if is_buy:
            stop_loss_rate = current_price * (1 - stop_loss_pct / 100)
        else:
            stop_loss_rate = current_price * (1 + stop_loss_pct / 100)

        payload = {
            "action": "open",
            "transaction": "buy" if is_buy else "sellShort",
            "instrumentId": int(instrument_id),
            "orderType": "mkt",
            "leverage": leverage,
            "amount": amount_usd,
            "orderCurrency": "usd",
            "stopLossRate": round(stop_loss_rate, 6),
            "stopLossType": "trailing" if trailing_stop else "fixed",
        }
        create_path = (
            "/api/v2/trading/execution/demo/orders"
            if self.is_demo
            else "/api/v2/trading/execution/orders"
        )
        created = await self._request("POST", create_path, json=payload)
        order_id = created.get("orderId") if isinstance(created, dict) else None
        if order_id is None:
            raise RuntimeError(f"open_position: no orderId in response: {created}")

        info = await self._await_order_fill(order_id)
        executions = info.get("positionExecutions") or []
        if not executions:
            raise OrderRejected(f"open_position: order {order_id} has no position executions: {info}")
        execution = executions[0]
        return {
            "positionId": str(execution.get("positionId")),
            "openRate": execution.get("openingData", {}).get("avgPrice", 0),
        }

    async def _await_order_fill(
        self, order_id: int, attempts: int = 10, delay: float = 1.0
    ) -> dict:
        """Poll orders:lookup until the order reaches a terminal status.

        NOTE: the documented status.id mapping (1=Executed, 2=Cancelled,
        3=Rejected) does NOT match production — a live order confirmed
        status.id=3 with status.name="Filled" (a successful fill). Key off
        the human-readable status.name instead, which is unambiguous.
        """
        lookup_path = (
            "/api/v2/trading/info/demo/orders:lookup"
            if self.is_demo
            else "/api/v2/trading/info/orders:lookup"
        )
        _SUCCESS_NAMES = {"filled", "executed"}
        _FAILURE_NAMES = {"cancelled", "canceled", "rejected", "failed"}
        last_info: dict = {}
        for _ in range(attempts):
            last_info = await self._request(
                "GET", lookup_path, params={"orderId": order_id}
            )
            status = last_info.get("status", {}) if isinstance(last_info, dict) else {}
            status_name = str(status.get("name", "")).lower()
            if status_name in _SUCCESS_NAMES:
                return last_info
            if status_name in _FAILURE_NAMES:
                raise OrderRejected(
                    f"order {order_id} {status.get('name')}: {status.get('errorMessage')}"
                )
            await asyncio.sleep(delay)
        raise TimeoutError(f"order {order_id} did not reach a terminal state in time: {last_info}")

    async def close_position(self, position_id: str, instrument_id: str) -> dict:
        """Endpoint: POST /api/v1/trading/execution/market-close-orders/positions/{positionId}
        (or .../demo/market-close-orders/... in demo mode). Omitting UnitsToDeduct
        closes the entire position."""
        path = (
            f"/api/v1/trading/execution/demo/market-close-orders/positions/{position_id}"
            if self.is_demo
            else f"/api/v1/trading/execution/market-close-orders/positions/{position_id}"
        )
        data = await self._request(
            "POST",
            path,
            json={"InstrumentId": int(instrument_id)},
        )
        return data if isinstance(data, dict) else {}

    async def update_stop_loss(
        self, position_id: str, instrument_id: str, new_stop_rate: float
    ) -> dict:
        """Update the stop loss on an open position.

        Endpoint: PATCH /api/v2/trading/positions/{positionId}
        (or /api/v2/trading/demo/positions/{positionId} in demo mode).
        new_stop_rate must be an absolute price (the API has no percentage field).
        """
        path = (
            f"/api/v2/trading/demo/positions/{position_id}"
            if self.is_demo
            else f"/api/v2/trading/positions/{position_id}"
        )
        data = await self._request(
            "PATCH",
            path,
            json={"stopLossRate": round(new_stop_rate, 6), "stopLossType": "fixed"},
        )
        return data if isinstance(data, dict) else {}
