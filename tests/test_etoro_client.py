"""
Tests for EtoroClient using httpx mock transport.
Endpoints/payloads mirror https://api-portal.etoro.com/api-reference/openapi.json
(eToro Api v1.279.0).
Run with: pytest tests/test_etoro_client.py -v
"""
import os
import sys
from pathlib import Path

import pytest
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("ETORO_PUBLIC_API_KEY", "test-api-key")
os.environ.setdefault("ETORO_USER_KEY", "test-user-key")
os.environ.setdefault("ETORO_MODE", "demo")

from src.core.etoro_client import EtoroClient, OrderRejected


# ---------------------------------------------------------------------------
# Mock transport
# ---------------------------------------------------------------------------

class MockTransport(httpx.AsyncBaseTransport):
    """Returns pre-configured responses keyed by (method, path prefix)."""

    def __init__(self, routes: dict):
        self._routes = routes  # {(method, path_prefix): (status, body_dict)}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        for (method, prefix), (status, body) in self._routes.items():
            if request.method == method and path.startswith(prefix):
                return httpx.Response(
                    status,
                    json=body,
                    headers={"Content-Type": "application/json"},
                )
        return httpx.Response(404, json={"errorCode": "RouteNotFound", "errorMessage": "Route not found"})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CANDLES_RESPONSE = {
    "interval": "OneDay",
    "candles": [
        {
            "instrumentId": 42,
            "candles": [
                {"instrumentID": 42, "fromDate": "2024-01-01T00:00:00Z", "open": 50000, "high": 51000, "low": 49500, "close": 50500, "volume": 1200},
                {"instrumentID": 42, "fromDate": "2024-01-02T00:00:00Z", "open": 50500, "high": 52000, "low": 50000, "close": 51800, "volume": 1500},
            ],
        }
    ],
}

RATES_RESPONSE = {
    "rates": [
        {"instrumentID": 42, "bid": 51700, "ask": 51900, "lastExecution": 51800},
    ]
}

INSTRUMENT_RESPONSE = {"instrumentId": 42, "symbol": "BTC", "displayName": "Bitcoin"}

PORTFOLIO_RESPONSE = {"clientPortfolio": {"positions": [], "credit": 10000.0}}

CREATE_ORDER_RESPONSE = {"token": "tok-1", "orderId": 555, "referenceId": "ref-1"}

ORDER_LOOKUP_EXECUTED_RESPONSE = {
    "orderId": 555,
    "status": {"id": 1, "name": "Executed", "errorCode": 0},
    "positionExecutions": [
        {"positionId": 9001, "state": "open", "remainingUnits": 10.5, "openingData": {"avgPrice": 51800.0}}
    ],
}

ORDER_LOOKUP_REJECTED_RESPONSE = {
    "orderId": 556,
    "status": {"id": 3, "name": "Rejected", "errorCode": 42, "errorMessage": "Insufficient funds"},
    "positionExecutions": [],
}

CLOSE_POSITION_RESPONSE = {
    "orderForClose": {"positionID": 9001, "instrumentID": 42, "orderID": 777},
    "token": "tok-2",
}

UPDATE_STOP_RESPONSE = {"operationId": "op-1", "positionId": 9001, "referenceId": "ref-2"}


@pytest.fixture
def mock_client():
    routes = {
        ("GET", "/api/v1/market-data/instruments/42/history/candles"): (200, CANDLES_RESPONSE),
        ("GET", "/api/v1/market-data/instruments/rates"): (200, RATES_RESPONSE),
        ("GET", "/api/v1/instruments/BTC"): (200, INSTRUMENT_RESPONSE),
        ("GET", "/api/v1/trading/info/demo/portfolio"): (200, PORTFOLIO_RESPONSE),
        ("POST", "/api/v2/trading/execution/demo/orders"): (200, CREATE_ORDER_RESPONSE),
        ("GET", "/api/v2/trading/info/demo/orders:lookup"): (200, ORDER_LOOKUP_EXECUTED_RESPONSE),
        ("POST", "/api/v1/trading/execution/demo/market-close-orders/positions/9001"): (200, CLOSE_POSITION_RESPONSE),
        ("PATCH", "/api/v2/trading/demo/positions/9001"): (200, UPDATE_STOP_RESPONSE),
    }
    transport = MockTransport(routes)
    client = EtoroClient()
    client._client = httpx.AsyncClient(transport=transport, timeout=10.0)
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_candles(mock_client):
    from unittest.mock import patch
    with patch("src.core.etoro_client._cache_id", return_value="42"):
        candles = await mock_client.get_candles("BTC", "D1", 2)
    assert len(candles) == 2
    assert candles[0]["close"] == 50500


@pytest.mark.asyncio
async def test_get_rates(mock_client):
    from unittest.mock import patch
    with patch("src.core.etoro_client._cache_id", return_value="42"):
        rates = await mock_client.get_rates(["BTC"])
    assert "BTC" in rates
    assert rates["BTC"]["bid"] == 51700


@pytest.mark.asyncio
async def test_get_instrument_id(mock_client):
    instr_id = await mock_client.get_instrument_id("BTC")
    assert instr_id == "42"


@pytest.mark.asyncio
async def test_get_balance(mock_client):
    balance = await mock_client.get_balance()
    assert balance == 10000.0


@pytest.mark.asyncio
async def test_get_portfolio(mock_client):
    portfolio = await mock_client.get_portfolio()
    assert portfolio == []


@pytest.mark.asyncio
async def test_open_position(mock_client):
    result = await mock_client.open_position(
        instrument_id="42",
        amount_usd=200.0,
        is_buy=True,
        stop_loss_pct=2.0,
        current_price=51800.0,
        trailing_stop=True,
    )
    assert result["positionId"] == "9001"
    assert result["openRate"] == 51800.0


@pytest.mark.asyncio
async def test_open_position_rejected():
    routes = {
        ("POST", "/api/v2/trading/execution/demo/orders"): (200, {"orderId": 556, "referenceId": "r", "token": "t"}),
        ("GET", "/api/v2/trading/info/demo/orders:lookup"): (200, ORDER_LOOKUP_REJECTED_RESPONSE),
    }
    transport = MockTransport(routes)
    client = EtoroClient()
    client._client = httpx.AsyncClient(transport=transport, timeout=10.0)
    with pytest.raises(OrderRejected):
        await client.open_position(
            instrument_id="42",
            amount_usd=200.0,
            is_buy=True,
            stop_loss_pct=2.0,
            current_price=51800.0,
        )


@pytest.mark.asyncio
async def test_close_position(mock_client):
    result = await mock_client.close_position("9001", "42")
    assert result["orderForClose"]["orderID"] == 777


@pytest.mark.asyncio
async def test_update_stop_loss(mock_client):
    result = await mock_client.update_stop_loss("9001", "42", 50000.0)
    assert result["positionId"] == 9001


@pytest.mark.asyncio
async def test_rate_limiter_allows_burst():
    """RateLimiter should allow calls up to max_calls without sleeping."""
    from src.core.etoro_client import RateLimiter
    limiter = RateLimiter(max_calls=5, period=60.0)
    for _ in range(5):
        await limiter.acquire()
    # All 5 acquired without sleeping — if we get here, the test passes


@pytest.mark.asyncio
async def test_is_demo_true_by_default():
    client = EtoroClient()
    assert client.is_demo is True


@pytest.mark.asyncio
async def test_is_demo_false_in_real_mode(monkeypatch):
    monkeypatch.setenv("ETORO_MODE", "real")
    client = EtoroClient()
    assert client.is_demo is False
