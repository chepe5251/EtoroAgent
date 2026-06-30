"""
Tests for EtoroClient using httpx mock transport.
Run with: pytest tests/test_etoro_client.py -v
"""
import json
import os
import sys
from pathlib import Path

import pytest
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("ETORO_PUBLIC_API_KEY", "test-api-key")
os.environ.setdefault("ETORO_USER_KEY", "test-user-key")
os.environ.setdefault("ETORO_MODE", "demo")

from src.core.etoro_client import EtoroClient


# ---------------------------------------------------------------------------
# Mock transport
# ---------------------------------------------------------------------------

class MockTransport(httpx.AsyncBaseTransport):
    """Returns pre-configured responses keyed by (method, path prefix)."""

    def __init__(self, routes: dict):
        self._routes = routes  # {"/path_prefix": (status, body_dict)}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        for prefix, (status, body) in self._routes.items():
            if path.startswith(prefix):
                return httpx.Response(
                    status,
                    json=body,
                    headers={"Content-Type": "application/json"},
                )
        return httpx.Response(404, json={"error": "not found"})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CANDLES_RESPONSE = {
    "data": [
        {"open": 50000, "high": 51000, "low": 49500, "close": 50500, "volume": 1200, "time": "2024-01-01T00:00:00Z"},
        {"open": 50500, "high": 52000, "low": 50000, "close": 51800, "volume": 1500, "time": "2024-01-01T00:15:00Z"},
    ]
}

RATES_RESPONSE = {
    "data": [
        {"instrumentName": "BTC", "bid": 51700, "ask": 51900, "close": 51800},
    ]
}

INSTRUMENT_RESPONSE = {
    "data": [{"instrumentId": "42", "instrumentName": "BTC"}]
}

BALANCE_RESPONSE = {
    "data": {"availableToTrade": 10000.0, "equity": 12000.0}
}

PORTFOLIO_RESPONSE = {
    "data": []
}

OPEN_POSITION_RESPONSE = {
    "data": {"positionId": "pos-001", "rate": 51800.0, "amount": 200.0}
}

CLOSE_POSITION_RESPONSE = {
    "data": {"positionId": "pos-001", "closeRate": 52100.0}
}


@pytest.fixture
def mock_client():
    routes = {
        "/api/v1/instruments/BTC/candles": (200, CANDLES_RESPONSE),
        "/api/v1/rates": (200, RATES_RESPONSE),
        "/api/v1/instruments": (200, INSTRUMENT_RESPONSE),
        "/api/v1/demo/balance": (200, BALANCE_RESPONSE),
        "/api/v1/demo/portfolio": (200, PORTFOLIO_RESPONSE),
        "/api/v1/demo/positions": (200, OPEN_POSITION_RESPONSE),
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
    candles = await mock_client.get_candles("BTC", "M15", 2)
    assert len(candles) == 2
    assert candles[0]["close"] == 50500


@pytest.mark.asyncio
async def test_get_rates(mock_client):
    rates = await mock_client.get_rates(["BTC"])
    assert "BTC" in rates
    assert rates["BTC"]["close"] == 51800


@pytest.mark.asyncio
async def test_get_instrument_id(mock_client):
    instr_id = await mock_client.get_instrument_id("BTC")
    assert instr_id == "42"


@pytest.mark.asyncio
async def test_get_balance(mock_client):
    balance = await mock_client.get_balance()
    assert balance["availableToTrade"] == 10000.0


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
        trailing_stop=True,
    )
    assert result["positionId"] == "pos-001"
    assert result["rate"] == 51800.0


@pytest.mark.asyncio
async def test_rate_limiter_allows_burst():
    """RateLimiter should allow calls up to max_calls without sleeping."""
    from src.core.etoro_client import RateLimiter
    limiter = RateLimiter(max_calls=5, period=60.0)
    import asyncio
    for _ in range(5):
        await limiter.acquire()
    # All 5 acquired without sleeping — if we get here, the test passes


@pytest.mark.asyncio
async def test_execution_prefix_demo():
    client = EtoroClient()
    assert client._execution_prefix() == "demo"


@pytest.mark.asyncio
async def test_execution_prefix_real(monkeypatch):
    monkeypatch.setenv("ETORO_MODE", "real")
    client = EtoroClient()
    assert client._execution_prefix() == "real"
