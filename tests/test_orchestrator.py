"""
Tests for Orchestrator's scan/execute split (pre-market scan, at-open execution).
Constructs a bare Orchestrator via __new__ to avoid the real EtoroClient/MCPManager
wiring in __init__ — only the attributes each method actually touches are set.
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.orchestrator import Orchestrator
from src.agents.screening_agent import ScreeningResult
from src.core.state import ProjectState


def _make_orchestrator(state: ProjectState) -> Orchestrator:
    orch = Orchestrator.__new__(Orchestrator)
    orch.state = state
    orch.screening_agent = MagicMock()
    orch.notification_agent = MagicMock()
    orch.notification_agent.send_critical_error = AsyncMock()
    orch._instrument_cache = {"AAPL": "1001"}
    return orch


@pytest.mark.asyncio
async def test_scan_region_stores_pending_signals():
    state = ProjectState()
    orch = _make_orchestrator(state)
    result = ScreeningResult(symbol="AAPL", breakout=True, score_tags=["breakout"])
    orch.screening_agent.run = AsyncMock(return_value=[result])

    with patch("src.core.orchestrator.is_trading_day", return_value=True), \
         patch("src.core.orchestrator.get_symbols", return_value=["AAPL"]):
        await orch._scan_region("US")

    assert "US" in state.pending_signals
    assert state.pending_signals["US"][0]["symbol"] == "AAPL"
    assert state.pending_signals["US"][0]["breakout"] is True


@pytest.mark.asyncio
async def test_scan_region_skipped_on_non_trading_day():
    state = ProjectState()
    orch = _make_orchestrator(state)
    orch.screening_agent.run = AsyncMock(return_value=[])

    with patch("src.core.orchestrator.is_trading_day", return_value=False):
        await orch._scan_region("US")

    orch.screening_agent.run.assert_not_called()
    assert "US" not in state.pending_signals


@pytest.mark.asyncio
async def test_execute_region_consumes_pending_signals():
    state = ProjectState()
    state.pending_signals["US"] = [
        {"symbol": "AAPL", "rsi": None, "ema20": None, "ema50": None, "ema200": None,
         "atr": None, "rel_volume": 2.0, "trend_up": True, "breakout": True,
         "pullback_resume": False, "score_tags": ["breakout"]}
    ]
    orch = _make_orchestrator(state)
    orch._get_balance = AsyncMock(return_value=1000.0)
    orch._unrealized_pnl = MagicMock(return_value=0.0)
    orch._build_and_execute = AsyncMock()

    with patch("src.core.orchestrator.is_trading_day", return_value=True):
        await orch._execute_region("US")

    orch._build_and_execute.assert_called_once()
    called_result = orch._build_and_execute.call_args.args[0]
    assert isinstance(called_result, ScreeningResult)
    assert called_result.symbol == "AAPL"
    # Pending signals must be cleared after execution, win or lose.
    assert "US" not in state.pending_signals


@pytest.mark.asyncio
async def test_execute_region_noop_when_nothing_pending():
    state = ProjectState()
    orch = _make_orchestrator(state)
    orch._get_balance = AsyncMock(return_value=1000.0)
    orch._build_and_execute = AsyncMock()

    await orch._execute_region("US")

    orch._build_and_execute.assert_not_called()


@pytest.mark.asyncio
async def test_execute_region_discards_pending_on_non_trading_day():
    """Guards against a scan on Friday evening + execute rescheduled onto a holiday."""
    state = ProjectState()
    state.pending_signals["US"] = [
        {"symbol": "AAPL", "rsi": None, "ema20": None, "ema50": None, "ema200": None,
         "atr": None, "rel_volume": None, "trend_up": True, "breakout": True,
         "pullback_resume": False, "score_tags": ["breakout"]}
    ]
    orch = _make_orchestrator(state)
    orch._build_and_execute = AsyncMock()

    with patch("src.core.orchestrator.is_trading_day", return_value=False):
        await orch._execute_region("US")

    orch._build_and_execute.assert_not_called()


def test_project_state_persists_pending_signals(tmp_path):
    state = ProjectState()
    state.pending_signals["EU"] = [{"symbol": "SAP", "breakout": True}]
    path = tmp_path / "state.json"
    state.save(path)

    reloaded = ProjectState.load(path)
    assert reloaded.pending_signals == {"EU": [{"symbol": "SAP", "breakout": True}]}
