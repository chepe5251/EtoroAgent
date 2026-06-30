"""
Tests for ReActRuntime using a mock OpenAI client and mock MCPManager.
Verifies loop behaviour, tool call dispatch, max_iterations cutoff,
and thesis JSON parsing.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Minimal stubs ────────────────────────────────────────────────────────────

def _make_choice(content: str | None, tool_calls: list | None, finish_reason: str):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


def _make_tool_call(id_: str, name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=id_,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments),
        ),
    )


class MockMCPManager:
    def __init__(self, tool_results: dict[str, Any] | None = None):
        self._tool_results = tool_results or {}

    async def list_openai_tools(self) -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Mock tool {name}",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in self._tool_results
        ]

    async def call_tool(self, name: str, arguments: dict) -> Any:
        if name in self._tool_results:
            return self._tool_results[name]
        raise ValueError(f"Mock: unknown tool {name}")


# ─── Tests ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_immediate_final_answer():
    """LLM returns a final answer on the first call (no tool calls)."""
    thesis_json = json.dumps({
        "symbol": "BTC",
        "action": "buy",
        "confidence": 0.78,
        "reasoning": "RSI oversold and CryptoPanic bullish.",
        "signals_used": ["indicators_full_analysis", "cryptopanic_get_news"],
        "suggested_stop_loss_atr_multiple": 1.5,
    })

    from src.llm.react_runtime import ReActRuntime
    manager = MockMCPManager()
    runtime = ReActRuntime(mcp_manager=manager, system_prompt="test", model="test-model")

    mock_response = _make_choice(thesis_json, None, "stop")
    with patch.object(runtime._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)):
        result = await runtime.run("Analyze BTC")

    assert result["error"] is None
    assert result["thesis"] is not None
    assert result["thesis"]["symbol"] == "BTC"
    assert result["thesis"]["action"] == "buy"
    assert result["trace"] == []  # no tool calls made


@pytest.mark.asyncio
async def test_one_tool_call_then_final_answer():
    """LLM makes 1 tool call, gets result, then produces final thesis."""
    thesis_json = json.dumps({
        "symbol": "ETH",
        "action": "sell",
        "confidence": 0.70,
        "reasoning": "EMA death cross confirmed. Sentiment neutral-to-negative.",
        "signals_used": ["indicators_full_analysis", "reddit_get_subreddit_sentiment"],
        "suggested_stop_loss_atr_multiple": 1.5,
    })
    tool_call = _make_tool_call("call_001", "indicators_full_analysis", {"symbol": "ETH"})
    indicator_result = {"symbol": "ETH", "rsi_14": 72, "ema_20": 1900, "ema_50": 2100}

    responses = [
        _make_choice(None, [tool_call], "tool_calls"),   # first call: tool
        _make_choice(thesis_json, None, "stop"),          # second call: answer
    ]

    from src.llm.react_runtime import ReActRuntime
    manager = MockMCPManager({"indicators_full_analysis": indicator_result})
    runtime = ReActRuntime(mcp_manager=manager, system_prompt="test", model="test-model")

    call_iter = iter(responses)
    with patch.object(
        runtime._client.chat.completions, "create",
        new=AsyncMock(side_effect=lambda **kw: next(call_iter))
    ):
        result = await runtime.run("Analyze ETH")

    assert result["error"] is None
    assert result["thesis"]["symbol"] == "ETH"
    assert len(result["trace"]) == 1
    assert result["trace"][0]["tool"] == "indicators_full_analysis"


@pytest.mark.asyncio
async def test_max_iterations_cutoff():
    """Loop stops at max_iterations even if LLM keeps making tool calls."""
    tool_call = _make_tool_call("call_loop", "indicators_full_analysis", {"symbol": "BTC"})

    from src.llm.react_runtime import ReActRuntime
    manager = MockMCPManager({"indicators_full_analysis": {"rsi_14": 50}})
    runtime = ReActRuntime(
        mcp_manager=manager, system_prompt="test", model="test-model", max_iterations=3
    )

    # Always return a tool_call (never a final answer)
    infinite_response = _make_choice(None, [tool_call], "tool_calls")
    with patch.object(
        runtime._client.chat.completions, "create",
        new=AsyncMock(return_value=infinite_response)
    ):
        result = await runtime.run("Analyze BTC")

    assert result["thesis"] is None
    assert result["error"] is not None
    assert "max_iterations" in result["error"]
    assert len(result["trace"]) == 3  # exactly max_iterations tool calls


@pytest.mark.asyncio
async def test_multiple_tool_calls_in_one_turn():
    """LLM requests 2 tools simultaneously, both get executed."""
    tc1 = _make_tool_call("c1", "indicators_full_analysis", {"symbol": "BTC"})
    tc2 = _make_tool_call("c2", "cryptopanic_get_news", {"currencies": ["BTC"]})
    thesis_json = json.dumps({
        "symbol": "BTC", "action": "buy", "confidence": 0.80,
        "reasoning": "Technical and sentiment both bullish. RSI=28, CryptoPanic bullish_count=15.",
        "signals_used": ["indicators_full_analysis", "cryptopanic_get_news"],
        "suggested_stop_loss_atr_multiple": 1.5,
    })

    from src.llm.react_runtime import ReActRuntime
    manager = MockMCPManager({
        "indicators_full_analysis": {"rsi_14": 28},
        "cryptopanic_get_news": [{"title": "BTC rallying", "positive_votes": 200}],
    })
    runtime = ReActRuntime(mcp_manager=manager, system_prompt="test", model="test-model")

    responses = [
        _make_choice(None, [tc1, tc2], "tool_calls"),
        _make_choice(thesis_json, None, "stop"),
    ]
    call_iter = iter(responses)
    with patch.object(
        runtime._client.chat.completions, "create",
        new=AsyncMock(side_effect=lambda **kw: next(call_iter))
    ):
        result = await runtime.run("Analyze BTC")

    assert result["thesis"]["confidence"] == 0.80
    assert len(result["trace"]) == 2  # both tools called


@pytest.mark.asyncio
async def test_failed_tool_call_does_not_crash():
    """If a tool raises an exception, the error is logged and the loop continues."""
    tool_call = _make_tool_call("c1", "broken_tool", {})
    thesis_json = json.dumps({
        "symbol": "BTC", "action": "hold", "confidence": 0.40,
        "reasoning": "Tool call failed, not enough data to form a thesis.",
        "signals_used": [],
        "suggested_stop_loss_atr_multiple": 1.5,
    })

    from src.llm.react_runtime import ReActRuntime
    manager = MockMCPManager()  # no tools registered → will raise ValueError
    runtime = ReActRuntime(mcp_manager=manager, system_prompt="test", model="test-model")

    responses = [
        _make_choice(None, [tool_call], "tool_calls"),
        _make_choice(thesis_json, None, "stop"),
    ]
    call_iter = iter(responses)
    with patch.object(
        runtime._client.chat.completions, "create",
        new=AsyncMock(side_effect=lambda **kw: next(call_iter))
    ):
        result = await runtime.run("Analyze BTC")

    # Should not raise; thesis is still parsed from the second response
    assert result["thesis"]["action"] == "hold"


@pytest.mark.asyncio
async def test_parse_thesis_from_markdown_fence():
    """ReActRuntime correctly strips ```json fences from the LLM response."""
    fenced = "```json\n{\"symbol\": \"AAPL\", \"action\": \"sell\", \"confidence\": 0.71, \"reasoning\": \"EMA death cross confirmed by news flow.\", \"signals_used\": [\"a\", \"b\"], \"suggested_stop_loss_atr_multiple\": 1.5}\n```"

    from src.llm.react_runtime import ReActRuntime
    manager = MockMCPManager()
    runtime = ReActRuntime(mcp_manager=manager, system_prompt="test", model="test-model")

    parsed = runtime._parse_thesis(fenced)
    assert parsed is not None
    assert parsed["symbol"] == "AAPL"
    assert parsed["action"] == "sell"


@pytest.mark.asyncio
async def test_deepseek_native_format_parsed():
    """
    DeepSeek emits tool calls as plain text with special tokens instead of
    the OpenAI tool_calls field.  The runtime should detect the format,
    parse the tool name + args, execute the real tool, and continue.
    """
    ds_content = (
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>indicators_full_analysis\n"
        "```json\n{\"symbol\": \"BTC\"}\n```"
        "<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
        # DeepSeek hallucinated output — runtime must ignore this
        "<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>"
        "{\"rsi_14\": 50}"
        "<｜tool▁output▁end｜><｜tool▁outputs▁end｜>"
        " The RSI is 50."
    )
    thesis_json = json.dumps({
        "symbol": "BTC", "action": "hold", "confidence": 0.45,
        "reasoning": "RSI is neutral at 50. Sentiment not checked. Insufficient conviction.",
        "signals_used": ["indicators_full_analysis"],
        "suggested_stop_loss_atr_multiple": 1.5,
    })

    from src.llm.react_runtime import ReActRuntime
    manager = MockMCPManager({"indicators_full_analysis": {"rsi_14": 28, "atr_14": 500}})
    runtime = ReActRuntime(mcp_manager=manager, system_prompt="test", model="test-model")

    responses = [
        # First response: DeepSeek native tool call format, finish_reason=stop
        _make_choice(ds_content, None, "stop"),
        # Second response: final thesis after seeing REAL tool data
        _make_choice(thesis_json, None, "stop"),
    ]
    call_iter = iter(responses)
    with patch.object(
        runtime._client.chat.completions, "create",
        new=AsyncMock(side_effect=lambda **kw: next(call_iter))
    ):
        result = await runtime.run("Analyze BTC")

    assert result["error"] is None
    assert result["thesis"]["symbol"] == "BTC"
    # Trace should show the REAL tool was called (not the hallucinated result)
    assert len(result["trace"]) == 1
    assert result["trace"][0]["tool"] == "indicators_full_analysis"


@pytest.mark.asyncio
async def test_deepseek_parser_strips_hallucinated_output():
    """Unit test for _deepseek_strip_output helper."""
    from src.llm.react_runtime import ReActRuntime
    runtime = ReActRuntime(
        mcp_manager=MockMCPManager(), system_prompt="test", model="test"
    )
    content = (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>my_tool\n"
        "{}\n<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        "<｜tool▁outputs▁begin｜>FAKE OUTPUT<｜tool▁outputs▁end｜>"
    )
    stripped = runtime._deepseek_strip_output(content)
    assert "<｜tool▁outputs▁begin｜>" not in stripped
    assert "FAKE OUTPUT" not in stripped
    assert "<｜tool▁calls▁begin｜>" in stripped


@pytest.mark.asyncio
async def test_deepseek_parse_calls_extracts_name_and_args():
    """Unit test for _parse_deepseek_calls helper."""
    from src.llm.react_runtime import ReActRuntime
    runtime = ReActRuntime(
        mcp_manager=MockMCPManager(), system_prompt="test", model="test"
    )
    content = (
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>indicators_full_analysis\n"
        "```json\n{\"symbol\": \"ETH\", \"interval\": \"M15\"}\n```"
        "<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
    )
    calls = runtime._parse_deepseek_calls(content)
    assert len(calls) == 1
    assert calls[0]["name"] == "indicators_full_analysis"
    assert calls[0]["args"] == {"symbol": "ETH", "interval": "M15"}


@pytest.mark.asyncio
async def test_llm_api_error_returns_error_dict():
    """If the LLM API itself throws, run() returns an error dict without raising."""
    from src.llm.react_runtime import ReActRuntime
    manager = MockMCPManager()
    runtime = ReActRuntime(mcp_manager=manager, system_prompt="test", model="test-model")

    with patch.object(
        runtime._client.chat.completions, "create",
        new=AsyncMock(side_effect=Exception("Connection refused"))
    ):
        result = await runtime.run("Analyze BTC")

    assert result["thesis"] is None
    assert result["error"] is not None
    assert "Connection refused" in result["error"]
