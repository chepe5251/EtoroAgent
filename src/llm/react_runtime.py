"""
Generic ReAct loop that runs against any OpenAI-compatible endpoint.
The LLM sees a set of read-only tools, iterates freely, and returns a
structured thesis. It never has access to order-execution tools.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

# DeepSeek native tool-calling tokens (unicode special chars)
_DS_CALLS_BEGIN  = "<｜tool▁calls▁begin｜>"
_DS_CALL_BEGIN   = "<｜tool▁call▁begin｜>"
_DS_CALL_SEP     = "<｜tool▁sep｜>"
_DS_CALL_END     = "<｜tool▁call▁end｜>"
_DS_CALLS_END    = "<｜tool▁calls▁end｜>"
_DS_OUT_BEGIN    = "<｜tool▁outputs▁begin｜>"
_DS_OUT_END      = "<｜tool▁outputs▁end｜>"

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from src.mcp_clients.mcp_manager import MCPManager

logger = logging.getLogger(__name__)

_LOGS_DIR = Path(__file__).parent.parent.parent / "logs"


class ReActRuntime:
    """
    Drives a ReAct loop against LM Studio (OpenAI-compatible API).

    Flow per iteration:
      1. Send system_prompt + accumulated messages + tools to LLM
      2. If response has tool_calls → execute each via MCPManager,
         append assistant + tool result messages, repeat
      3. If response has plain text (no tool_calls) → parse as JSON thesis,
         return
      4. Hard-stop at max_iterations to cap local compute cost
    """

    def __init__(
        self,
        mcp_manager: "MCPManager",
        system_prompt: str,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_iterations: int | None = None,
    ):
        self.mcp_manager = mcp_manager
        self.system_prompt = system_prompt
        self.model = model or os.getenv("LLM_MODEL", "qwen2.5-7b-instruct")
        self.temperature = temperature if temperature is not None else float(
            os.getenv("LLM_TEMPERATURE", "0.3")
        )
        self.max_iterations = max_iterations or int(
            os.getenv("LLM_MAX_ITERATIONS", "8")
        )

        self._client = AsyncOpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
            api_key=api_key or os.getenv("LLM_API_KEY", "lm-studio"),
        )

    async def run(self, user_prompt: str) -> dict[str, Any]:
        """
        Execute the ReAct loop.

        Returns:
            {
                "thesis": dict | None,      # parsed JSON thesis from LLM
                "trace": list[dict],        # every tool call + result
                "error": str | None,        # set if loop exhausted or parse failed
            }
        """
        tools_schema = await self.mcp_manager.list_openai_tools()
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        trace: list[dict] = []

        for iteration in range(self.max_iterations):
            logger.debug("ReAct iteration %d/%d", iteration + 1, self.max_iterations)

            try:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools_schema if tools_schema else None,
                    tool_choice="auto" if tools_schema else None,
                    temperature=self.temperature,
                )
            except Exception as exc:
                logger.error("LLM call failed (iteration %d): %s", iteration + 1, exc)
                return {"thesis": None, "trace": trace, "error": str(exc)}

            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # ── Tool calls ────────────────────────────────────────────────
            if message.tool_calls:
                # Add assistant turn (with tool_calls) to context
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                })

                for tc in message.tool_calls:
                    tool_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    logger.info("ReAct → tool call: %s(%s)", tool_name, args)
                    try:
                        result = await self.mcp_manager.call_tool(tool_name, args)
                    except Exception as exc:
                        result = f"ERROR: {exc}"
                        logger.warning("Tool %s failed: %s", tool_name, exc)

                    result_str = (
                        json.dumps(result) if not isinstance(result, str) else result
                    )
                    logger.info("ReAct ← %s result (len=%d)", tool_name, len(result_str))

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })
                    trace.append({
                        "iteration": iteration,
                        "tool": tool_name,
                        "args": args,
                        "result_preview": result_str[:300],
                    })
                continue  # next iteration

            # ── DeepSeek native tool-call format (fallback) ───────────────
            # DeepSeek emits tool calls as plain content with special tokens
            # instead of the OpenAI tool_calls field.  We parse them, execute
            # the real tools, strip the hallucinated output, and re-inject.
            content = (message.content or "").strip()
            if _DS_CALLS_BEGIN in content:
                ds_calls = self._parse_deepseek_calls(content)
                if ds_calls:
                    # Keep only the tool-call section; discard hallucinated output
                    clean = self._deepseek_strip_output(content)
                    messages.append({"role": "assistant", "content": clean})

                    for ds_call in ds_calls:
                        name = ds_call["name"]
                        args = ds_call["args"]
                        call_id = ds_call["id"]
                        logger.info("ReAct → DeepSeek tool call: %s(%s)", name, args)
                        try:
                            result = await self.mcp_manager.call_tool(name, args)
                        except Exception as exc:
                            result = f"ERROR: {exc}"
                            logger.warning("Tool %s failed: %s", name, exc)

                        result_str = (
                            json.dumps(result) if not isinstance(result, str) else result
                        )
                        logger.info("ReAct ← %s result (len=%d)", name, len(result_str))
                        trace.append({
                            "iteration": iteration,
                            "tool": name,
                            "args": args,
                            "result_preview": result_str[:300],
                        })
                        # Re-inject as tool role — LM Studio will forward to DeepSeek
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": result_str,
                        })
                    continue  # next iteration with real data

            # ── Final answer (no tool calls) ──────────────────────────────
            logger.info("ReAct: LLM produced final answer (%d chars)", len(content))

            thesis_dict = self._parse_thesis(content)
            result = {"thesis": thesis_dict, "trace": trace, "error": None}
            self._save_trace(user_prompt, result)
            return result

        # Max iterations hit without a final answer
        logger.warning(
            "ReAct: max_iterations=%d reached without final answer", self.max_iterations
        )
        result = {
            "thesis": None,
            "trace": trace,
            "error": f"max_iterations ({self.max_iterations}) reached",
        }
        self._save_trace(user_prompt, result)
        return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # DeepSeek format helpers
    # ------------------------------------------------------------------ #

    def _parse_deepseek_calls(self, content: str) -> list[dict]:
        """
        Extract tool calls from DeepSeek's native token format.
        Returns list of {id, name, args} dicts.
        Ignores the hallucinated <｜tool▁outputs▁begin｜> section.
        """
        calls: list[dict] = []
        # Work only with the tool-calls block, before any hallucinated output
        block_start = content.find(_DS_CALLS_BEGIN)
        if block_start == -1:
            return []
        block_end = content.find(_DS_CALLS_END, block_start)
        calls_block = content[block_start + len(_DS_CALLS_BEGIN) : block_end if block_end != -1 else None]

        # Split on individual call markers
        raw_calls = calls_block.split(_DS_CALL_BEGIN)
        for i, raw in enumerate(raw_calls):
            raw = raw.strip()
            if not raw or _DS_CALL_SEP not in raw:
                continue
            _, rest = raw.split(_DS_CALL_SEP, 1)
            rest = rest.replace(_DS_CALL_END, "").strip()
            # First line = tool name; rest = JSON args (possibly in ```json``` fence)
            lines = rest.splitlines()
            tool_name = lines[0].strip() if lines else ""
            if not tool_name:
                continue
            # Extract JSON from optional code fence
            json_src = "\n".join(lines[1:]).strip()
            json_src = re.sub(r"^```(?:json)?", "", json_src, flags=re.MULTILINE).strip()
            json_src = re.sub(r"```$", "", json_src, flags=re.MULTILINE).strip()
            try:
                args = json.loads(json_src) if json_src else {}
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": f"ds_{i}", "name": tool_name, "args": args})
        return calls

    def _deepseek_strip_output(self, content: str) -> str:
        """
        Remove the hallucinated tool-output section that DeepSeek sometimes
        appends in a single forward pass, keeping only the tool-call tokens.
        This is the content we store as the 'assistant' turn so that the
        model can observe real results in the next turn.
        """
        # Drop everything from <｜tool▁outputs▁begin｜> onwards
        out_start = content.find(_DS_OUT_BEGIN)
        if out_start != -1:
            content = content[:out_start].strip()
        # Also ensure we close the tool_calls_end if it was stripped
        if _DS_CALLS_BEGIN in content and _DS_CALLS_END not in content:
            content += _DS_CALLS_END
        return content

    def _parse_thesis(self, content: str) -> dict | None:
        """Extract JSON object from the LLM's final response."""
        # Try direct parse
        text = content.strip()
        # Strip markdown fences
        if "```" in text:
            for block in text.split("```"):
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue
        # Try finding the first {...} blob
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        logger.warning("ReAct: could not parse JSON thesis from: %s", text[:200])
        return None

    def _save_trace(self, prompt: str, result: dict):
        """Persist the full reasoning trace to disk for audit."""
        _LOGS_DIR.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        # Extract symbol from prompt if possible
        symbol = "unknown"
        for word in prompt.split():
            if word.isupper() and 2 <= len(word) <= 8:
                symbol = word
                break
        filename = _LOGS_DIR / f"research_trace_{symbol}_{ts}.json"
        payload = {
            "prompt": prompt,
            "thesis": result.get("thesis"),
            "trace": result.get("trace"),
            "error": result.get("error"),
        }
        try:
            filename.write_text(json.dumps(payload, indent=2, default=str))
            logger.debug("Trace saved → %s", filename)
        except Exception as exc:
            logger.warning("Could not save trace: %s", exc)
