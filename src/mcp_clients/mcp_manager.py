"""
MCPManager — starts all MCP servers as subprocesses and keeps them alive
for the lifetime of the trading bot. Provides a unified interface for
listing tools (in OpenAI function-calling schema) and calling them by name.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

_SERVERS_DIR = Path(__file__).parent.parent / "mcp_servers"

# All MCP servers that will be started. Order doesn't matter.
# Value is either a python script filename (relative to _SERVERS_DIR, run with
# the current interpreter) or a {"command", "args", "env"} spec for a
# non-python server (e.g. an npx-launched Node MCP server).
#
# The bot is 100% rule-based (no LLM) — the only MCP tool actually called
# in production is indicators_full_analysis (see orchestrator._fetch_price_and_atr).
# etoro/finnhub/cryptopanic/reddit/exa were used by the LLM ReAct research
# loop that has since been replaced by src/agents/thesis_builder.py.
_DEFAULT_SERVERS: dict[str, str | dict] = {
    "indicators":  "indicators_server.py",
}


class MCPManager:
    """
    Lifecycle manager for all MCP sub-servers.

    Usage:
        manager = MCPManager()
        await manager.start()
        result = await manager.call_tool("indicators_full_analysis", {"symbol": "BTC"})
        await manager.stop()
    """

    def __init__(self, server_scripts: dict[str, str | dict] | None = None):
        # name → script filename (relative to _SERVERS_DIR)
        self._server_scripts = server_scripts or _DEFAULT_SERVERS
        # tool_name → (ClientSession, tool_schema)
        self._tool_registry: dict[str, tuple[ClientSession, Any]] = {}
        self._exit_stack = AsyncExitStack()
        self._started = False
        # One asyncio.Lock per session to prevent concurrent stdio stream access
        # from overlapping APScheduler jobs (e.g. screen_US and screen_CRYPTO).
        self._session_locks: dict[int, asyncio.Lock] = {}

    async def start(self):
        """Start all MCP server subprocesses and build the tool registry."""
        if self._started:
            return
        await self._exit_stack.__aenter__()

        base_env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent.parent)}

        for name, spec in self._server_scripts.items():
            if isinstance(spec, dict):
                params = StdioServerParameters(
                    command=spec["command"],
                    args=spec.get("args", []),
                    env={**base_env, **spec.get("env", {})},
                )
            else:
                script_path = _SERVERS_DIR / spec
                if not script_path.exists():
                    logger.warning("MCP server script not found: %s — skipping", script_path)
                    continue
                params = StdioServerParameters(
                    command=sys.executable,
                    args=[str(script_path)],
                    env=base_env,
                )
            try:
                read, write = await self._exit_stack.enter_async_context(
                    stdio_client(params)
                )
                session: ClientSession = await self._exit_stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()

                tools_response = await session.list_tools()
                for tool in tools_response.tools:
                    self._tool_registry[tool.name] = (session, tool)
                    logger.debug("Registered MCP tool: %s (from %s)", tool.name, name)

                logger.info(
                    "MCP server '%s' started — %d tools registered",
                    name, len(tools_response.tools),
                )
            except Exception as exc:
                logger.error("Failed to start MCP server '%s': %s", name, exc)
                # Non-fatal: continue with remaining servers

        self._started = True
        logger.info(
            "MCPManager ready — %d tools available: %s",
            len(self._tool_registry),
            list(self._tool_registry.keys()),
        )

    async def stop(self):
        """Shut down all server subprocesses."""
        await self._exit_stack.aclose()
        self._started = False
        logger.info("MCPManager stopped")

    # ------------------------------------------------------------------ #
    # Tool access
    # ------------------------------------------------------------------ #

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Execute a tool by name via its MCP server session.

        Returns the parsed result (dict/list/str depending on the tool).
        Raises ValueError if the tool is not registered.

        Each server's ClientSession uses stdio streams that are not safe for
        concurrent use from multiple asyncio tasks.  We serialize calls per
        session via a per-session asyncio.Lock.
        """
        if tool_name not in self._tool_registry:
            raise ValueError(
                f"Tool '{tool_name}' not found. Available: {list(self._tool_registry)}"
            )
        session, _ = self._tool_registry[tool_name]
        session_id = id(session)
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        async with self._session_locks[session_id]:
            result = await session.call_tool(tool_name, arguments)

        # MCP result.content is a list of Content items (TextContent, etc.)
        if not result.content:
            return {}

        first = result.content[0]
        # TextContent has .text attribute
        text = getattr(first, "text", str(first))

        # Try to parse as JSON for structured tools
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text

    def available_tools(self) -> list[str]:
        return list(self._tool_registry.keys())
