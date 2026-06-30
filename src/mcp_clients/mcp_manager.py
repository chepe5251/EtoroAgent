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
_DEFAULT_SERVERS: dict[str, str] = {
    "etoro":       "etoro_server.py",
    "indicators":  "indicators_server.py",
    "finnhub":     "finnhub_server.py",
    "cryptopanic": "cryptopanic_server.py",
    "reddit":      "reddit_server.py",
}


class MCPManager:
    """
    Lifecycle manager for all MCP sub-servers.

    Usage:
        manager = MCPManager()
        await manager.start()
        tools = await manager.list_openai_tools()
        result = await manager.call_tool("indicators_full_analysis", {"symbol": "BTC"})
        await manager.stop()
    """

    def __init__(self, server_scripts: dict[str, str] | None = None):
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

        env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent.parent)}

        for name, script_file in self._server_scripts.items():
            script_path = _SERVERS_DIR / script_file
            if not script_path.exists():
                logger.warning("MCP server script not found: %s — skipping", script_path)
                continue

            params = StdioServerParameters(
                command=sys.executable,
                args=[str(script_path)],
                env=env,
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

    async def list_openai_tools(self) -> list[dict]:
        """
        Return all registered tools in OpenAI function-calling schema format.
        This is what gets passed to the LLM in every ReAct iteration.
        """
        schemas = []
        for tool_name, (_, tool) in self._tool_registry.items():
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": (tool.description or "").strip(),
                    "parameters": tool.inputSchema or {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
            schemas.append(schema)
        return schemas

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
