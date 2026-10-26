"""Official MCP Client for Agentic DataPilot.

Manages connection to the local MCP server over stdio transport,
discovers registered tools, and executes tool invocations.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MCPClient:
    """Asynchronous MCP Client managing stdio transport and tool execution."""

    def __init__(
        self,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        self.command = command or sys.executable
        self.args = args or ["-m", "mcp_server.server"]
        self.cwd = cwd or str(PROJECT_ROOT)
        self.env = env or dict(os.environ)
        self._exit_stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None
        self._discovered_tools: Dict[str, Dict[str, Any]] = {}
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    async def connect(self) -> None:
        """Connect to local MCP server via stdio transport and initialize session."""
        if self._is_connected:
            return

        params = StdioServerParameters(
            command=self.command,
            args=self.args,
            cwd=self.cwd,
            env=self.env,
        )

        self._exit_stack = AsyncExitStack()
        try:
            read_stream, write_stream = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
            self._is_connected = True
            await self.discover_tools()
        except Exception as e:
            await self.disconnect()
            raise RuntimeError(f"Failed to connect to local MCP server: {e}") from e

    async def disconnect(self) -> None:
        """Cleanly disconnect session and shut down stdio process."""
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception:
                pass
            finally:
                self._exit_stack = None
        self._session = None
        self._is_connected = False
        self._discovered_tools.clear()

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()

    async def discover_tools(self) -> List[Dict[str, Any]]:
        """Discover tools exposed by the MCP server and cache their schemas."""
        if not self._session or not self._is_connected:
            raise RuntimeError("MCP client is not connected.")

        list_result = await self._session.list_tools()
        self._discovered_tools.clear()

        tools_summary = []
        for tool in list_result.tools:
            info = {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema if hasattr(tool, "inputSchema") else {},
            }
            self._discovered_tools[tool.name] = info
            tools_summary.append(info)

        return tools_summary

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return cached tool definitions for LLM prompting or inspection."""
        return list(self._discovered_tools.values())

    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Invoke an MCP tool by name with arguments and return the parsed result.

        Args:
            name: Registered tool name.
            arguments: Dictionary of arguments for the tool.

        Returns:
            Parsed JSON result dictionary from tool execution.
        """
        if not self._session or not self._is_connected:
            raise RuntimeError("MCP client is not connected.")

        if name not in self._discovered_tools:
            return {
                "status": "error",
                "error_type": "UnknownToolError",
                "message": f"Tool '{name}' is not registered on the MCP server. Available tools: {list(self._discovered_tools.keys())}",
            }

        args = arguments or {}

        try:
            call_result = await self._session.call_tool(name=name, arguments=args)
        except Exception as e:
            return {
                "status": "error",
                "error_type": "MCPInvocationError",
                "message": f"Exception while invoking tool '{name}': {str(e)}",
            }

        # Parse text content from CallToolResult
        if not call_result.content:
            return {
                "status": "success",
                "message": "Tool executed with empty output.",
            }

        raw_text = ""
        for block in call_result.content:
            if hasattr(block, "text") and block.text:
                raw_text += block.text

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return {
                "status": "success",
                "raw_output": raw_text,
            }

    # Synchronous helper for non-async callers
    def run_sync(self, coro):
        """Execute an asynchronous coroutine synchronously."""
        return asyncio.run(coro)
