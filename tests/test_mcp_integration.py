"""Integration tests for genuine local MCP client and server communication over stdio."""

import pytest
from mcp_server.client import MCPClient


@pytest.mark.anyio
class TestMCPIntegration:
    """End-to-end integration tests using official MCP stdio transport."""

    async def test_connect_discover_and_disconnect(self):
        async with MCPClient() as client:
            assert client.is_connected
            tools = client.get_tool_definitions()
            tool_names = [t["name"] for t in tools]
            assert "get_sales_data" in tool_names
            assert "calculate_metrics" in tool_names
            assert "compare_periods" in tool_names

        # After exiting context, client should be cleanly disconnected
        assert not client.is_connected

    async def test_invoke_get_sales_data_via_mcp(self):
        async with MCPClient() as client:
            result = await client.call_tool(
                "get_sales_data",
                {"start_date": "2024-10-01", "end_date": "2024-10-31"},
            )
            assert result["status"] == "success"
            assert result["total_records"] > 0
            assert all("2024-10-01" <= r["date"] <= "2024-10-31" for r in result["records"])

    async def test_invoke_calculate_metrics_via_mcp(self):
        async with MCPClient() as client:
            result = await client.call_tool(
                "calculate_metrics",
                {"start_date": "2024-10-01", "end_date": "2024-10-31"},
            )
            assert result["status"] == "success"
            assert result["top_product"] == "Quantum Laptop"
            assert result["top_product_revenue"] == 68400.0

    async def test_invoke_compare_periods_via_mcp(self):
        async with MCPClient() as client:
            result = await client.call_tool(
                "compare_periods",
                {
                    "period_a_start": "2024-09-01",
                    "period_a_end": "2024-09-30",
                    "period_b_start": "2024-10-01",
                    "period_b_end": "2024-10-31",
                    "product": "Quantum Laptop",
                },
            )
            assert result["status"] == "success"
            assert result["period_a"]["total_revenue"] == 44400.0
            assert result["period_b"]["total_revenue"] == 68400.0
            assert result["comparison"]["absolute_change"] == 24000.0
            assert result["comparison"]["percentage_growth"] == 54.05

    async def test_invoke_unknown_tool_returns_error(self):
        async with MCPClient() as client:
            result = await client.call_tool("nonexistent_tool_123")
            assert result["status"] == "error"
            assert result["error_type"] == "UnknownToolError"
