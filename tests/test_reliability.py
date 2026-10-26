"""Reliability and failure-mode tests for Agentic DataPilot."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import pytest

from agent.core import DataPilotAgent
from agent.llm import OllamaLLM, OllamaUnavailableError
from agent.schemas import AgentDecision
from mcp_server.client import MCPClient
from tools.sales_tools import calculate_metrics, get_sales_data


class TestDatasetReliability:
    """Tests for malformed, corrupted, and edge-case CSV datasets."""

    def test_completely_empty_csv_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("")
            temp_path = f.name
        try:
            res = get_sales_data(csv_path=temp_path)
            assert res["status"] == "error"
            assert res["error_type"] == "EmptyFileError"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_corrupted_csv_rows_handled_gracefully(self):
        # Header present, but rows have bad numbers or invalid dates
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("order_id,date,product,category,units_sold,unit_price,revenue\n")
            f.write("ORD-1,not-a-date,Widget,Cat,10,10.0,100.0\n")  # bad date
            f.write("ORD-2,2024-10-01,Widget,Cat,not_an_int,10.0,100.0\n")  # bad int
            f.write("ORD-3,2024-10-02,GoodWidget,Cat,5,20.0,100.0\n")  # valid row
            temp_path = f.name

        try:
            res = get_sales_data(csv_path=temp_path)
            assert res["status"] == "success"
            # Only the valid row should be ingested
            assert res["total_records"] == 1
            assert res["records"][0]["product"] == "GoodWidget"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_zero_division_safety_in_metrics(self):
        # Metrics on empty list should return zeros, never raise ZeroDivisionError
        res = calculate_metrics(records=[])
        assert res["status"] == "success"
        assert res["total_revenue"] == 0.0
        assert res["average_order_revenue"] == 0.0
        assert res["product_rankings"] == []


class BrokenJSONLLM:
    """Mock LLM returning invalid non-JSON output."""

    def get_decision(self, user_query: str, tools: List[Any], history: List[Any]) -> AgentDecision:
        raise ValueError("Failed to parse Ollama output as JSON: Not valid JSON content")


@pytest.mark.anyio
class TestAgentReliability:
    """Tests for agent loop resilience during runtime faults and service interruptions."""

    async def test_ollama_unavailable_raises_clear_error(self):
        # Points to a nonexistent port where Ollama is definitely not running
        dead_llm = OllamaLLM(base_url="http://127.0.0.1:59999", timeout=1.0)
        assert dead_llm.check_health() is False

        with pytest.raises(OllamaUnavailableError) as exc_info:
            dead_llm.get_decision("Hello", [], [])
        err_str = str(exc_info.value).lower()
        assert "ollama" in err_str and ("cannot connect" in err_str or "timed out" in err_str)

    async def test_agent_handles_llm_crash_gracefully(self):
        broken_llm = BrokenJSONLLM()
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=broken_llm)
            response = await agent.run("What was the top product?")
            assert response.status == "error"
            assert "Not valid JSON" in (response.error_message or "")
            assert len(response.steps) == 1
            assert response.steps[0].is_error is True

    async def test_agent_handles_unknown_tool_and_continues(self):
        class StepwiseLLM:
            def __init__(self):
                self.calls = 0

            def get_decision(self, user_query: str, tools: List[Any], history: List[Any]) -> AgentDecision:
                self.calls += 1
                if self.calls == 1:
                    # Tries invalid tool
                    return AgentDecision(
                        action="call_tool",
                        rationale="Attempting imaginary tool.",
                        tool_name="delete_database_now",
                        tool_args={},
                    )
                # Corrects self and calls final answer
                return AgentDecision(
                    action="final_answer",
                    rationale="Recovered from invalid tool attempt.",
                    answer="Acknowledged invalid tool and recovered.",
                )

        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=StepwiseLLM())
            response = await agent.run("Test recovery")
            assert response.status == "completed"
            assert response.total_iterations == 2
            # Step 1 should record error
            assert response.steps[0].is_error is True
            assert "does not exist on MCP server" in response.steps[0].observation_summary
