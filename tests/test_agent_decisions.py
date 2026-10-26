"""Deterministic tests for agent decision schemas and dynamic execution loop."""

from typing import Any, Dict, List
import pytest
from pydantic import ValidationError

from agent.core import DataPilotAgent
from agent.schemas import AgentDecision
from mcp_server.client import MCPClient


class MockLLM:
    """Mock LLM returning predefined sequence of AgentDecisions for testing."""

    def __init__(self, decisions: List[AgentDecision]):
        self.decisions = list(decisions)
        self.call_count = 0

    def get_decision(
        self,
        user_query: str,
        tools: List[Dict[str, Any]],
        history: List[Any],
    ) -> AgentDecision:
        if self.call_count < len(self.decisions):
            decision = self.decisions[self.call_count]
            self.call_count += 1
            return decision
        # Default fallback
        return AgentDecision(
            action="final_answer",
            rationale="No more mock decisions, terminating.",
            answer="Default mock final answer.",
        )


class TestAgentDecisionSchemas:
    """Tests for Pydantic schema validation."""

    def test_valid_call_tool_decision(self):
        dec = AgentDecision(
            action="call_tool",
            rationale="Querying October metrics.",
            tool_name="calculate_metrics",
            tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
        )
        assert dec.action == "call_tool"
        assert dec.tool_name == "calculate_metrics"

    def test_call_tool_missing_name_fails(self):
        with pytest.raises(ValidationError):
            AgentDecision(
                action="call_tool",
                rationale="Missing tool name",
            )

    def test_valid_final_answer_decision(self):
        dec = AgentDecision(
            action="final_answer",
            rationale="All evidence is gathered.",
            answer="The top product is Quantum Laptop.",
        )
        assert dec.action == "final_answer"
        assert "Quantum Laptop" in (dec.answer or "")

    def test_final_answer_missing_answer_fails(self):
        with pytest.raises(ValidationError):
            AgentDecision(
                action="final_answer",
                rationale="Missing answer field",
            )

    def test_valid_clarify_decision(self):
        dec = AgentDecision(
            action="clarify",
            rationale="Ambiguous date range.",
            question="Which year do you mean?",
        )
        assert dec.action == "clarify"
        assert dec.question == "Which year do you mean?"

    def test_clarify_missing_question_fails(self):
        with pytest.raises(ValidationError):
            AgentDecision(
                action="clarify",
                rationale="Missing question field",
            )


@pytest.mark.anyio
class TestAgentLoopExecution:
    """Tests for dynamic agent loop with genuine MCP client."""

    async def test_terminates_immediately_on_final_answer(self):
        mock_llm = MockLLM([
            AgentDecision(
                action="final_answer",
                rationale="Question is conversational.",
                answer="Hello! I am ready to analyze your sales data.",
            )
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm)
            response = await agent.run("Hello")
            assert response.status == "completed"
            assert response.total_iterations == 1
            assert response.final_answer == "Hello! I am ready to analyze your sales data."

    async def test_terminates_on_clarify(self):
        mock_llm = MockLLM([
            AgentDecision(
                action="clarify",
                rationale="Missing product category.",
                question="Could you specify whether you mean Electronics or Furniture?",
            )
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm)
            response = await agent.run("Which was the best product?")
            assert response.status == "clarification_needed"
            assert response.total_iterations == 1
            assert "Electronics or Furniture" in (response.clarification_needed or "")

    async def test_dynamic_multi_step_tool_dispatch(self):
        mock_llm = MockLLM([
            AgentDecision(
                action="call_tool",
                rationale="Step 1: Check October metrics to find top product.",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
            ),
            AgentDecision(
                action="call_tool",
                rationale="Step 2: Compare October vs September for Quantum Laptop.",
                tool_name="compare_periods",
                tool_args={
                    "period_a_start": "2024-09-01",
                    "period_a_end": "2024-09-30",
                    "period_b_start": "2024-10-01",
                    "period_b_end": "2024-10-31",
                    "product": "Quantum Laptop",
                },
            ),
            AgentDecision(
                action="final_answer",
                rationale="Step 3: All data gathered.",
                answer="Quantum Laptop was the highest revenue product in October with $68,400.00 and 54.05% growth over September.",
            ),
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm)
            response = await agent.run("Which product generated highest revenue in Oct and its growth vs Sep?")
            assert response.status == "completed"
            assert response.total_iterations == 3
            assert len(response.steps) == 3
            assert response.steps[0].decision.tool_name == "calculate_metrics"
            assert response.steps[1].decision.tool_name == "compare_periods"
            assert "Quantum Laptop" in (response.final_answer or "")

    async def test_detects_and_prevents_duplicate_calls(self):
        # LLM tries calling the exact same tool twice
        mock_llm = MockLLM([
            AgentDecision(
                action="call_tool",
                rationale="Call 1",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
            ),
            AgentDecision(
                action="call_tool",
                rationale="Call 2 (Repeated identical call)",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
            ),
            AgentDecision(
                action="final_answer",
                rationale="Stopping after warning.",
                answer="Resolved.",
            ),
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm)
            response = await agent.run("Analyze October.")
            assert response.status == "completed"
            assert response.total_iterations == 3
            # Check step 2 flagged warning
            assert "Repeated tool call" in response.steps[1].observation_summary

    async def test_handles_max_iterations_exhaustion(self):
        # 6 continuous tool calls
        endless_decisions = [
            AgentDecision(
                action="call_tool",
                rationale=f"Call {i}",
                tool_name="get_sales_data",
                tool_args={"start_date": f"2024-10-0{i}", "end_date": f"2024-10-0{i}"},
            )
            for i in range(1, 7)
        ]
        mock_llm = MockLLM(endless_decisions)
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_iterations=5)
            response = await agent.run("Loop test.")
            assert response.status == "max_iterations_reached"
            assert response.total_iterations == 5
            assert "maximum limit" in (response.error_message or "")
