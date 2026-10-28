"""Deterministic tests for agent decision schemas and dynamic execution loop."""

from typing import Any, Dict, List
import pytest
from pydantic import ValidationError

from agent.core import DataPilotAgent
from agent.schemas import AgentDecision, extract_requested_product
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

    def get_corrected_answer(
        self,
        user_query: str,
        candidate_answer: str,
        feedback: str,
    ) -> str:
        return candidate_answer


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

    async def test_run_query_interface_compatibility(self):
        """Verify that both run() and run_query() can be invoked without interface mismatch."""
        mock_llm = MockLLM([
            AgentDecision(
                action="final_answer",
                rationale="Direct answer.",
                answer="Interface compatibility verified.",
            )
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm)
            assert hasattr(agent, "run")
            assert hasattr(agent, "run_query")
            # Verify run_query executes properly with optional tools arg
            response = await agent.run_query("Test interface", tools=[])
            assert response.status == "completed"
            assert response.final_answer == "Interface compatibility verified."

    async def test_prevent_list_of_products_as_filter_in_agent_loop(self):
        """Ensure agent loop catches and rejects a list of product names before tool execution."""
        mock_llm = MockLLM([
            # Attempt 1: Invalid tool args (list of products)
            AgentDecision(
                action="call_tool",
                rationale="Attempting to filter by multiple products as a list.",
                tool_name="get_sales_data",
                tool_args={"product": ["Quantum Laptop", "CyberPhone Pro"]},
            ),
            # Attempt 2: Recovered valid call to calculate_metrics without product filter
            AgentDecision(
                action="call_tool",
                rationale="Corrected: analyze all products with calculate_metrics.",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
            ),
            # Attempt 3: Final answer
            AgentDecision(
                action="final_answer",
                rationale="Final answer based on observed metrics.",
                answer="SolarKey Mechanical Keyboard generated the lowest revenue in October with $4,000.00.",
            ),
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_iterations=5)
            response = await agent.run("Which product generated lowest revenue in October?")
            assert response.status == "completed"
            assert response.total_iterations == 3
            # Verify step 1 recorded argument error
            assert "Argument Error" in response.steps[0].observation_summary
            assert response.steps[0].is_error is True
            # Verify step 2 executed calculate_metrics successfully
            assert response.steps[1].decision.tool_name == "calculate_metrics"
            assert "SolarKey Mechanical Keyboard" in (response.final_answer or "")

    async def test_lowest_revenue_october_product_flow(self):
        """Verify complete execution flow for finding the lowest revenue product in October."""
        mock_llm = MockLLM([
            AgentDecision(
                action="call_tool",
                rationale="Calculate October sales metrics across all products to determine lowest revenue.",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
            ),
            AgentDecision(
                action="final_answer",
                rationale="SolarKey Mechanical Keyboard had the lowest revenue in October.",
                answer="SolarKey Mechanical Keyboard generated the lowest total revenue in October 2024 with $4,000.00.",
            ),
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_iterations=5)
            response = await agent.run("Which product generated the lowest total revenue in October?")
            assert response.status == "completed"
            assert response.total_iterations == 2
            assert response.total_iterations <= 5
            assert response.verification is not None
            assert response.verification["is_verified"] is True
            assert response.verification["confidence_score"] == 1.0
            assert "SolarKey Mechanical Keyboard" in (response.final_answer or "")
            assert "4,000" in (response.final_answer or "")

    async def test_valid_ranking_observation_followed_by_final_answer(self):
        """Regression 1: Valid ranking observation followed by a final answer."""
        mock_llm = MockLLM([
            AgentDecision(
                action="call_tool",
                rationale="Retrieve complete October metrics and product rankings.",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
            ),
            AgentDecision(
                action="final_answer",
                rationale="Formulate final answer directly from verified rankings.",
                answer="SolarKey Mechanical Keyboard generated the lowest total revenue in October with $4,000.00.",
            ),
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_iterations=5)
            response = await agent.run("Which product generated the lowest total revenue in October?")
            assert response.status == "completed"
            assert response.total_iterations == 2
            assert response.verification["is_verified"] is True
            assert "SolarKey Mechanical Keyboard" in (response.final_answer or "")
            assert "4,000" in (response.final_answer or "")

    async def test_invalid_product_filter_attempt_followed_by_successful_recovery(self):
        """Regression 2: An invalid product-filter attempt ('lowest_revenue_product') followed by successful recovery."""
        mock_llm = MockLLM([
            # Attempt 1: Invalid analytical operation as product filter
            AgentDecision(
                action="call_tool",
                rationale="Attempting to filter by analytical operation name.",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31", "product": "lowest_revenue_product"},
            ),
            # Attempt 2: Recovered - call calculate_metrics without product filter to obtain rankings
            AgentDecision(
                action="call_tool",
                rationale="Corrected: calling calculate_metrics without product filter to obtain rankings.",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
            ),
            # Attempt 3: Final answer based on valid rankings
            AgentDecision(
                action="final_answer",
                rationale="State verified lowest revenue product.",
                answer="SolarKey Mechanical Keyboard generated the lowest total revenue in October with $4,000.00.",
            ),
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_iterations=5)
            response = await agent.run("Which product generated the lowest total revenue in October?")
            assert response.status == "completed"
            assert response.total_iterations == 3
            # Ensure attempt 1 was flagged as error / invalid analytical argument
            assert response.steps[0].is_error is True
            assert "analytical operation" in response.steps[0].observation_summary.lower()
            # Ensure answer is verified
            assert response.verification["is_verified"] is True
            assert "SolarKey Mechanical Keyboard" in (response.final_answer or "")

    async def test_repeated_call_warning_followed_by_corrected_decision(self):
        """Regression 3: A repeated-call warning followed by a corrected decision."""
        mock_llm = MockLLM([
            # Attempt 1: Call get_sales_data
            AgentDecision(
                action="call_tool",
                rationale="Fetch raw sales data for October.",
                tool_name="get_sales_data",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
            ),
            # Attempt 2: Repeated call with identical args
            AgentDecision(
                action="call_tool",
                rationale="Accidentally repeating identical tool call.",
                tool_name="get_sales_data",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
            ),
            # Attempt 3: Corrected decision after warning
            AgentDecision(
                action="call_tool",
                rationale="Correcting: aggregate metrics to determine rankings.",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
            ),
            # Attempt 4: Conclude with final answer
            AgentDecision(
                action="final_answer",
                rationale="Final answer based on calculated metrics.",
                answer="SolarKey Mechanical Keyboard generated the lowest total revenue in October with $4,000.00.",
            ),
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_iterations=5)
            response = await agent.run("Which product generated the lowest total revenue in October?")
            assert response.status == "completed"
            assert response.total_iterations == 4
            # Verify step 2 was flagged as a repeated call warning and an error
            assert response.steps[1].is_error is True
            assert "Repeated tool call" in response.steps[1].observation_summary
            # Verify step 3 executed calculate_metrics
            assert response.steps[2].decision.tool_name == "calculate_metrics"
            assert response.verification["is_verified"] is True

    async def test_answer_remains_unsupported_must_not_be_marked_verified(self):
        """Regression 4: An answer that remains unsupported and must not be marked verified."""
        mock_llm = MockLLM([
            AgentDecision(
                action="call_tool",
                rationale="Fetch October metrics.",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
            ),
            # Step 2: Hallucinated/contradictory answer that asserts wrong product & revenue
            AgentDecision(
                action="final_answer",
                rationale="Assert unsupported product.",
                answer="PulseWatch Ultra generated the lowest total revenue in October with $150.00.",
            ),
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_iterations=5)
            response = await agent.run("Which product generated the lowest total revenue in October?")
            # Must NOT be marked completed/verified
            assert response.status == "unverified"
            assert response.verification is not None
            assert response.verification["is_verified"] is False
            assert len(response.verification["contradictions"]) > 0 or len(response.verification["unsupported_claims"]) > 0

    async def test_total_revenue_one_call_followed_by_grounded_final_answer(self):
        """Regression: Total-revenue question with one successful metrics call followed by grounded final answer."""
        mock_llm = MockLLM([
            AgentDecision(
                action="call_tool",
                rationale="Calculate total revenue across all products for October 2024.",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
            ),
            AgentDecision(
                action="final_answer",
                rationale="Formulate final answer directly from observed total revenue.",
                answer="The total revenue across all products in October 2024 was $139,200.00 across 12 orders.",
            ),
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_iterations=5)
            response = await agent.run("What was the total revenue across all products in October 2024?")
            assert response.status == "completed"
            assert response.total_iterations == 2
            assert response.verification is not None
            assert response.verification["is_verified"] is True
            assert "139,200" in (response.final_answer or "")
            # Ensure no unnecessary tools were called
            assert len(response.steps) == 2
            assert response.steps[0].decision.tool_name == "calculate_metrics"

    async def test_total_revenue_repeated_call_recovers_instead_of_looping(self):
        """Regression: Repeated identical call recovers into verified final answer instead of looping."""
        mock_llm = MockLLM([
            # Step 1: Calculate metrics returns $139,200
            AgentDecision(
                action="call_tool",
                rationale="Calculate October metrics.",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
            ),
            # Step 2: Agent attempts to repeat identical calculate_metrics call
            AgentDecision(
                action="call_tool",
                rationale="Accidentally repeating identical call.",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
            ),
            # Step 3: Synthesis decision triggered
            AgentDecision(
                action="final_answer",
                rationale="Concluding with verified total revenue.",
                answer="The total revenue across all products in October 2024 was $139,200.00 across 12 orders.",
            ),
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_iterations=5)
            response = await agent.run("What was the total revenue across all products in October 2024?")
            assert response.status == "completed"
            assert response.total_iterations <= 3
            assert response.total_iterations < 5  # Must NOT consume all 5 iterations
            assert response.verification["is_verified"] is True
            assert "139,200" in (response.final_answer or "")
            # Step 1 was calculate_metrics
            assert response.steps[0].decision.tool_name == "calculate_metrics"
            # Previous observation preserved in warning summary
            assert any("Previous observation was:" in (s.observation_summary or "") for s in response.steps)

    async def test_repeated_call_stops_safely_when_no_evidence(self):
        """Regression: Repeated identical call stops safely instead of endlessly spinning when no evidence gathered."""
        mock_llm = MockLLM([
            # Attempt 1: Call nonexistent product
            AgentDecision(
                action="call_tool",
                rationale="Call 1.",
                tool_name="get_sales_data",
                tool_args={"product": "NonexistentGadget"},
            ),
            # Attempt 2: Repeat call 1
            AgentDecision(
                action="call_tool",
                rationale="Call 2 repeated.",
                tool_name="get_sales_data",
                tool_args={"product": "NonexistentGadget"},
            ),
            # Attempt 3: Repeat call 1 again
            AgentDecision(
                action="call_tool",
                rationale="Call 3 repeated again.",
                tool_name="get_sales_data",
                tool_args={"product": "NonexistentGadget"},
            ),
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_iterations=5)
            response = await agent.run("Find NonexistentGadget revenue.")
            assert response.status in ["error", "completed"]
            assert response.total_iterations <= 3

    async def test_lowest_revenue_product_completing_within_iteration_limit(self):
        """Regression: The lowest-revenue-product question completing within the 5-iteration limit."""
        mock_llm = MockLLM([
            # Attempt 1: Invalid filter
            AgentDecision(
                action="call_tool",
                rationale="Attempting lowest_revenue_product filter.",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31", "product": "lowest_revenue_product"},
            ),
            # Attempt 2: Recovered ranking call
            AgentDecision(
                action="call_tool",
                rationale="Aggregate metrics for all products.",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
            ),
            # Attempt 3: Conclude with final answer
            AgentDecision(
                action="final_answer",
                rationale="State verified lowest revenue product.",
                answer="SolarKey Mechanical Keyboard generated the lowest total revenue in October with $4,000.00.",
            ),
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_iterations=5)
            response = await agent.run("Which product generated the lowest total revenue in October?")
            assert response.status == "completed"
            assert response.total_iterations <= 5
            assert response.verification["is_verified"] is True
            assert "SolarKey Mechanical Keyboard" in (response.final_answer or "")
            assert "4,000" in (response.final_answer or "")

    async def test_complete_six_product_ranking_returns_descending_order(self):
        """Regression: The complete six-product ranking question returns all six products in descending revenue order."""
        query = "Rank all products by total revenue in October 2024, from highest to lowest. Show each product's name and total revenue."
        mock_llm = MockLLM([
            AgentDecision(
                action="call_tool",
                rationale="Calculate metrics for all products in October.",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
            ),
            AgentDecision(
                action="final_answer",
                rationale="Formulate complete 6-product ranking in descending order.",
                answer=(
                    "Product rankings by total revenue in October 2024 from highest to lowest:\n"
                    "1. Quantum Laptop: $68,400.00\n"
                    "2. CyberPhone Pro: $32,000.00\n"
                    "3. AeroDesk Ergonomic: $15,050.00\n"
                    "4. PulseWatch Ultra: $13,750.00\n"
                    "5. EchoPods Wireless: $6,000.00\n"
                    "6. SolarKey Mechanical Keyboard: $4,000.00"
                ),
            ),
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_iterations=5)
            response = await agent.run(query)
            assert response.status == "completed"
            assert response.total_iterations <= 3
            assert response.verification["is_verified"] is True
            ans = response.final_answer or ""
            # Verify all six products are in the answer
            expected_prods = [
                "Quantum Laptop", "CyberPhone Pro", "AeroDesk Ergonomic",
                "PulseWatch Ultra", "EchoPods Wireless", "SolarKey Mechanical Keyboard"
            ]
            for prod in expected_prods:
                assert prod in ans
            # Verify descending order of appearance
            indices = [ans.find(prod) for prod in expected_prods]
            assert indices == sorted(indices)

    async def test_prevent_all_products_literal_filter_in_agent_loop(self):
        """Regression: Ensure 'all_products' is not accepted as a literal product filter and recovers cleanly."""
        query = "Rank all products by total revenue in October 2024, from highest to lowest. Show each product's name and total revenue."
        mock_llm = MockLLM([
            # Attempt 1: Mistaken call with product: "all_products"
            AgentDecision(
                action="call_tool",
                rationale="Querying all products.",
                tool_name="get_sales_data",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31", "product": "all_products"},
            ),
            # Attempt 2: Recovered calculate_metrics call without product filter
            AgentDecision(
                action="call_tool",
                rationale="Calculate metrics for all products without filter.",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
            ),
            # Attempt 3: Complete 6-product ranking answer
            AgentDecision(
                action="final_answer",
                rationale="Provide verified complete ranking.",
                answer=(
                    "Product rankings by total revenue in October 2024 from highest to lowest:\n"
                    "1. Quantum Laptop: $68,400.00\n"
                    "2. CyberPhone Pro: $32,000.00\n"
                    "3. AeroDesk Ergonomic: $15,050.00\n"
                    "4. PulseWatch Ultra: $13,750.00\n"
                    "5. EchoPods Wireless: $6,000.00\n"
                    "6. SolarKey Mechanical Keyboard: $4,000.00"
                ),
            ),
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_iterations=5)
            response = await agent.run(query)
            assert response.status == "completed"
            assert response.total_iterations <= 5
            assert response.verification["is_verified"] is True
            assert "Quantum Laptop" in (response.final_answer or "")
            assert "SolarKey Mechanical Keyboard" in (response.final_answer or "")

    async def test_highest_revenue_october_product_flow(self):
        """Regression: Highest-revenue product question completes and verifies correctly."""
        mock_llm = MockLLM([
            AgentDecision(
                action="call_tool",
                rationale="Calculate October metrics.",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
            ),
            AgentDecision(
                action="final_answer",
                rationale="Conclude with highest revenue product.",
                answer="Quantum Laptop generated the highest revenue in October with $68,400.00.",
            ),
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_iterations=5)
            response = await agent.run("Which product generated the highest revenue in October 2024?")
            assert response.status == "completed"
            assert response.verification["is_verified"] is True
    async def test_profit_margin_unavailable_metric_flow_states_limitation(self):
        """Regression: Profit margin question with limitation answer completes, verifies, and maintains sequential step numbering."""
        query = "What was the profit margin of Quantum Laptop in October 2024?"
        mock_llm = MockLLM([
            AgentDecision(
                action="call_tool",
                rationale="Retrieve sales data for Quantum Laptop in October 2024.",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31", "product": "Quantum Laptop"},
            ),
            AgentDecision(
                action="final_answer",
                rationale="State verified revenue and explain that profit margin cannot be computed due to dataset limitations.",
                answer=(
                    "In October 2024, Quantum Laptop generated $68,400.00 across 3 orders (57 units sold). "
                    "However, profit margin cannot be calculated because the sales dataset does not track cost of goods sold (COGS), operating expenses, or profit data."
                ),
            ),
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_iterations=5)
            response = await agent.run(query)
            assert response.status == "completed"
            assert response.verification["is_verified"] is True
            assert response.verification["confidence_score"] == 1.0
            assert "Quantum Laptop" in (response.final_answer or "")
            assert "cannot be calculated" in (response.final_answer or "")
            # Ensure strictly sequential step numbering
            step_nums = [s.step_number for s in response.steps]
            assert step_nums == [1, 2]

    async def test_unknown_product_preserves_product_filter_in_call(self):
        """Regression: Question about Galaxy Tablet applies calculate_metrics with product filter and verifies 0 records correctly."""
        query = "What was the total revenue for the product Galaxy Tablet in October 2024?"
        mock_llm = MockLLM([
            AgentDecision(
                action="call_tool",
                rationale="Query metrics for Galaxy Tablet in October 2024.",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31", "product": "Galaxy Tablet"},
            ),
            AgentDecision(
                action="final_answer",
                rationale="Formulate grounded answer for Galaxy Tablet with 0 sales records.",
                answer="In October 2024, Galaxy Tablet had no sales records in the observed period, resulting in $0.00 total revenue across 0 orders.",
            ),
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_iterations=5)
            response = await agent.run(query)
            assert response.status == "completed"
            assert response.total_iterations <= 5
            assert response.verification["is_verified"] is True
            assert response.verification["confidence_score"] == 1.0
            # Confirm tool args preserved the product filter
            assert response.steps[0].decision.tool_args.get("product") == "Galaxy Tablet"
            assert "Galaxy Tablet" in (response.final_answer or "")
            assert "$0.00" in (response.final_answer or "")

    async def test_unknown_product_rankings_check_flow(self):
        """Regression: When calculate_metrics returns all products, agent checks whether requested product appears and verifies absence."""
        query = "What was the total revenue for the product Galaxy Tablet in October 2024?"
        mock_llm = MockLLM([
            AgentDecision(
                action="call_tool",
                rationale="Calculate October metrics across all products.",
                tool_name="calculate_metrics",
                tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
            ),
            AgentDecision(
                action="final_answer",
                rationale="State that Galaxy Tablet does not appear in October 2024 rankings and has $0.00 revenue.",
                answer="Galaxy Tablet does not appear in the October 2024 sales records and had $0.00 total revenue across 0 orders.",
            ),
        ])
        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_iterations=5)
            response = await agent.run(query)
            assert response.status == "completed"
            assert response.total_iterations <= 5
            assert response.verification["is_verified"] is True
            assert response.verification["confidence_score"] == 1.0
            assert "Galaxy Tablet" in (response.final_answer or "")
            assert "$0.00" in (response.final_answer or "")


class TestOllamaPromptGuidance:
    """Verify prompt formatting and instructions in OllamaLLM."""

    def test_prompt_includes_analytical_guidance_and_checklist(self):
        from agent.llm import OllamaLLM
        llm = OllamaLLM()
        prompt = llm._build_prompt(
            user_query="Which product generated the lowest total revenue in October?",
            tools=[{"name": "calculate_metrics", "description": "Calc metrics", "parameters": {"properties": {}}}],
            history=[],
        )
        assert "analytical operations, not literal product names" in prompt.lower()
        assert "lowest_revenue_product" in prompt
        assert "INSTRUCTIONS:" in prompt
        assert "calculate_metrics" in prompt

    def test_prompt_includes_profit_margin_guidance(self):
        from agent.llm import OllamaLLM
        llm = OllamaLLM()
        prompt = llm._build_prompt(
            user_query="What was the profit margin of Quantum Laptop in October 2024?",
            tools=[{"name": "calculate_metrics", "description": "Calc metrics", "parameters": {"properties": {}}}],
            history=[],
        )
        assert "profit or cost / unavailable metric" in prompt.lower()
        assert "cost of goods sold (cogs)" in prompt.lower()
        assert "cannot be calculated" in prompt.lower()

    def test_prompt_includes_specific_product_guidance(self):
        from agent.llm import OllamaLLM
        llm = OllamaLLM()
        prompt = llm._build_prompt(
            user_query="What was the total revenue for the product Galaxy Tablet in October 2024?",
            tools=[{"name": "calculate_metrics", "description": "Calc metrics", "parameters": {"properties": {}}}],
            history=[],
        )
        assert "galaxy tablet" in prompt.lower()
        assert "product='galaxy tablet'" in prompt.lower() or 'product="galaxy tablet"' in prompt.lower() or "galaxy tablet" in prompt.lower()
        assert "0 records" in prompt.lower()
        assert "do not claim" in prompt.lower() and "underlying database" in prompt.lower()

    def test_extract_requested_product_utility(self):
        assert extract_requested_product("What was the total revenue for the product Galaxy Tablet in October 2024?") == "Galaxy Tablet"
        assert extract_requested_product("What was the revenue of Quantum Laptop in October?") == "Quantum Laptop"
        assert extract_requested_product("How much did CyberPhone Pro make?") == "CyberPhone Pro"
        assert extract_requested_product("Rank all products by total revenue in October 2024, from highest to lowest.") is None
        assert extract_requested_product("Which product generated the lowest total revenue in October 2024?") is None
        assert extract_requested_product("Which product generated the highest revenue in October 2024?") is None
        assert extract_requested_product("What was the total revenue across all products in October 2024?") is None




