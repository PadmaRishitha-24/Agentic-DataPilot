"""Automated tests for evidence tracking, deterministic verification, and bounded correction."""

from typing import Any, Dict, List
import pytest

from agent.core import DataPilotAgent
from agent.evidence import EvidenceRecord
from agent.schemas import AgentDecision
from agent.verifier import AnswerVerifier
from mcp_server.client import MCPClient


@pytest.fixture
def sample_october_evidence() -> EvidenceRecord:
    """Fixture providing populated EvidenceRecord matching benchmark targets."""
    evidence = EvidenceRecord()
    # Add calculate_metrics observation for October
    evidence.add_observation(
        tool_name="calculate_metrics",
        tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
        tool_result={
            "status": "success",
            "total_records": 12,
            "total_revenue": 139200.0,
            "top_product": "Quantum Laptop",
            "top_product_revenue": 68400.0,
            "lowest_product": "SolarKey Mechanical Keyboard",
            "lowest_product_revenue": 4000.0,
            "product_rankings": [
                {"product": "Quantum Laptop", "total_revenue": 68400.0},
                {"product": "CyberPhone Pro", "total_revenue": 32000.0},
                {"product": "SolarKey Mechanical Keyboard", "total_revenue": 4000.0},
            ],
        },
    )
    # Add compare_periods observation for Quantum Laptop (Sep vs Oct)
    evidence.add_observation(
        tool_name="compare_periods",
        tool_args={
            "period_a_start": "2024-09-01",
            "period_a_end": "2024-09-30",
            "period_b_start": "2024-10-01",
            "period_b_end": "2024-10-31",
            "product": "Quantum Laptop",
        },
        tool_result={
            "status": "success",
            "product_filter": "Quantum Laptop",
            "period_a": {"total_revenue": 44400.0},
            "period_b": {"total_revenue": 68400.0},
            "comparison": {
                "absolute_change": 24000.0,
                "percentage_growth": 54.05,
                "zero_baseline_detected": False,
                "interpretation": "+54.05% growth",
            },
        },
    )
    return evidence


@pytest.fixture
def complete_six_product_october_evidence() -> EvidenceRecord:
    """Fixture providing full 6-product EvidenceRecord matching actual October data."""
    evidence = EvidenceRecord()
    evidence.add_observation(
        tool_name="calculate_metrics",
        tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
        tool_result={
            "status": "success",
            "total_records": 12,
            "total_revenue": 139200.0,
            "top_product": "Quantum Laptop",
            "top_product_revenue": 68400.0,
            "lowest_product": "SolarKey Mechanical Keyboard",
            "lowest_product_revenue": 4000.0,
            "product_rankings": [
                {"product": "Quantum Laptop", "total_revenue": 68400.0},
                {"product": "CyberPhone Pro", "total_revenue": 32000.0},
                {"product": "AeroDesk Ergonomic", "total_revenue": 15050.0},
                {"product": "PulseWatch Ultra", "total_revenue": 13750.0},
                {"product": "EchoPods Wireless", "total_revenue": 6000.0},
                {"product": "SolarKey Mechanical Keyboard", "total_revenue": 4000.0},
            ],
        },
    )
    return evidence


@pytest.fixture
def galaxy_tablet_october_zero_records_evidence() -> EvidenceRecord:
    """Fixture providing EvidenceRecord where Galaxy Tablet returns 0 records in October 2024."""
    evidence = EvidenceRecord()
    evidence.add_observation(
        tool_name="calculate_metrics",
        tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31", "product": "Galaxy Tablet"},
        tool_result={
            "status": "success",
            "total_records": 0,
            "total_revenue": 0.0,
            "top_product": None,
            "top_product_revenue": 0.0,
            "lowest_product": None,
            "lowest_product_revenue": 0.0,
            "product_rankings": [],
        },
    )
    return evidence


class TestAnswerVerifier:
    """Tests for deterministic verification engine."""

    def test_grounded_answer_verified_successfully(self, sample_october_evidence):
        answer = "Quantum Laptop generated the highest revenue in October with $68,400.00, representing a 54.05% growth compared with September."
        res = AnswerVerifier.verify(answer, sample_october_evidence)
        assert res.is_verified is True
        assert res.confidence_score == 1.0
        assert len(res.contradictions) == 0
        assert len(res.unsupported_claims) == 0
        assert any("Quantum Laptop" in c for c in res.verified_claims)
        assert any("54.05" in c for c in res.verified_claims)

    def test_unsupported_revenue_fails_verification(self, sample_october_evidence):
        # Claiming $95,000.00 which never appears in observations
        answer = "Quantum Laptop generated $95,000.00 in revenue in October."
        res = AnswerVerifier.verify(answer, sample_october_evidence)
        assert res.is_verified is False
        assert len(res.unsupported_claims) > 0
        assert any("95,000" in u for u in res.unsupported_claims)

    def test_contradictory_percentage_growth_detected(self, sample_october_evidence):
        # Claiming 80.0% growth instead of actual 54.05%
        answer = "Quantum Laptop was the top product with $68,400.00 revenue and 80.0% growth over September."
        res = AnswerVerifier.verify(answer, sample_october_evidence)
        assert res.is_verified is False
        assert len(res.contradictions) > 0
        assert any("contradicts" in c for c in res.contradictions)

    def test_contradictory_top_product_fails(self, sample_october_evidence):
        # Claiming CyberPhone Pro was the top product when Quantum Laptop was
        answer = "CyberPhone Pro generated the highest revenue in October with $68,400.00."
        res = AnswerVerifier.verify(answer, sample_october_evidence)
        assert res.is_verified is False
        assert len(res.contradictions) > 0
        assert any("CyberPhone Pro" in c for c in res.contradictions)

    def test_zero_evidence_fails_verification(self):
        empty_evidence = EvidenceRecord()
        answer = "Quantum Laptop had $68,400 revenue."
        res = AnswerVerifier.verify(answer, empty_evidence)
        assert res.is_verified is False
        assert len(res.missing_evidence) > 0

    def test_lowest_revenue_product_passes_verification(self, sample_october_evidence):
        query = "Which product generated the lowest total revenue in October 2024?"
        answer = "SolarKey Mechanical Keyboard with $4,000.00 revenue."
        res = AnswerVerifier.verify(answer, sample_october_evidence, user_query=query)
        assert res.is_verified is True
        assert res.confidence_score > 0.0
        assert len(res.contradictions) == 0
        assert len(res.unsupported_claims) == 0
        assert any("SolarKey Mechanical Keyboard" in c for c in res.verified_claims)

    def test_highest_revenue_product_passes_verification(self, sample_october_evidence):
        query = "Which product generated the highest revenue in October 2024?"
        answer = "Quantum Laptop generated the highest revenue with $68,400.00."
        res = AnswerVerifier.verify(answer, sample_october_evidence, user_query=query)
        assert res.is_verified is True
        assert res.confidence_score > 0.0
        assert len(res.contradictions) == 0
        assert len(res.unsupported_claims) == 0
        assert any("Quantum Laptop" in c for c in res.verified_claims)

    def test_lowest_product_incorrectly_presented_as_highest_fails(self, sample_october_evidence):
        query = "Which product generated the highest revenue in October 2024?"
        answer = "SolarKey Mechanical Keyboard generated the highest revenue with $4,000.00."
        res = AnswerVerifier.verify(answer, sample_october_evidence, user_query=query)
        assert res.is_verified is False
        assert len(res.contradictions) > 0
        assert any("SolarKey Mechanical Keyboard" in c for c in res.contradictions)

    def test_top_product_incorrectly_presented_as_lowest_fails(self, sample_october_evidence):
        query = "Which product generated the lowest total revenue in October 2024?"
        answer = "Quantum Laptop was the lowest revenue product with $68,400.00."
        res = AnswerVerifier.verify(answer, sample_october_evidence, user_query=query)
        assert res.is_verified is False
        assert len(res.contradictions) > 0
        assert any("Quantum Laptop" in c for c in res.contradictions)

    def test_total_revenue_not_rejected_by_product_ranking(self, sample_october_evidence):
        query = "What was the total revenue across all products in October 2024?"
        answer = "The total revenue across all products in October 2024 was $139,200.00 across 12 orders."
        res = AnswerVerifier.verify(answer, sample_october_evidence, user_query=query)
        assert res.is_verified is True
        assert len(res.contradictions) == 0
        assert len(res.unsupported_claims) == 0
        assert any("139,200.00" in c for c in res.verified_claims)

    def test_unsupported_revenue_and_incorrect_period_rejected(self, sample_october_evidence):
        # Unsupported revenue
        query_rev = "Which product generated the lowest total revenue in October 2024?"
        answer_bad_rev = "SolarKey Mechanical Keyboard generated $9,999.00 revenue in October."
        res_bad_rev = AnswerVerifier.verify(answer_bad_rev, sample_october_evidence, user_query=query_rev)
        assert res_bad_rev.is_verified is False
        assert any("9,999" in u for u in res_bad_rev.unsupported_claims)

        # Incorrect period
        query_period = "Which product generated the lowest total revenue in October 2024?"
        answer_bad_period = "SolarKey Mechanical Keyboard generated $4,000.00 in December 2024."
        res_bad_period = AnswerVerifier.verify(answer_bad_period, sample_october_evidence, user_query=query_period)
        assert res_bad_period.is_verified is False
        assert any("December" in u for u in res_bad_period.unsupported_claims)

    def test_complete_six_product_ranking_passes_verification(self, complete_six_product_october_evidence):
        query = "Rank all products by total revenue in October 2024, from highest to lowest. Show each product's name and total revenue."
        answer = (
            "Product rankings by total revenue in October 2024 from highest to lowest:\n"
            "1. Quantum Laptop: $68,400.00\n"
            "2. CyberPhone Pro: $32,000.00\n"
            "3. AeroDesk Ergonomic: $15,050.00\n"
            "4. PulseWatch Ultra: $13,750.00\n"
            "5. EchoPods Wireless: $6,000.00\n"
            "6. SolarKey Mechanical Keyboard: $4,000.00"
        )
        res = AnswerVerifier.verify(answer, complete_six_product_october_evidence, user_query=query)
        assert res.is_verified is True
        assert res.confidence_score > 0.8
        assert len(res.contradictions) == 0
        assert len(res.unsupported_claims) == 0
        assert len(res.missing_evidence) == 0

    def test_partial_answer_to_full_ranking_fails_completeness_check(self, complete_six_product_october_evidence):
        query = "Rank all products by total revenue in October 2024, from highest to lowest. Show each product's name and total revenue."
        # Candidate answer contains only SolarKey Mechanical Keyboard at $4,000
        answer = "SolarKey Mechanical Keyboard with $4,000.00 revenue."
        res = AnswerVerifier.verify(answer, complete_six_product_october_evidence, user_query=query)
        assert res.is_verified is False
        assert len(res.missing_evidence) > 0
        assert any("completeness" in u.lower() or "missing" in u.lower() for u in res.unsupported_claims)
        assert any("partial" in c.lower() or "ranking" in c.lower() for c in res.contradictions)

    def test_inverted_order_to_full_ranking_fails_direction_check(self, complete_six_product_october_evidence):
        query = "Rank all products by total revenue in October 2024, from highest to lowest. Show each product's name and total revenue."
        # Candidate answer lists products in ascending (lowest to highest) order
        answer = (
            "1. SolarKey Mechanical Keyboard: $4,000.00\n"
            "2. EchoPods Wireless: $6,000.00\n"
            "3. PulseWatch Ultra: $13,750.00\n"
            "4. AeroDesk Ergonomic: $15,050.00\n"
            "5. CyberPhone Pro: $32,000.00\n"
            "6. Quantum Laptop: $68,400.00"
        )
        res = AnswerVerifier.verify(answer, complete_six_product_october_evidence, user_query=query)
        assert res.is_verified is False
        assert any("order" in c.lower() or "contradicts" in c.lower() for c in res.contradictions)

    def test_incorrect_revenue_value_in_ranking_fails(self, complete_six_product_october_evidence):
        query = "Rank all products by total revenue in October 2024, from highest to lowest. Show each product's name and total revenue."
        # Quantum Laptop revenue is incorrect ($99,000.00 instead of $68,400.00)
        answer = (
            "Product rankings by total revenue in October 2024 from highest to lowest:\n"
            "1. Quantum Laptop: $99,000.00\n"
            "2. CyberPhone Pro: $32,000.00\n"
            "3. AeroDesk Ergonomic: $15,050.00\n"
            "4. PulseWatch Ultra: $13,750.00\n"
            "5. EchoPods Wireless: $6,000.00\n"
            "6. SolarKey Mechanical Keyboard: $4,000.00"
        )
        res = AnswerVerifier.verify(answer, complete_six_product_october_evidence, user_query=query)
        assert res.is_verified is False
        assert res.confidence_score == 0.0
        assert any("99,000" in u for u in res.unsupported_claims)

    def test_missing_product_revenue_in_ranking_fails(self, complete_six_product_october_evidence):
        query = "Rank all products by total revenue in October 2024, from highest to lowest. Show each product's name and total revenue."
        # Products are named in order, but revenues are omitted
        answer = (
            "Rankings from highest to lowest:\n"
            "1. Quantum Laptop\n"
            "2. CyberPhone Pro\n"
            "3. AeroDesk Ergonomic\n"
            "4. PulseWatch Ultra\n"
            "5. EchoPods Wireless\n"
            "6. SolarKey Mechanical Keyboard"
        )
        res = AnswerVerifier.verify(answer, complete_six_product_october_evidence, user_query=query)
        assert res.is_verified is False
        assert res.confidence_score == 0.0
        assert any("revenue" in u.lower() or "missing" in u.lower() for u in res.unsupported_claims)

    def test_incomplete_answer_never_receives_100_percent_confidence(self, complete_six_product_october_evidence):
        query = "Rank all products by total revenue in October 2024, from highest to lowest. Show each product's name and total revenue."
        # Incomplete answer with only 1 product and its revenue
        answer = "SolarKey Mechanical Keyboard with $4,000.00 revenue."
        res = AnswerVerifier.verify(answer, complete_six_product_october_evidence, user_query=query)
        assert res.is_verified is False
        # Must strictly be 0.0, never 1.0 (100%)
        assert res.confidence_score == 0.0

    def test_profit_margin_limitation_statement_verified(self, sample_october_evidence):
        """When user asks for profit margin, an answer that accurately states the available data and explains that profit/cost data is unavailable is verified."""
        query = "What was the profit margin of Quantum Laptop in October 2024?"
        answer = (
            "In October 2024, Quantum Laptop generated $68,400.00 in total revenue. "
            "However, profit margin cannot be calculated because the sales dataset does not track cost of goods sold (COGS), operating expenses, or profit data."
        )
        res = AnswerVerifier.verify(answer, sample_october_evidence, user_query=query)
        assert res.is_verified is True
        assert res.confidence_score == 1.0
        assert len(res.contradictions) == 0
        assert len(res.unsupported_claims) == 0
        assert any("limitation" in c.lower() for c in res.verified_claims)
        assert any("Quantum Laptop" in c for c in res.verified_claims)

    def test_hallucinated_profit_margin_percentage_rejected(self, sample_october_evidence):
        """When user asks for profit margin, hallucinating a percentage (e.g. 49.43%) must be rejected with 0.0 confidence."""
        query = "What was the profit margin of Quantum Laptop in October 2024?"
        answer = "Quantum Laptop had a profit margin of 49.43% based on $68,400.00 revenue."
        res = AnswerVerifier.verify(answer, sample_october_evidence, user_query=query)
        assert res.is_verified is False
        assert res.confidence_score == 0.0
        assert len(res.contradictions) > 0
        assert any("profit margin" in c.lower() or "cost" in c.lower() or "49.43" in c.lower() for c in res.contradictions + res.unsupported_claims)

    def test_profit_margin_unrelated_lowest_rationale_rejected(self, sample_october_evidence):
        """When user asks for profit margin, answering with lowest product SolarKey is rejected."""
        query = "What was the profit margin of Quantum Laptop in October 2024?"
        answer = "SolarKey Mechanical Keyboard had the lowest revenue of $4,000.00 in October 2024."
        res = AnswerVerifier.verify(answer, sample_october_evidence, user_query=query)
        assert res.is_verified is False
        assert res.confidence_score == 0.0
        assert len(res.contradictions) > 0

    def test_unknown_product_observation_absence_verified(self, galaxy_tablet_october_zero_records_evidence):
        """Regression: Accurate statement that Galaxy Tablet had no sales records in October 2024 ($0.00 revenue) is verified."""
        query = "What was the total revenue for the product Galaxy Tablet in October 2024?"
        answer = "In October 2024, Galaxy Tablet had no sales records in the observed period, resulting in $0.00 total revenue across 0 orders."
        res = AnswerVerifier.verify(answer, galaxy_tablet_october_zero_records_evidence, user_query=query)
        assert res.is_verified is True
        assert res.confidence_score == 1.0
        assert len(res.contradictions) == 0
        assert len(res.unsupported_claims) == 0
        assert any("Galaxy Tablet" in c for c in res.verified_claims)
        assert any("$0.00" in c or "no sales" in c.lower() for c in res.verified_claims)

    def test_unknown_product_absence_from_rankings_verified(self, complete_six_product_october_evidence):
        """Regression: When tool returns complete rankings, stating that Galaxy Tablet does not appear and generated $0.00 is verified."""
        query = "What was the total revenue for the product Galaxy Tablet in October 2024?"
        answer = "Galaxy Tablet does not appear in the October 2024 sales records and had $0.00 total revenue across 0 orders."
        res = AnswerVerifier.verify(answer, complete_six_product_october_evidence, user_query=query)
        assert res.is_verified is True
        assert res.confidence_score == 1.0
        assert len(res.contradictions) == 0
        assert len(res.unsupported_claims) == 0
        assert any("Galaxy Tablet" in c for c in res.verified_claims)

    def test_unknown_product_dataset_wide_overclaim_rejected(self, galaxy_tablet_october_zero_records_evidence):
        """Regression: Claiming a product does not exist in the entire dataset when only October was queried must fail verification."""
        query = "What was the total revenue for the product Galaxy Tablet in October 2024?"
        answer = "Galaxy Tablet does not exist in the company's entire dataset."
        res = AnswerVerifier.verify(answer, galaxy_tablet_october_zero_records_evidence, user_query=query)
        assert res.is_verified is False
        assert res.confidence_score == 0.0
        assert len(res.unsupported_claims) > 0
        assert any("entire dataset" in u.lower() for u in res.unsupported_claims)

    def test_unknown_product_hallucinated_positive_revenue_rejected(self, galaxy_tablet_october_zero_records_evidence):
        """Regression: Claiming positive revenue for an absent product must be rejected as a contradiction."""
        query = "What was the total revenue for the product Galaxy Tablet in October 2024?"
        answer = "Galaxy Tablet generated $12,500.00 in total revenue in October 2024."
        res = AnswerVerifier.verify(answer, galaxy_tablet_october_zero_records_evidence, user_query=query)
        assert res.is_verified is False
        assert res.confidence_score == 0.0
        assert len(res.contradictions) > 0
        assert any("positive revenue" in c.lower() or "0 sales records" in c.lower() for c in res.contradictions)

    def test_unknown_product_conversational_shortcut_blocked(self, galaxy_tablet_october_zero_records_evidence):
        """Regression: Uninformative non-answers to specific product data queries must not bypass verification via conversational shortcut."""
        query = "What was the total revenue for the product Galaxy Tablet in October 2024?"
        answer = "I could not find that information."
        res = AnswerVerifier.verify(answer, galaxy_tablet_october_zero_records_evidence, user_query=query)
        assert res.is_verified is False
        assert res.confidence_score == 0.0


class MockLLMWithCorrection:
    """Mock LLM supporting structured decision and bounded correction."""

    def __init__(self, decisions: List[AgentDecision], corrected_answer: str):
        self.decisions = list(decisions)
        self.corrected_answer = corrected_answer
        self.call_count = 0
        self.correction_invoked = False
        self.last_user_query = None

    def get_decision(
        self,
        user_query: str,
        tools: List[Dict[str, Any]],
        history: List[Any],
    ) -> AgentDecision:
        self.last_user_query = user_query
        if self.call_count < len(self.decisions):
            decision = self.decisions[self.call_count]
            self.call_count += 1
            return decision
        return self.decisions[-1]

    def get_corrected_answer(
        self,
        user_query: str,
        candidate_answer: str,
        feedback: str,
    ) -> str:
        self.correction_invoked = True
        self.last_user_query = user_query
        return self.corrected_answer


@pytest.mark.anyio
class TestBoundedCorrectionFlow:
    """Tests for bounded correction integration in DataPilotAgent."""

    async def test_bounded_correction_succeeds(self):
        # Initial candidate answer has wrong growth percentage (75.0%), corrected answer fixes it (54.05%)
        mock_llm = MockLLMWithCorrection(
            decisions=[
                AgentDecision(
                    action="call_tool",
                    rationale="Check metrics.",
                    tool_name="calculate_metrics",
                    tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
                ),
                AgentDecision(
                    action="call_tool",
                    rationale="Compare periods.",
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
                    rationale="Formulating answer.",
                    # Erroneous initial answer: 75%
                    answer="Quantum Laptop was #1 in October with $68,400.00 revenue and 75.0% growth.",
                ),
            ],
            # Corrected answer
            corrected_answer="Quantum Laptop was the highest revenue product in October with $68,400.00, achieving 54.05% growth compared with September.",
        )

        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_corrections=1)
            response = await agent.run("October top product and growth vs September?")
            assert mock_llm.correction_invoked is True
            assert mock_llm.last_user_query == "October top product and growth vs September?"
            assert response.status == "completed"
            assert response.verification is not None
            assert response.verification["is_verified"] is True
            assert "54.05%" in (response.final_answer or "")

    async def test_partial_ranking_answer_corrected_and_preserves_query(self):
        """Regression: Partial ranking answer is rejected by completeness check, corrected, and preserves original query."""
        query = "Rank all products by total revenue in October 2024, from highest to lowest. Show each product's name and total revenue."
        mock_llm = MockLLMWithCorrection(
            decisions=[
                AgentDecision(
                    action="call_tool",
                    rationale="Calculate October metrics for all products.",
                    tool_name="calculate_metrics",
                    tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
                ),
                AgentDecision(
                    action="final_answer",
                    rationale="Formulating answer.",
                    # Erroneous partial answer: only SolarKey
                    answer="SolarKey Mechanical Keyboard with $4,000.00 revenue.",
                ),
            ],
            # Corrected answer providing complete ranking of all 6 products in descending order
            corrected_answer=(
                "Rankings of all products in October 2024 from highest to lowest revenue:\n"
                "1. Quantum Laptop: $68,400.00\n"
                "2. CyberPhone Pro: $32,000.00\n"
                "3. AeroDesk Ergonomic: $15,050.00\n"
                "4. PulseWatch Ultra: $13,750.00\n"
                "5. EchoPods Wireless: $6,000.00\n"
                "6. SolarKey Mechanical Keyboard: $4,000.00"
            ),
        )

        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_corrections=1)
            response = await agent.run(query)
            assert mock_llm.correction_invoked is True
            # Verifies original user question is preserved throughout bounded correction
            assert mock_llm.last_user_query == query
            assert response.status == "completed"
            assert response.verification is not None
            assert response.verification["is_verified"] is True
            assert "Quantum Laptop" in (response.final_answer or "")
            assert "SolarKey Mechanical Keyboard" in (response.final_answer or "")

    async def test_bounded_correction_failure_marks_unverified(self):
        # LLM correction still fails to fix the discrepancy
        mock_llm = MockLLMWithCorrection(
            decisions=[
                AgentDecision(
                    action="call_tool",
                    rationale="Check metrics.",
                    tool_name="calculate_metrics",
                    tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31"},
                ),
                AgentDecision(
                    action="final_answer",
                    rationale="Formulating answer.",
                    answer="Product X had $999,999.00 revenue.",
                ),
            ],
            # Flawed correction
            corrected_answer="Still Product X with $999,999.00 revenue.",
        )

        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_corrections=1)
            response = await agent.run("Top product?")
            assert mock_llm.correction_invoked is True
            assert response.status == "unverified"
            assert response.verification is not None
            assert response.verification["is_verified"] is False
            assert "failed" in (response.error_message or "").lower()
            assert "verification" in (response.error_message or "").lower()

    async def test_profit_margin_bounded_correction_and_sequential_step_numbering(self):
        """Regression: Profit margin question with initial hallucination triggers correction, preserves query, verifies limitation, and guarantees sequential step numbering."""
        query = "What was the profit margin of Quantum Laptop in October 2024?"
        mock_llm = MockLLMWithCorrection(
            decisions=[
                AgentDecision(
                    action="call_tool",
                    rationale="Check Quantum Laptop metrics for October 2024.",
                    tool_name="calculate_metrics",
                    tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31", "product": "Quantum Laptop"},
                ),
                AgentDecision(
                    action="final_answer",
                    rationale="Formulating profit margin answer.",
                    # Erroneous candidate answer calculating revenue share / hallucinating margin
                    answer="Quantum Laptop had a profit margin of 49.43% based on $68,400.00 revenue.",
                ),
            ],
            # Corrected answer stating limitation and referencing verified revenue
            corrected_answer=(
                "In October 2024, Quantum Laptop generated $68,400.00 in total revenue. "
                "However, profit margin cannot be calculated because the sales dataset does not track cost of goods sold (COGS), operating expenses, or profit data."
            ),
        )

        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_corrections=1)
            response = await agent.run(query)
            assert mock_llm.correction_invoked is True
            assert mock_llm.last_user_query == query
            assert response.status == "completed"
            assert response.verification is not None
            assert response.verification["is_verified"] is True
            assert response.verification["confidence_score"] == 1.0
            assert "Quantum Laptop" in (response.final_answer or "")
            assert "cannot be calculated" in (response.final_answer or "")

            # Verify strictly sequential 1-indexed step numbering (no repeated Step 2)
            step_numbers = [s.step_number for s in response.steps]
            assert step_numbers == list(range(1, len(response.steps) + 1))
            assert len(step_numbers) == 3  # Step 1: Tool call, Step 2: Failed answer, Step 3: Corrected answer

    async def test_galaxy_tablet_bounded_correction_flow(self):
        """Regression: Overclaiming dataset-wide absence for Galaxy Tablet triggers correction, receives feedback, and corrects to scoped absence."""
        query = "What was the total revenue for the product Galaxy Tablet in October 2024?"
        mock_llm = MockLLMWithCorrection(
            decisions=[
                AgentDecision(
                    action="call_tool",
                    rationale="Check sales metrics for Galaxy Tablet in October 2024.",
                    tool_name="calculate_metrics",
                    tool_args={"start_date": "2024-10-01", "end_date": "2024-10-31", "product": "Galaxy Tablet"},
                ),
                AgentDecision(
                    action="final_answer",
                    rationale="Formulating answer.",
                    # Erroneous candidate answer claiming dataset-wide absence
                    answer="Galaxy Tablet does not exist in the company's entire dataset.",
                ),
            ],
            # Corrected answer scoped to observed October 2024 period
            corrected_answer=(
                "In October 2024, Galaxy Tablet had no sales records in the observed period, "
                "resulting in $0.00 total revenue across 0 orders."
            ),
        )

        async with MCPClient() as client:
            agent = DataPilotAgent(mcp_client=client, llm=mock_llm, max_corrections=1)
            response = await agent.run(query)
            assert mock_llm.correction_invoked is True
            assert mock_llm.last_user_query == query
            assert response.status == "completed"
            assert response.verification is not None
            assert response.verification["is_verified"] is True
            assert response.verification["confidence_score"] == 1.0
            assert "Galaxy Tablet" in (response.final_answer or "")
            assert "$0.00" in (response.final_answer or "") or "no sales" in (response.final_answer or "").lower()

            step_numbers = [s.step_number for s in response.steps]
            assert step_numbers == list(range(1, len(response.steps) + 1))
            assert len(step_numbers) == 3

