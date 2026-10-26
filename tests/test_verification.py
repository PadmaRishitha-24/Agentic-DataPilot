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
            "product_rankings": [
                {"product": "Quantum Laptop", "total_revenue": 68400.0},
                {"product": "CyberPhone Pro", "total_revenue": 32000.0},
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


class MockLLMWithCorrection:
    """Mock LLM supporting structured decision and bounded correction."""

    def __init__(self, decisions: List[AgentDecision], corrected_answer: str):
        self.decisions = list(decisions)
        self.corrected_answer = corrected_answer
        self.call_count = 0
        self.correction_invoked = False

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
        return self.decisions[-1]

    def get_corrected_answer(
        self,
        user_query: str,
        candidate_answer: str,
        feedback: str,
    ) -> str:
        self.correction_invoked = True
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
            assert response.status == "completed"
            assert response.verification is not None
            assert response.verification["is_verified"] is True
            assert "54.05%" in (response.final_answer or "")

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
