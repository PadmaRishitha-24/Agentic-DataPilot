"""Structured decision and execution schemas for Agentic DataPilot."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class DecisionType(str, Enum):
    CALL_TOOL = "call_tool"
    FINAL_ANSWER = "final_answer"
    CLARIFY = "clarify"


class AgentDecision(BaseModel):
    """Structured decision made by the agent LLM at each step."""

    action: Literal["call_tool", "final_answer", "clarify"] = Field(
        description="The action type: 'call_tool' to run an MCP tool, 'final_answer' when evidence is complete, or 'clarify' when user input is ambiguous."
    )
    rationale: str = Field(
        description="A concise 1-2 sentence explanation of why this action was chosen."
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="Name of the MCP tool to call (required if action is 'call_tool').",
    )
    tool_args: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Dictionary of arguments matching the tool schema.",
    )
    answer: Optional[Any] = Field(
        default=None,
        description="The final grounded answer (required if action is 'final_answer').",
    )
    question: Optional[str] = Field(
        default=None,
        description="The clarifying question to the user (required if action is 'clarify').",
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> "AgentDecision":
        if self.action == "call_tool":
            if not self.tool_name or not self.tool_name.strip():
                raise ValueError("tool_name is required when action is 'call_tool'.")
            if self.tool_args is None:
                self.tool_args = {}
        elif self.action == "final_answer":
            if self.answer is None:
                raise ValueError("answer text is required when action is 'final_answer'.")
            if isinstance(self.answer, dict):
                parts = [f"{str(k).replace('_', ' ').title()}: {v}" for k, v in self.answer.items()]
                self.answer = ". ".join(parts)
            elif not str(self.answer).strip():
                raise ValueError("answer text is required when action is 'final_answer'.")
            self.answer = str(self.answer).strip()
        elif self.action == "clarify":
            if not self.question or not self.question.strip():
                raise ValueError("question text is required when action is 'clarify'.")
        return self


class ExecutionStep(BaseModel):
    """Record of a single step executed by the agent loop."""

    step_number: int
    decision: AgentDecision
    tool_result: Optional[Dict[str, Any]] = None
    observation_summary: Optional[str] = None
    is_error: bool = False


class AgentResponse(BaseModel):
    """Complete response returned by the agent upon terminating the loop."""

    query: str
    status: Literal[
        "completed",
        "clarification_needed",
        "max_iterations_reached",
        "error",
        "unverified",
    ]
    final_answer: Optional[str] = None
    clarification_needed: Optional[str] = None
    steps: List[ExecutionStep] = Field(default_factory=list)
    total_iterations: int = 0
    verification: Optional[Dict[str, Any]] = None
    evidence_summary: Optional[str] = None
    error_message: Optional[str] = None


KNOWN_PRODUCTS: List[str] = [
    "Quantum Laptop",
    "CyberPhone Pro",
    "AeroDesk Ergonomic",
    "PulseWatch Ultra",
    "EchoPods Wireless",
    "SolarKey Mechanical Keyboard",
]


def extract_requested_product(query: Optional[str]) -> Optional[str]:
    """Extract requested product name from natural language query if present."""
    if not query:
        return None
    import re
    q = query.strip()
    q_lower = q.lower()

    # 1. Check known products first
    for p in KNOWN_PRODUCTS:
        if p.lower() in q_lower:
            return p

    # 2. Check explicit product markers
    patterns = [
        r'(?:for\s+(?:the\s+)?product|product\s*[:=]|product\s+named|product\s+called)\s+[\"\'\`]?([A-Za-z0-9\s\-]+?)[\"\'\`]?(?:\s+(?:in|during|for|from|between|with|across)\b|[,\.\?]|$|\s+revenue|\s+sales)',
        r'(?:total\s+revenue|revenue|total\s+sales|sales|profit\s*margin|growth)\s+(?:of|for)\s+(?:the\s+product\s+)?[\"\'\`]?([A-Za-z0-9\s\-]+?)[\"\'\`]?(?:\s+(?:in|during|for|from|between|with|across)\b|[,\.\?]|$)',
        r'(?:did|does)\s+[\"\'\`]?([A-Za-z0-9\s\-]+?)[\"\'\`]?\s+(?:make|generate|produce|earn)\b',
    ]
    for pat in patterns:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            cand_lower = candidate.lower()
            excluded_analytical = {
                "all", "all products", "all_products", "lowest", "highest", "top", "bottom",
                "each", "every", "rank", "ranking", "rankings", "each product", "every product",
                "products", "the products", "the product",
            }
            if cand_lower not in excluded_analytical:
                if not any(cand_lower.startswith(w) for w in ["lowest", "highest", "top", "bottom", "all "]):
                    return candidate

    return None

