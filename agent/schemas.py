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
    answer: Optional[str] = Field(
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
            if not self.answer or not self.answer.strip():
                raise ValueError("answer text is required when action is 'final_answer'.")
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
    status: Literal["completed", "clarification_needed", "max_iterations_reached", "error"]
    final_answer: Optional[str] = None
    clarification_needed: Optional[str] = None
    steps: List[ExecutionStep] = Field(default_factory=list)
    total_iterations: int = 0
    verification: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
