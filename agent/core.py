"""Dynamic Agent Loop for Agentic DataPilot.

Orchestrates local LLM decision-making, genuine MCP tool invocation,
observation feedback, cycle detection, and iteration limits.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from config import MAX_AGENT_ITERATIONS
from mcp_server.client import MCPClient
from agent.llm import OllamaLLM
from agent.schemas import AgentDecision, AgentResponse, ExecutionStep


class DataPilotAgent:
    """Autonomous agent driving MCP tool execution via local LLM reasoning."""

    def __init__(
        self,
        mcp_client: Optional[MCPClient] = None,
        llm: Optional[OllamaLLM] = None,
        max_iterations: int = MAX_AGENT_ITERATIONS,
    ):
        self.mcp_client = mcp_client or MCPClient()
        self.llm = llm or OllamaLLM()
        self.max_iterations = max_iterations

    def _summarize_observation(self, tool_name: str, result: Dict[str, Any]) -> str:
        """Create a concise, informative observation summary from tool result."""
        if result.get("status") == "error":
            return f"Error ({result.get('error_type', 'ExecutionError')}): {result.get('message', 'Unknown error')}"

        if tool_name == "calculate_metrics":
            top_p = result.get("top_product")
            top_rev = result.get("top_product_revenue", 0.0)
            tot_rev = result.get("total_revenue", 0.0)
            records = result.get("total_records", 0)
            rankings = result.get("product_rankings", [])
            summary = f"Total revenue: ${tot_rev:,.2f} across {records} orders. Top product: '{top_p}' with ${top_rev:,.2f} revenue."
            if len(rankings) > 1:
                top_3 = [f"{r['product']}: ${r['total_revenue']:,.2f}" for r in rankings[:3]]
                summary += f" Top 3: [{', '.join(top_3)}]."
            return summary

        elif tool_name == "compare_periods":
            comp = result.get("comparison", {})
            p_a = result.get("period_a", {})
            p_b = result.get("period_b", {})
            rev_a = p_a.get("total_revenue", 0.0)
            rev_b = p_b.get("total_revenue", 0.0)
            growth = comp.get("percentage_growth")
            interp = comp.get("interpretation", "")
            return f"Period A revenue: ${rev_a:,.2f}. Period B revenue: ${rev_b:,.2f}. Comparison: {interp}"

        elif tool_name == "get_sales_data":
            count = result.get("total_records", 0)
            records = result.get("records", [])
            if not records:
                return "0 records found matching criteria."
            sample_products = list({r.get("product") for r in records[:5]})
            return f"{count} records found. Products included: {sample_products}."

        return json.dumps(result)[:250]

    async def run(self, query: str) -> AgentResponse:
        """Execute the dynamic agent loop for the provided natural language query."""
        if not self.mcp_client.is_connected:
            await self.mcp_client.connect()

        tools = self.mcp_client.get_tool_definitions()
        if not tools:
            tools = await self.mcp_client.discover_tools()

        valid_tool_names = {t["name"] for t in tools}
        history: List[ExecutionStep] = []
        seen_calls: Set[str] = set()

        for iteration in range(1, self.max_iterations + 1):
            try:
                decision = self.llm.get_decision(
                    user_query=query,
                    tools=tools,
                    history=history,
                )
            except Exception as e:
                step = ExecutionStep(
                    step_number=iteration,
                    decision=AgentDecision(
                        action="clarify",
                        rationale="Encountered an unexpected error generating agent decision.",
                        question=f"Execution halted: {str(e)}",
                    ),
                    is_error=True,
                    observation_summary=f"LLM decision failure: {str(e)}",
                )
                history.append(step)
                return AgentResponse(
                    query=query,
                    status="error",
                    error_message=str(e),
                    steps=history,
                    total_iterations=iteration,
                )

            # Case 1: Final Answer Reached
            if decision.action == "final_answer":
                history.append(
                    ExecutionStep(
                        step_number=iteration,
                        decision=decision,
                        observation_summary="Agent determined that sufficient evidence exists to formulate the final answer.",
                    )
                )
                return AgentResponse(
                    query=query,
                    status="completed",
                    final_answer=decision.answer,
                    steps=history,
                    total_iterations=iteration,
                )

            # Case 2: Clarification Requested
            elif decision.action == "clarify":
                history.append(
                    ExecutionStep(
                        step_number=iteration,
                        decision=decision,
                        observation_summary="Agent requested clarification from the user.",
                    )
                )
                return AgentResponse(
                    query=query,
                    status="clarification_needed",
                    clarification_needed=decision.question,
                    steps=history,
                    total_iterations=iteration,
                )

            # Case 3: Call Tool
            elif decision.action == "call_tool":
                tool_name = decision.tool_name or ""
                tool_args = decision.tool_args or {}

                # Validate tool name
                if tool_name not in valid_tool_names:
                    summary = f"Error: Tool '{tool_name}' does not exist on MCP server. Valid tools: {sorted(list(valid_tool_names))}."
                    history.append(
                        ExecutionStep(
                            step_number=iteration,
                            decision=decision,
                            tool_result={"status": "error", "message": summary},
                            observation_summary=summary,
                            is_error=True,
                        )
                    )
                    continue

                # Cycle / repeated call detection
                call_signature = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
                if call_signature in seen_calls:
                    summary = f"Repeated tool call: '{tool_name}' with identical arguments was already executed. Use previous observation to proceed to final_answer."
                    history.append(
                        ExecutionStep(
                            step_number=iteration,
                            decision=decision,
                            tool_result={"status": "warning", "message": summary},
                            observation_summary=summary,
                            is_error=False,
                        )
                    )
                    continue
                seen_calls.add(call_signature)

                # Genuine MCP tool execution
                tool_result = await self.mcp_client.call_tool(tool_name, tool_args)
                obs_summary = self._summarize_observation(tool_name, tool_result)

                history.append(
                    ExecutionStep(
                        step_number=iteration,
                        decision=decision,
                        tool_result=tool_result,
                        observation_summary=obs_summary,
                        is_error=tool_result.get("status") == "error",
                    )
                )

        # Reached max iterations without convergence
        return AgentResponse(
            query=query,
            status="max_iterations_reached",
            steps=history,
            total_iterations=self.max_iterations,
            error_message=f"Agent exceeded maximum limit of {self.max_iterations} iterations without providing a final answer.",
        )
