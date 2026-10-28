"""Dynamic Agent Loop for Agentic DataPilot.

Orchestrates local LLM decision-making, genuine MCP tool invocation,
observation feedback, cycle detection, and iteration limits.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from config import MAX_AGENT_ITERATIONS, MAX_CORRECTION_ATTEMPTS
from mcp_server.client import MCPClient
from agent.evidence import EvidenceRecord
from agent.llm import OllamaLLM
from agent.schemas import AgentDecision, AgentResponse, ExecutionStep, extract_requested_product
from agent.verifier import AnswerVerifier


class DataPilotAgent:
    """Autonomous agent driving MCP tool execution via local LLM reasoning."""

    def __init__(
        self,
        mcp_client: Optional[MCPClient] = None,
        llm: Optional[OllamaLLM] = None,
        max_iterations: int = MAX_AGENT_ITERATIONS,
        max_corrections: int = MAX_CORRECTION_ATTEMPTS,
    ):
        self.mcp_client = mcp_client or MCPClient()
        self.llm = llm or OllamaLLM()
        self.max_iterations = max_iterations
        self.max_corrections = max_corrections

    def _append_step(self, history: List[ExecutionStep], step: ExecutionStep) -> None:
        """Append an execution step with strictly sequential 1-indexed numbering."""
        step.step_number = len(history) + 1
        history.append(step)

    def _validate_tool_arguments(
        self, tool_name: str, tool_args: Dict[str, Any]
    ) -> Optional[str]:
        """Validate tool arguments against schemas before execution."""
        if not isinstance(tool_args, dict):
            return f"Tool arguments must be a dictionary, got {type(tool_args).__name__}."

        # Validate product parameter: must be single string, not a list or list representation
        if "product" in tool_args and tool_args["product"] is not None:
            prod_val = tool_args["product"]
            if isinstance(prod_val, (list, tuple, set, dict)):
                return (
                    f"Invalid argument 'product': expected a single product name string, but received {type(prod_val).__name__}: {prod_val}. "
                    "A list of product names cannot be used as a single product filter. "
                    "To analyze or rank all products in this period, omit the 'product' argument and call 'calculate_metrics'."
                )
            prod_str = str(prod_val).strip()
            if (prod_str.startswith("[") and prod_str.endswith("]")) or (prod_str.startswith("(") and prod_str.endswith(")")):
                return (
                    f"Invalid argument 'product': received a list representation '{prod_str}'. "
                    "The 'product' argument only accepts a single product name string (e.g. 'Quantum Laptop'). "
                    "To analyze or rank all products, omit the 'product' argument and call 'calculate_metrics'."
                )

            prod_clean = prod_str.lower().strip()
            analytical_keywords = {
                "lowest_revenue_product", "highest_revenue_product", "top_product", "lowest_product",
                "bottom_product", "best_product", "most_revenue_product", "least_revenue_product",
                "lowest", "highest", "top", "bottom", "all", "none", "null", "undefined",
            }
            if prod_clean in analytical_keywords or any(term in prod_clean for term in ["lowest_revenue", "highest_revenue", "top_product", "lowest_product"]):
                return (
                    f"Invalid argument 'product': '{prod_val}' is an analytical operation, not a literal product name. "
                    "Values like 'lowest_revenue_product' or 'highest_revenue_product' cannot be used as product filters. "
                    "To obtain product rankings or find the lowest/highest revenue product, call 'calculate_metrics' without a product filter."
                )

        # Validate date fields if present
        for date_key in ["start_date", "end_date", "period_a_start", "period_a_end", "period_b_start", "period_b_end"]:
            if date_key in tool_args and tool_args[date_key] is not None:
                val = tool_args[date_key]
                if not isinstance(val, str):
                    return f"Invalid argument '{date_key}': expected string in YYYY-MM-DD format, got {type(val).__name__}."

        return None

    def _has_sufficient_evidence(self, query: str, evidence: EvidenceRecord) -> bool:
        """Check whether accumulated tool observations are sufficient to answer the query."""
        if not evidence.facts and not evidence.raw_observations:
            return False

        q_lower = query.lower()

        # If profit margin, profit, or cost is requested (unavailable metrics in dataset)
        profit_or_cost_needed = bool(
            re.search(r'\b(profit\s*margin|profit|margin|costs?|cogs|expenses?|net\s*income)\b', q_lower)
        )
        if profit_or_cost_needed:
            # Once a tool has inspected the sales data, the agent has confirmed available fields
            # and has sufficient empirical evidence to formulate the verified limitation.
            return bool(evidence.facts or evidence.raw_observations)

        # If percentage growth or period comparison is requested
        growth_needed = any(w in q_lower for w in ["growth", "compare", "percentage"])
        if growth_needed:
            return any(f.fact_type == "percentage_growth" for f in evidence.facts)

        # Check if full ranking of all products is requested
        rank_all_needed = bool(
            any(phrase in q_lower for phrase in [
                "rank all", "ranking all", "rank the products", "rank products", "ranking of all",
                "show all products", "all products by", "each product", "all products ranked",
                "list all products", "product rankings", "order all products",
                "highest to lowest", "lowest to highest"
            ]) or (
                any(w in q_lower for w in ["rank", "ranking", "rankings", "order"])
                and any(w in q_lower for w in ["all", "every", "each", "products"])
            )
        )
        if rank_all_needed:
            for obs in evidence.raw_observations:
                if obs.get("tool_name") == "calculate_metrics":
                    res = obs.get("tool_result", {})
                    if res.get("product_rankings") and len(res["product_rankings"]) >= 2:
                        return True
            return False

        # If lowest product is requested (single extreme)
        lowest_needed = bool(re.search(r'\b(lowest|least|bottom|worst|smallest|minimum)\b', q_lower))
        if lowest_needed:
            return any(f.fact_type == "lowest_product" for f in evidence.facts)

        # If top product is requested (single extreme)
        top_needed = bool(
            re.search(r'\b(top|highest|best|leader|first)\b|\bmost revenue\b|\bmaximum revenue\b|\bgreatest revenue\b|\brank\s*1\b|\b#1\b', q_lower)
        )
        if top_needed and not lowest_needed:
            return any(f.fact_type == "top_product" for f in evidence.facts)

        # If asking about a specific product (known or unknown)
        target_product = extract_requested_product(query)
        if target_product and not lowest_needed and not top_needed:
            has_product_fact = any(
                f.fact_type in ("product_revenue", "product_zero_records")
                and target_product.lower() in f.subject.lower()
                for f in evidence.facts
            )
            has_rankings = any(
                obs.get("tool_name") == "calculate_metrics" and obs.get("tool_result", {}).get("product_rankings")
                for obs in evidence.raw_observations
            )
            has_prod_obs = any(
                obs.get("tool_args", {}).get("product", "").lower() == target_product.lower()
                for obs in evidence.raw_observations
            )
            if has_product_fact or has_prod_obs or has_rankings:
                return True

        # If asking about total revenue across all products or overall period revenue
        total_rev_needed = any(
            phrase in q_lower for phrase in [
                "total revenue", "total sales", "overall revenue", "overall sales",
                "revenue across all", "sales across all", "how much revenue", "how much was made",
                "what was the revenue", "what was the total", "revenue in", "sales in"
            ]
        )
        if total_rev_needed and not growth_needed and not lowest_needed and not top_needed and not target_product:
            return any(f.fact_type == "total_revenue" for f in evidence.facts)

        return False

    def _summarize_observation(self, tool_name: str, result: Dict[str, Any]) -> str:
        """Create a concise, informative observation summary from tool result."""
        if result.get("status") == "error":
            return f"Error ({result.get('error_type', 'ExecutionError')}): {result.get('message', 'Unknown error')}"

        if tool_name == "calculate_metrics":
            top_p = result.get("top_product")
            top_rev = result.get("top_product_revenue", 0.0)
            low_p = result.get("lowest_product")
            low_rev = result.get("lowest_product_revenue", 0.0)
            tot_rev = result.get("total_revenue", 0.0)
            records = result.get("total_records", 0)
            rankings = result.get("product_rankings", [])
            summary = f"Total revenue: ${tot_rev:,.2f} across {records} orders. Top product: '{top_p}' with ${top_rev:,.2f} revenue."
            if low_p and low_p != top_p:
                summary += f" Lowest product: '{low_p}' with ${low_rev:,.2f} revenue."
            if rankings:
                ranks_str = [f"{r['product']}: ${r['total_revenue']:,.2f}" for r in rankings]
                summary += f" Product rankings: [{', '.join(ranks_str)}]."
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
            all_products = sorted(list({r.get("product") for r in records if r.get("product")}))
            return (
                f"{count} records found across products: {all_products}. "
                "Note: get_sales_data returns individual unaggregated order rows. "
                "For grouped product rankings, totals, or top/lowest revenue products, call calculate_metrics without a product filter."
            )

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
        repeat_counts: Dict[str, int] = {}
        evidence = EvidenceRecord()

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
                self._append_step(history, step)
                return AgentResponse(
                    query=query,
                    status="error",
                    error_message=str(e),
                    steps=history,
                    total_iterations=iteration,
                    evidence_summary=evidence.get_summary_text(),
                )

            # Case 1: Final Answer Reached
            if decision.action == "final_answer":
                candidate_answer = decision.answer or ""
                verification = AnswerVerifier.verify(
                    candidate_answer=candidate_answer,
                    evidence=evidence,
                    user_query=query,
                )

                # Bounded correction attempt if verification failed
                if not verification.is_verified and self.max_corrections > 0:
                    self._append_step(
                        history,
                        ExecutionStep(
                            step_number=iteration,
                            decision=decision,
                            observation_summary=f"Initial answer failed verification: {verification.explanation}. Triggering bounded correction.",
                            is_error=True,
                        ),
                    )

                    if hasattr(self.llm, "get_corrected_answer"):
                        corrected_answer = self.llm.get_corrected_answer(
                            user_query=query,
                            candidate_answer=candidate_answer,
                            feedback=verification.feedback_for_correction or "",
                        )
                    else:
                        corrected_answer = candidate_answer

                    re_ver = AnswerVerifier.verify(
                        candidate_answer=corrected_answer,
                        evidence=evidence,
                        user_query=query,
                    )

                    if re_ver.is_verified:
                        self._append_step(
                            history,
                            ExecutionStep(
                                step_number=iteration,
                                decision=AgentDecision(
                                    action="final_answer",
                                    rationale="Corrected candidate answer based on verified ground truth.",
                                    answer=corrected_answer,
                                ),
                                observation_summary="Answer successfully corrected and verified against evidence.",
                            ),
                        )
                        return AgentResponse(
                            query=query,
                            status="completed",
                            final_answer=corrected_answer,
                            verification=re_ver.model_dump(),
                            evidence_summary=evidence.get_summary_text(),
                            steps=history,
                            total_iterations=iteration,
                        )
                    else:
                        # Refuse to present as verified if correction still failed
                        self._append_step(
                            history,
                            ExecutionStep(
                                step_number=iteration,
                                decision=AgentDecision(
                                    action="final_answer",
                                    rationale="Candidate answer remained unverified after correction attempt.",
                                    answer=corrected_answer,
                                ),
                                observation_summary=f"Answer remains unverified: {re_ver.explanation}",
                                is_error=True,
                            ),
                        )
                        return AgentResponse(
                            query=query,
                            status="unverified",
                            final_answer=corrected_answer,
                            verification=re_ver.model_dump(),
                            evidence_summary=evidence.get_summary_text(),
                            steps=history,
                            total_iterations=iteration,
                            error_message=f"Answer could not be fully verified against evidence: {re_ver.explanation}",
                        )

                # Answer passed verification immediately
                self._append_step(
                    history,
                    ExecutionStep(
                        step_number=iteration,
                        decision=decision,
                        observation_summary="Answer successfully formulated and verified against tool evidence.",
                    ),
                )
                return AgentResponse(
                    query=query,
                    status="completed" if verification.is_verified else "unverified",
                    final_answer=candidate_answer,
                    verification=verification.model_dump(),
                    evidence_summary=evidence.get_summary_text(),
                    steps=history,
                    total_iterations=iteration,
                )

            # Case 2: Clarification Requested
            elif decision.action == "clarify":
                self._append_step(
                    history,
                    ExecutionStep(
                        step_number=iteration,
                        decision=decision,
                        observation_summary="Agent requested clarification from the user.",
                    ),
                )
                return AgentResponse(
                    query=query,
                    status="clarification_needed",
                    clarification_needed=decision.question,
                    evidence_summary=evidence.get_summary_text(),
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
                    self._append_step(
                        history,
                        ExecutionStep(
                            step_number=iteration,
                            decision=decision,
                            tool_result={"status": "error", "message": summary},
                            observation_summary=summary,
                            is_error=True,
                        ),
                    )
                    continue

                # Validate tool arguments against schema
                arg_error = self._validate_tool_arguments(tool_name, tool_args)
                if arg_error:
                    summary = f"Argument Error: {arg_error}"
                    self._append_step(
                        history,
                        ExecutionStep(
                            step_number=iteration,
                            decision=decision,
                            tool_result={"status": "error", "message": summary},
                            observation_summary=summary,
                            is_error=True,
                        ),
                    )
                    # If sufficient evidence is already accumulated, attempt final answer formulation
                    if self._has_sufficient_evidence(query, evidence):
                        try:
                            synth_step = ExecutionStep(
                                step_number=iteration,
                                decision=AgentDecision(
                                    action="final_answer",
                                    rationale="Formulating final answer as sufficient verified evidence is already present.",
                                ),
                                observation_summary="All required evidence is already observed. Formulating final answer.",
                            )
                            final_dec = self.llm.get_decision(
                                user_query=query,
                                tools=[],
                                history=history + [synth_step],
                            )
                            if final_dec.action == "final_answer" and final_dec.answer:
                                ver = AnswerVerifier.verify(
                                    candidate_answer=final_dec.answer,
                                    evidence=evidence,
                                    user_query=query,
                                )
                                if ver.is_verified:
                                    self._append_step(
                                        history,
                                        ExecutionStep(
                                            step_number=iteration,
                                            decision=final_dec,
                                            observation_summary="Answer formulated from verified evidence observations.",
                                        ),
                                    )
                                    return AgentResponse(
                                        query=query,
                                        status="completed",
                                        final_answer=final_dec.answer,
                                        verification=ver.model_dump(),
                                        evidence_summary=evidence.get_summary_text(),
                                        steps=history,
                                        total_iterations=iteration,
                                    )
                        except Exception:
                            pass
                    continue

                # Cycle / repeated call detection
                call_signature = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
                if call_signature in seen_calls:
                    repeat_counts[call_signature] = repeat_counts.get(call_signature, 0) + 1

                    prev_obs = None
                    for prev_s in reversed(history):
                        if prev_s.decision.tool_name == tool_name and not prev_s.is_error and prev_s.observation_summary:
                            prev_obs = prev_s.observation_summary
                            break

                    summary = (
                        f"Repeated tool call: '{tool_name}' with identical arguments was already executed. "
                        + (f"Previous observation was: {prev_obs}. " if prev_obs else "")
                        + "Do not repeat identical tool calls. "
                        "The next decision must either use existing evidence to answer, make a genuinely necessary different call, or transition to 'final_answer'."
                    )
                    self._append_step(
                        history,
                        ExecutionStep(
                            step_number=iteration,
                            decision=decision,
                            tool_result={"status": "warning", "message": summary},
                            observation_summary=summary,
                            is_error=True,
                        ),
                    )
                    # If sufficient evidence is already accumulated, formulate final answer
                    if self._has_sufficient_evidence(query, evidence):
                        try:
                            synth_step = ExecutionStep(
                                step_number=iteration,
                                decision=AgentDecision(
                                    action="final_answer",
                                    rationale="Formulating final answer as sufficient verified evidence is already present.",
                                ),
                                observation_summary="All required evidence is already observed. Formulating final answer.",
                            )
                            final_dec = self.llm.get_decision(
                                user_query=query,
                                tools=[],
                                history=history + [synth_step],
                            )
                            if final_dec.action == "final_answer" and final_dec.answer:
                                ver = AnswerVerifier.verify(
                                    candidate_answer=final_dec.answer,
                                    evidence=evidence,
                                    user_query=query,
                                )
                                if ver.is_verified:
                                    self._append_step(
                                        history,
                                        ExecutionStep(
                                            step_number=iteration,
                                            decision=final_dec,
                                            observation_summary="Answer formulated from verified evidence after repeated call was intercepted.",
                                        ),
                                    )
                                    return AgentResponse(
                                        query=query,
                                        status="completed",
                                        final_answer=final_dec.answer,
                                        verification=ver.model_dump(),
                                        evidence_summary=evidence.get_summary_text(),
                                        steps=history,
                                        total_iterations=iteration,
                                    )
                        except Exception:
                            pass

                    # If this exact call was repeated multiple times despite warning, do not let it consume all iterations
                    if repeat_counts[call_signature] >= 2:
                        if evidence.facts:
                            try:
                                synth_step = ExecutionStep(
                                    step_number=iteration,
                                    decision=AgentDecision(
                                        action="final_answer",
                                        rationale="Synthesizing final answer from available evidence after repeated tool calls.",
                                    ),
                                    observation_summary="Synthesizing final answer from established observations.",
                                )
                                final_dec = self.llm.get_decision(
                                    user_query=query,
                                    tools=[],
                                    history=history + [synth_step],
                                )
                                if final_dec.action == "final_answer" and final_dec.answer:
                                    ver = AnswerVerifier.verify(
                                        candidate_answer=final_dec.answer,
                                        evidence=evidence,
                                        user_query=query,
                                    )
                                    if ver.is_verified:
                                        self._append_step(
                                            history,
                                            ExecutionStep(
                                                step_number=iteration,
                                                decision=final_dec,
                                                observation_summary="Answer formulated after repeated call loop was terminated.",
                                            ),
                                        )
                                        return AgentResponse(
                                            query=query,
                                            status="completed",
                                            final_answer=final_dec.answer,
                                            verification=ver.model_dump(),
                                            evidence_summary=evidence.get_summary_text(),
                                            steps=history,
                                            total_iterations=iteration,
                                        )
                            except Exception:
                                pass

                        return AgentResponse(
                            query=query,
                            status="error",
                            steps=history,
                            total_iterations=iteration,
                            evidence_summary=evidence.get_summary_text(),
                            error_message=f"Agent safely stopped: repeated identical tool call '{tool_name}' was requested multiple times without gathering new data.",
                        )
                    continue
                seen_calls.add(call_signature)

                # Genuine MCP tool execution
                tool_result = await self.mcp_client.call_tool(tool_name, tool_args)
                obs_summary = self._summarize_observation(tool_name, tool_result)

                # Record fact in evidence store
                evidence.add_observation(tool_name, tool_args, tool_result)

                self._append_step(
                    history,
                    ExecutionStep(
                        step_number=iteration,
                        decision=decision,
                        tool_result=tool_result,
                        observation_summary=obs_summary,
                        is_error=tool_result.get("status") == "error",
                    ),
                )

        # Check if sufficient observations exist to produce a grounded final answer
        if evidence.facts:
            try:
                synthesis_step = ExecutionStep(
                    step_number=self.max_iterations,
                    decision=AgentDecision(
                        action="final_answer",
                        rationale="Synthesizing final answer from established evidence observations.",
                    ),
                    observation_summary="Iteration budget exhausted. Synthesizing final answer from established observations.",
                )
                final_decision = self.llm.get_decision(
                    user_query=query,
                    tools=[],
                    history=history + [synthesis_step],
                )
                if final_decision.action == "final_answer" and final_decision.answer:
                    verification = AnswerVerifier.verify(
                        candidate_answer=final_decision.answer,
                        evidence=evidence,
                        user_query=query,
                    )
                    if verification.is_verified:
                        self._append_step(
                            history,
                            ExecutionStep(
                                step_number=self.max_iterations,
                                decision=final_decision,
                                observation_summary="Answer formulated from evidence upon iteration limit and successfully verified.",
                            ),
                        )
                        return AgentResponse(
                            query=query,
                            status="completed",
                            final_answer=final_decision.answer,
                            verification=verification.model_dump(),
                            evidence_summary=evidence.get_summary_text(),
                            steps=history,
                            total_iterations=self.max_iterations,
                        )
            except Exception:
                pass

        # Reached max iterations without convergence
        return AgentResponse(
            query=query,
            status="max_iterations_reached",
            steps=history,
            total_iterations=self.max_iterations,
            evidence_summary=evidence.get_summary_text(),
            error_message=(
                f"Agent exceeded maximum limit of {self.max_iterations} iterations without providing a final answer. "
                f"Evidence gathered: {evidence.get_summary_text()}"
            ),
        )

    async def run_query(
        self,
        query: str,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentResponse:
        """Execute a user query (compatibility alias for run)."""
        return await self.run(query)

