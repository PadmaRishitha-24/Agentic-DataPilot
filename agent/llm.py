"""Local Ollama client for structured agent decision generation."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import httpx

from config import DEFAULT_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT_SECONDS
from agent.schemas import AgentDecision, ExecutionStep


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama service cannot be reached."""
    pass


class OllamaLLM:
    """Interface to local Ollama instance for structured JSON agent decisions."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = OLLAMA_TIMEOUT_SECONDS,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def check_health(self) -> bool:
        """Check if local Ollama daemon is active and responsive."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def _build_prompt(
        self,
        user_query: str,
        tools: List[Dict[str, Any]],
        history: List[ExecutionStep],
    ) -> str:
        """Construct prompt incorporating discovered MCP tools and historical observations."""
        tools_desc = []
        for t in tools:
            name = t.get("name", "unknown")
            desc = t.get("description", "")
            params = t.get("parameters", {}).get("properties", {})
            param_list = ", ".join(f"{p}: {info.get('type', 'any')}" for p, info in params.items())
            tools_desc.append(f"- {name}({param_list}): {desc}")

        tools_block = "\n".join(tools_desc)

        history_lines = []
        for step in history:
            dec = step.decision
            if dec.action == "call_tool":
                history_lines.append(
                    f"Step {step.step_number}: Called tool '{dec.tool_name}' with args {json.dumps(dec.tool_args)}."
                )
                if step.observation_summary:
                    history_lines.append(f"  Observation: {step.observation_summary}")

        history_block = (
            "\n".join(history_lines)
            if history_lines
            else "No tools have been called yet."
        )

        prompt = f"""You are Agentic DataPilot, an intelligent, deterministic data analysis agent.
Your objective is to answer the user's question accurately using grounded evidence from the available local tools.

AVAILABLE MCP TOOLS:
{tools_block}

CURRENT EXECUTION HISTORY & OBSERVATIONS:
{history_block}

USER QUESTION:
"{user_query}"

INSTRUCTIONS:
1. If you need more information from the dataset, choose action "call_tool", specify the exact tool_name, and provide the exact arguments.
2. If you have gathered all necessary numerical evidence and observations to answer the user's question fully, choose action "final_answer" and provide a clear, factual answer stating the exact metrics and observations.
3. If the user question is genuinely ambiguous or missing critical context, choose action "clarify" and provide your clarifying question.
4. Never invent or hallucinate metrics or product numbers. All claims must come directly from tool observations.
5. Provide a concise 1-sentence rationale explaining your choice. Do NOT reveal internal chain-of-thought.

You MUST respond ONLY with a valid JSON object matching this schema:
{{
  "action": "call_tool" | "final_answer" | "clarify",
  "rationale": "1-sentence explanation of why this action was taken",
  "tool_name": "tool_name_here (required if action is call_tool)",
  "tool_args": {{"arg_name": "arg_value"}} (required if action is call_tool),
  "answer": "Complete final answer with verified metrics (required if action is final_answer)",
  "question": "Clarification question (required if action is clarify)"
}}
"""
        return prompt

    def get_decision(
        self,
        user_query: str,
        tools: List[Dict[str, Any]],
        history: List[ExecutionStep],
    ) -> AgentDecision:
        """Query local Ollama and return a validated AgentDecision."""
        prompt = self._build_prompt(user_query, tools, history)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for deterministic adherence
            },
        }

        try:
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
        except httpx.ConnectError as e:
            raise OllamaUnavailableError(
                f"Cannot connect to local Ollama at {self.base_url}. Ensure Ollama is running ('ollama serve')."
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaUnavailableError(
                f"Ollama request timed out after {self.timeout} seconds."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Error querying local Ollama: {str(e)}") from e

        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama API returned error HTTP {resp.status_code}: {resp.text}"
            )

        raw_response = resp.json().get("response", "").strip()
        if not raw_response:
            raise ValueError("Ollama returned an empty response.")

        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Ollama output as JSON: {raw_response}") from e

        # Normalize common LLM output variations
        action = data.get("action", "").lower().strip()
        if action not in ["call_tool", "final_answer", "clarify"]:
            # If tool_name is present, assume call_tool
            if data.get("tool_name"):
                data["action"] = "call_tool"
            elif data.get("answer"):
                data["action"] = "final_answer"
            elif data.get("question"):
                data["action"] = "clarify"
            else:
                data["action"] = "call_tool"

        if not data.get("rationale"):
            data["rationale"] = f"Proceeding with {data['action']} based on current observations."

        # If product was hallucinated as placeholder like 'product_name', clean it up
        tool_args = data.get("tool_args") or {}
        if isinstance(tool_args, dict):
            cleaned_args = {}
            for k, v in tool_args.items():
                if isinstance(v, str) and v.lower() in [
                    "product_name",
                    "none",
                    "all",
                    "null",
                    "",
                ]:
                    continue  # Omit dummy placeholder strings
                cleaned_args[k] = v
            data["tool_args"] = cleaned_args

        return AgentDecision.model_validate(data)
