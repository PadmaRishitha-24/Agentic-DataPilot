"""End-to-End Live Demonstration Runner for Agentic DataPilot.

Executes genuine local MCP server tool discovery and invocation,
orchestrated by the local Ollama LLM (llama3.2:latest), with deterministic
answer verification and grounding.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Line buffer stdout for real-time logs
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from agent.core import DataPilotAgent
from agent.llm import OllamaLLM
from mcp_server.client import MCPClient


async def run_scenario(
    agent: DataPilotAgent,
    query: str,
    scenario_title: str,
    transcript_lines: list,
) -> None:
    """Execute a single end-to-end query scenario and log results."""
    def log(msg: str = ""):
        print(msg)
        transcript_lines.append(msg)

    log("=" * 80)
    log(f"SCENARIO: {scenario_title}")
    log(f"USER QUERY: \"{query}\"")
    log("=" * 80)

    response = await agent.run(query)

    log(f"\nSTATUS: {response.status.upper()}")
    log(f"TOTAL ITERATIONS: {response.total_iterations}")

    log("\n--- AGENT EXECUTION TRACE ---")
    for step in response.steps:
        dec = step.decision
        log(f"\n[Step {step.step_number}] Action: {dec.action.upper()}")
        log(f"  Rationale: {dec.rationale}")
        if dec.action == "call_tool":
            log(f"  Tool Name: {dec.tool_name}")
            log(f"  Tool Args: {dec.tool_args}")
            log(f"  Observation: {step.observation_summary}")
            if step.is_error:
                log("  Flag: [ERROR/WARNING]")

    log("\n--- VERIFICATION RESULT ---")
    if response.verification:
        ver = response.verification
        log(f"  Is Verified: {ver.get('is_verified')}")
        log(f"  Confidence Score: {ver.get('confidence_score')}")
        log(f"  Verified Claims: {ver.get('verified_claims')}")
        if ver.get("contradictions"):
            log(f"  Contradictions: {ver.get('contradictions')}")
        if ver.get("unsupported_claims"):
            log(f"  Unsupported Claims: {ver.get('unsupported_claims')}")
        log(f"  Explanation: {ver.get('explanation')}")
    else:
        log("  No verification record generated.")

    log("\n--- FINAL ANSWER ---")
    log(response.final_answer or "No final answer produced.")
    log("\n" + "=" * 80 + "\n")


async def main():
    transcript: list[str] = [
        "# Agentic DataPilot: Genuine End-to-End Live Demonstration Transcript",
        "**Date & Time:** 2026-10-26",
        "**Environment:** 100% Local (Windows, Ollama llama3.2:latest, Official Python MCP SDK stdio transport)",
        "\n",
    ]

    print("Checking local Ollama status...")
    llm = OllamaLLM()
    if not llm.check_health():
        print("ERROR: Local Ollama daemon is not running on http://127.0.0.1:11434. Please start it with 'ollama serve'.")
        sys.exit(1)

    print("Starting genuine local MCP Client & Server over stdio transport...")
    async with MCPClient() as mcp_client:
        tools = mcp_client.get_tool_definitions()
        print(f"Discovered {len(tools)} MCP Tools:")
        for t in tools:
            print(f" - {t['name']}: {t['description']}")

        transcript.append(f"### Discovered MCP Tools ({len(tools)}):")
        for t in tools:
            transcript.append(f"- **`{t['name']}`**: {t['description']}")
        transcript.append("\n")

        agent = DataPilotAgent(mcp_client=mcp_client, llm=llm)

        # Scenario 1: Required Benchmark Question
        await run_scenario(
            agent=agent,
            query="Which product generated the highest revenue in October, and what was its percentage revenue growth compared with September?",
            scenario_title="Scenario 1: Benchmark Core Question (Top Product & Period Growth)",
            transcript_lines=transcript,
        )

        # Scenario 2: Partial Data / Zero-Baseline Edge Case
        await run_scenario(
            agent=agent,
            query="What was the revenue and percentage growth for SolarKey Mechanical Keyboard between September and October 2024?",
            scenario_title="Scenario 2: Zero-Baseline Growth Edge Case",
            transcript_lines=transcript,
        )

    # Save transcript to file
    out_file = PROJECT_ROOT / "demo_transcript.md"
    out_file.write_text("\n".join(transcript), encoding="utf-8")
    print(f"\nLive demonstration completed successfully. Transcript saved to: {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
