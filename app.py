"""Agentic DataPilot: Streamlit Web Application.

A genuine local-only agentic dashboard showcasing dynamic MCP tool dispatch,
deterministic grounding verification, and transparent execution traces.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st

from config import DEFAULT_MODEL, OLLAMA_BASE_URL, SALES_DATA_PATH
from agent.core import DataPilotAgent
from agent.llm import OllamaLLM, OllamaUnavailableError
from agent.schemas import AgentResponse
from mcp_server.client import MCPClient


# Page configuration
st.set_page_config(
    page_title="Agentic DataPilot",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    /* Main container styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-bar {
        display: flex;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    .badge {
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }
    .badge-green {
        background-color: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
    }
    .badge-blue {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }
    .badge-purple {
        background-color: #FAF5FF;
        color: #7E22CE;
        border: 1px solid #E9D5FF;
    }

    /* Cards */
    .result-card {
        background-color: #FFFFFF;
        border-radius: 0.75rem;
        padding: 1.25rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .step-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        border-radius: 0 0.5rem 0.5rem 0;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
    }
    .step-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .metric-bubble {
        background: #F1F5F9;
        padding: 0.2rem 0.6rem;
        border-radius: 0.375rem;
        font-family: monospace;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def run_coroutine_in_worker(coro):
    """Run an async coroutine synchronously on a dedicated thread."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


async def discover_tools_async() -> List[Dict[str, Any]]:
    """Discover tools exposed by the local MCP server over stdio."""
    async with MCPClient() as client:
        return await client.discover_tools()


async def execute_agent_query_async(query: str, max_iterations: int = 5) -> Tuple[AgentResponse, List[Dict[str, Any]]]:
    """Execute the DataPilot agent over genuine MCP client and local Ollama."""
    async with MCPClient() as client:
        tools = await client.discover_tools()
        llm = OllamaLLM()
        agent = DataPilotAgent(
            mcp_client=client,
            llm=llm,
            max_iterations=max_iterations,
        )
        response = await agent.run_query(query, tools=tools)
        return response, tools


# Check Ollama and Dataset Availability
ollama_llm = OllamaLLM()
ollama_ok = ollama_llm.check_health()
dataset_exists = Path(SALES_DATA_PATH).exists()


# SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/airplane-take-off.png", width=64)
    st.title("Agentic DataPilot")
    st.caption("L2 MCP-Based Multi-Tool AI Agent")

    st.markdown("---")
    st.subheader("System Status")
    if ollama_ok:
        st.success(f"🟢 Ollama Online (`{DEFAULT_MODEL}`)")
    else:
        st.error("🔴 Ollama Offline (`http://127.0.0.1:11434`)")
        st.caption("Run `ollama serve` in a terminal to start local inference.")

    if dataset_exists:
        st.success(f"🟢 Dataset Loaded (`{Path(SALES_DATA_PATH).name}`)")
    else:
        st.error(f"🔴 Dataset Missing: {SALES_DATA_PATH}")

    st.markdown("---")
    st.subheader("Discovered MCP Tools")
    st.caption("Tools dynamically registered via MCP `stdio` protocol:")

    # Tool discovery
    if "discovered_tools" not in st.session_state:
        try:
            st.session_state.discovered_tools = run_coroutine_in_worker(discover_tools_async())
        except Exception as e:
            st.session_state.discovered_tools = []
            st.warning(f"Could not connect to MCP server: {e}")

    for tool in st.session_state.discovered_tools:
        with st.expander(f"🔧 **{tool['name']}**"):
            st.write(tool.get("description", ""))
            params = tool.get("parameters", {}).get("properties", {})
            if params:
                st.caption("**Parameters:**")
                for p_name, p_info in params.items():
                    req = " *(required)*" if p_name in tool.get("parameters", {}).get("required", []) else ""
                    st.text(f"• {p_name}: {p_info.get('type', 'any')}{req}")

    st.markdown("---")
    st.subheader("Dataset Summary")
    if dataset_exists:
        try:
            df = pd.read_csv(SALES_DATA_PATH)
            st.text(f"Total Rows: {len(df)}")
            st.text(f"Date Range: {df['date'].min()} to {df['date'].max()}")
            st.text(f"Unique Products: {df['product'].nunique()}")
            if st.checkbox("View Raw CSV Sample"):
                st.dataframe(df.head(6), use_container_width=True)
        except Exception as e:
            st.text(f"Error previewing dataset: {e}")

    st.markdown("---")
    max_iter = st.slider("Max Agent Iterations", min_value=2, max_value=8, value=5)


# MAIN CONTENT AREA
st.markdown('<div class="main-header">✈️ Agentic DataPilot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">An MCP-Based Multi-Tool AI Agent for Sales Data Analysis and Grounded Decision-Making</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="badge-bar">
        <span class="badge badge-green">🛡️ 100% Local-Only & Offline</span>
        <span class="badge badge-blue">🔌 Official Python MCP SDK (stdio)</span>
        <span class="badge badge-purple">🧠 Ollama llama3.2:latest</span>
        <span class="badge badge-green">⚖️ Deterministic Grounding Verifier</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Example Queries
st.markdown("##### 💡 Select a benchmark scenario or type your own question:")
col1, col2, col3 = st.columns(3)

default_query = "Which product generated the highest revenue in October, and what was its percentage revenue growth compared with September?"

with col1:
    if st.button("🚀 Scenario 1: Top Product & Growth", use_container_width=True):
        st.session_state["user_input_query"] = (
            "Which product generated the highest revenue in October, and what was its percentage revenue growth compared with September?"
        )

with col2:
    if st.button("⚡ Scenario 2: Zero-Baseline Edge Case", use_container_width=True):
        st.session_state["user_input_query"] = (
            "What was the revenue and percentage growth for SolarKey Mechanical Keyboard between September and October 2024?"
        )

with col3:
    if st.button("📊 Scenario 3: Product Performance", use_container_width=True):
        st.session_state["user_input_query"] = (
            "What was the total revenue and order count for CyberPhone Pro in October 2024?"
        )

initial_text = st.session_state.get("user_input_query", default_query)
user_query = st.text_area(
    "User Query",
    value=initial_text,
    height=85,
    help="Ask a natural-language question about sales figures, products, or monthly comparisons.",
)

run_button = st.button("🚀 Analyze with Agentic DataPilot", type="primary", use_container_width=True)


if run_button:
    if not user_query.strip():
        st.warning("Please enter a question to analyze.")
    elif not ollama_ok:
        st.error(
            "Local Ollama instance is not running on http://127.0.0.1:11434. "
            "Please start Ollama with 'ollama serve' and ensure model 'llama3.2:latest' is available."
        )
    else:
        with st.spinner("Agentic DataPilot is dynamically selecting MCP tools and reasoning..."):
            try:
                response, tools = run_coroutine_in_worker(
                    execute_agent_query_async(user_query, max_iterations=max_iter)
                )
                st.session_state["last_response"] = response
                st.session_state["last_tools"] = tools
            except OllamaUnavailableError as e:
                st.error(f"Ollama Connection Error: {e}")
            except Exception as e:
                st.error(f"Unexpected execution error: {e}")


# DISPLAY RESULTS
if "last_response" in st.session_state:
    resp: AgentResponse = st.session_state["last_response"]

    st.markdown("---")
    st.subheader("🎯 Grounded Result & Verification")

    # 1. Final Answer Card
    st.markdown("### Grounded Final Answer")
    if resp.status == "completed":
        st.success(f"**Verified Answer:** {resp.final_answer}")
    elif resp.status == "unverified":
        st.warning(f"**Unverified Candidate Answer:** {resp.final_answer}")
        st.caption("⚠️ The agent refused to present this candidate answer as authoritative because it could not be 100% verified against evidence.")
    elif resp.status == "clarification_needed":
        st.info(f"**Clarification Requested:** {resp.clarification_needed}")
    else:
        st.error(f"**Execution Halted:** {resp.error_message or 'Max iterations reached without verified conclusion.'}")

    # 2. Deterministic Verification Card
    if resp.verification:
        ver = resp.verification
        is_verified = ver.get("is_verified", False)
        confidence = ver.get("confidence_score", 0.0)

        v_col1, v_col2 = st.columns([1, 3])
        with v_col1:
            if is_verified:
                st.metric("Grounding Status", "VERIFIED ✅", delta="100% Match")
            else:
                st.metric("Grounding Status", "UNVERIFIED ⚠️", delta="-Discrepancy", delta_color="inverse")
            st.metric("Confidence Score", f"{confidence * 100:.1f}%")

        with v_col2:
            st.markdown(f"**Verifier Explanation:** {ver.get('explanation', '')}")
            
            # Verified Claims
            claims = ver.get("verified_claims", [])
            if claims:
                st.markdown("**Grounded Facts Checked:**")
                for c in claims:
                    st.markdown(f"- ✅ `{c}`")

            # Contradictions / Unsupported
            contradictions = ver.get("contradictions", [])
            if contradictions:
                st.markdown("**Contradictions Detected:**")
                for cd in contradictions:
                    st.markdown(f"- ❌ `{cd}`")

            unsupported = ver.get("unsupported_claims", [])
            if unsupported:
                st.markdown("**Unsupported Numbers/Claims:**")
                for u in unsupported:
                    st.markdown(f"- ⚠️ `{u}`")

    # 3. Transparent Execution Trace
    st.markdown("---")
    st.subheader(f"🔍 Transparent Execution Trace ({len(resp.steps)} Steps)")
    st.caption("Step-by-step breakdown of agent decisions, tool invocations, and MCP observations:")

    for step in resp.steps:
        dec = step.decision
        with st.expander(
            f"Step {step.step_number}: {dec.action.upper()}"
            + (f" → `{dec.tool_name}`" if dec.action == 'call_tool' else ""),
            expanded=True,
        ):
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown(f"**Action:** `{dec.action}`")
                st.markdown(f"**Rationale:** *\"{dec.rationale}\"*")
                if dec.action == "call_tool":
                    st.markdown(f"**MCP Tool:** `{dec.tool_name}`")
                    st.json(dec.tool_args or {})
            with col_b:
                if step.observation_summary:
                    st.markdown("**Tool Observation:**")
                    if step.is_error:
                        st.error(step.observation_summary)
                    else:
                        st.info(step.observation_summary)

    # 4. Evidence Store Inspector
    st.markdown("---")
    with st.expander("🗄️ Inspect Evidence Store (Raw Structured Dataset Facts)"):
        if resp.evidence_summary:
            st.text(resp.evidence_summary)
        else:
            st.caption("No evidence records gathered.")
