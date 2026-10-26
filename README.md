# Agentic DataPilot: An MCP-Based Multi-Tool AI Agent for Data Analysis and Grounded Decision-Making

> **Level 2 (L2) Agentic AI Engineering Project**  
> **Environment:** 100% Local (Windows 11, Python 3.12, Local Ollama `llama3.2:latest`, Official Python MCP SDK)  
> **Guarantee:** Completely offline, zero cloud endpoints, zero external API keys, zero remote inference.

---

## 1. Executive Summary

**Agentic DataPilot** is a genuine, local-first agentic AI system designed to answer complex, natural-language analytical questions over sales records. Rather than relying on a cloud LLM to perform fragile mental arithmetic or hallucinate numbers, DataPilot operates as an autonomous agent that:

1. **Dynamically Discovers Local Tools** via the official Model Context Protocol (MCP) over standard input/output (`stdio`) transport.
2. **Plans & Executes Multi-Step Investigations** using a locally hosted small language model (`llama3.2:latest` via Ollama) with structured JSON decisions.
3. **Performs Deterministic Data Operations** (record filtering, aggregations, product rankings, and period-over-period comparisons) in clean Python/Pandas logic.
4. **Maintains an Evidence Store** indexing every observation and raw dataset metric gathered during execution.
5. **Enforces Deterministic Grounding Verification**: Independently recalculates figures and verifies that all product names, revenue numbers, and percentage growths strictly match tool observations before presenting answers.
6. **Employs Bounded Reflection & Correction**: If candidate answers contain unsupported figures or contradictions, the system triggers targeted correction; if an answer cannot be grounded, it explicitly refuses to guess and marks the response as unverified.

---

## 2. Mandatory 100% Local-Only & Offline Guarantee

This project runs **entirely on your local Windows machine**:
- **Local LLM Engine**: Powered by [Ollama](https://ollama.com/) running model `llama3.2:latest` (3.2B parameters, ~2.0 GB) on `http://127.0.0.1:11434`.
- **Local MCP Architecture**: Communicates with an internal MCP server process over standard input/output (`stdio`), requiring no hosted MCP endpoints or remote servers.
- **Local Data Storage**: Reads from a bundled local CSV dataset (`data/sales_data.csv`).
- **No Cloud Dependencies**: No OpenAI, Anthropic, Gemini, or Azure APIs. No API keys, cloud databases, paid subscriptions, or telemetry.
- **Strict Isolation**: Runs completely isolated in its own workspace; never touches or modifies any other project.

---

## 3. System Architecture

```
                          +-------------------------------------------------------+
                          |                   User Interface                      |
                          |  - Streamlit Web Dashboard (app.py)                   |
                          |  - CLI Automated Demonstration (demo.py)              |
                          +---------------------------+---------------------------+
                                                      |
                                             Natural Language Query
                                                      v
                          +-------------------------------------------------------+
                          |                 Agent Orchestrator                    |
                          |                  (agent/core.py)                      |
                          |  - Multi-Step Dynamic Reasoning Loop (Max 5 steps)    |
                          |  - Cycle & Repeated Call Detection Guard              |
                          |  - Bounded Self-Correction Trigger                    |
                          +-----------+-------------------------------+-----------+
                                      |                               ^
                   Structured Decisions & Prompts           Observations
                                      v                               |
+-------------------------------------+--------+  +-------------------+-------------------+
|         Local LLM Engine                     |  |          Local MCP Client             |
|          (agent/llm.py)                      |  |      (mcp_server/client.py)           |
|                                              |  |                                       |
| - Local Ollama Daemon (http://127.0.0.1:11434)|  | - Official Python MCP SDK             |
| - Model: llama3.2:latest (Local 3.2B)        |  | - JSON-RPC over stdio Transport       |
| - Strict JSON Mode (AgentDecision schema)    |  | - Dynamic Tool Discovery              |
+----------------------------------------------+  +-------------------+-------------------+
                                                                      |
                                                          JSON-RPC over stdio
                                                                      v
                                                  +---------------------------------------+
                                                  |           Local MCP Server            |
                                                  |        (mcp_server/server.py)         |
                                                  |                                       |
                                                  | - Tool: get_sales_data                |
                                                  | - Tool: calculate_metrics             |
                                                  | - Tool: compare_periods               |
                                                  +-------------------+-------------------+
                                                                      |
                                                              Python Operations
                                                                      v
                                                  +---------------------------------------+
                                                  |          Sales Analysis Engine        |
                                                  |         (tools/sales_tools.py)        |
                                                  |                                       |
                                                  | - Deterministic Aggregations & Math   |
                                                  | - Zero-Baseline Protection            |
                                                  | - CSV Validation (data/sales_data.csv)|
                                                  +-------------------+-------------------+
                                                                      |
                                                       Populates Observed Facts
                                                                      v
                          +---------------------------------------------------------------+
                          |                 Evidence Store & Fact Index                   |
                          |                     (agent/evidence.py)                       |
                          |  - Tracks raw tool results, timestamps, and numeric facts     |
                          +-------------------------------+-------------------------------+
                                                          |
                                           Evidence & Candidate Answer
                                                          v
                          +---------------------------------------------------------------+
                          |               Deterministic Answer Verifier                   |
                          |                    (agent/verifier.py)                        |
                          |                                                               |
                          | - Independent Math & Metric Recalculation                     |
                          | - Entity & Ranking Check (Quantum Laptop vs Others)          |
                          | - Percentage & Revenue Grounding Validation                  |
                          | - Contradiction & Unsupported Claim Detection                 |
                          +-------------------------------+-------------------------------+
                                                          |
                                          Verified Result (Score: 1.0)
                                                          v
                                        Authoritative Grounded Answer
```

---

## 4. Key Components

### 4.1 Local MCP Server & Client (`mcp_server/`)
- Built using the official Python Model Context Protocol SDK (`mcp` v2.3+).
- **Server (`server.py`)**: Runs an in-process MCP server exposing three analytical tools. It communicates over standard input/output (`stdio`), completely offline.
- **Client (`client.py`)**: Asynchronously launches the server subprocess, performs tool discovery via `session.list_tools()`, and converts MCP results into clean Python dictionaries.

### 4.2 Sales Analysis Tools (`tools/sales_tools.py`)
- **`get_sales_data(start_date, end_date, product)`**: Filters local CSV sales records with input validation and case-insensitive matching.
- **`calculate_metrics(start_date, end_date, product)`**: Deterministically computes total revenue, order count, total units sold, and ranked products.
- **`compare_periods(period_a_start, period_a_end, period_b_start, period_b_end, product)`**: Calculates revenue change ($) and percentage growth (%) between two distinct date ranges. Handles edge cases:
  - When baseline revenue is `$0.00`, percentage growth is flagged as `null` with `zero_baseline_detected: True` to prevent mathematical zero-division errors.

### 4.3 Agent Decision Loop (`agent/core.py`, `agent/llm.py`)
- Employs a structured JSON schema (`AgentDecision`) with three possible actions:
  - `call_tool`: Dispatch an MCP tool with specific arguments.
  - `final_answer`: Present the conclusion with verified metrics.
  - `clarify`: Request missing parameters from the user.
- Enforces defensive guards:
  - Prevents infinite loops through cycle detection (`seen_calls` tracker).
  - Normalizes aliases generated by small LLMs (e.g. `product_name` -> `product`).
  - Limits execution to a maximum of 5 iterations.

### 4.4 Deterministic Grounding Verifier (`agent/verifier.py`)
- Audits the LLM's final answer against the `EvidenceRecord`.
- Extracts all numerical values, dollar figures, percentages, and entity names from the answer.
- Checks whether:
  1. The named top product matches the tool observation.
  2. Every revenue figure exists in observed tool outputs.
  3. Every percentage matches deterministic calculations within rounding tolerance.
  4. Any claimed number is ungrounded or contradictory.
- Calculates an objective **Confidence Score** ($0.0$ to $1.0$).
- If discrepancies exist, triggers bounded reflection (1 retry) or marks the response as `unverified`.

---

## 5. Dataset Ground Truth & Benchmark Queries

The dataset (`data/sales_data.csv`) contains 26 realistic transaction records across September, October, and November 2024:

| Month | Top Product | Top Product Revenue | Total Monthly Revenue | Orders |
| :--- | :--- | :--- | :--- | :--- |
| **September 2024** | Quantum Laptop | $44,400.00 | $107,400.00 | 11 |
| **October 2024** | **Quantum Laptop** | **$68,400.00** | **$139,200.00** | **12** |
| **November 2024** | CloudServer Blade | $18,000.00 | $26,500.00 | 3 |

### Benchmark Question
> **"Which product generated the highest revenue in October, and what was its percentage revenue growth compared with September?"**
- **October Top Product:** `Quantum Laptop` with **$68,400.00** revenue.
- **September Baseline:** `Quantum Laptop` with **$44,400.00** revenue.
- **Absolute Growth:** `+$24,000.00`.
- **Percentage Growth:** **`+54.05%`** ($\frac{68,400 - 44,400}{44,400} \times 100$).

### Zero-Baseline Edge Case
> **"What was the revenue and percentage growth for SolarKey Mechanical Keyboard between September and October 2024?"**
- **September Baseline:** `$0.00` (0 units sold).
- **October Revenue:** `$4,000.00` (40 units sold).
- **Percentage Growth:** **Mathematically Undefined** (reported cleanly without divide-by-zero crashes).

---

## 6. Installation & Setup (Windows PowerShell)

### Prerequisites
1. **Python 3.12** installed on Windows.
2. **Ollama for Windows** installed and running:
   ```powershell
   ollama --version
   ```
3. Pull the required model locally:
   ```powershell
   ollama pull llama3.2:latest
   ```

### 1. Clone & Navigate to Repository
```powershell
cd "C:\Users\PadmaRishitaPothula\Downloads\Agentic DataPilot"
```

### 2. Create and Activate Virtual Environment
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell blocks script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### 3. Install Pinned Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Verify Local Ollama Daemon
Ensure Ollama is running in the background. In PowerShell:
```powershell
curl http://127.0.0.1:11434/api/tags
```
You should see `llama3.2:latest` listed.

---

## 7. How to Run

> **💡 Windows PowerShell Tip:** You can either activate the virtual environment or run directly via `.\.venv\Scripts\`.

### Option 1: Run the Automated Pytest Suite
Execute all 43 unit, integration, decision, verification, and reliability tests:

```powershell
# Directly without script activation (Recommended):
.\.venv\Scripts\pytest -v

# Or after activating the venv:
pytest -v
```
**Expected Output:** `43 passed in ~16s`.

---

### Option 2: Run the Live Command-Line Demonstration
Run the live end-to-end demonstration exercising both the benchmark core question and the zero-baseline edge case:

```powershell
# Directly without script activation:
.\.venv\Scripts\python.exe demo.py

# Or after activating the venv:
python demo.py
```
This prints the complete execution trace, tool arguments, observations, and grounding verification scores in real time, and saves the markdown transcript to `demo_transcript.md`.

---

### Option 3: Launch the Streamlit Web Application
Launch the rich interactive web interface:

```powershell
# Directly without script activation:
.\.venv\Scripts\streamlit.exe run app.py

# Or after activating the venv:
streamlit run app.py
```
Open your browser to: **`http://localhost:8501`**.

#### Streamlit Features:
- **Discovered Tools Sidebar**: Displays dynamically discovered MCP tools and schemas.
- **1-Click Benchmark Buttons**: Instantly pre-populate test scenarios.
- **Visual Step-by-Step Execution Trace**: Inspect rationales, tool calls, JSON arguments, and observations.
- **Deterministic Verification Card**: See the green **VERIFIED** badge, 100% confidence score, and checked numbers.
- **Raw Evidence Store**: Expandable raw structured dataset facts collected during execution.

---

## 8. Reliability & Error Handling

DataPilot includes an automated reliability test suite (`tests/test_reliability.py`) verifying resilient behavior under stress:

1. **Empty File Handling**: When `sales_data.csv` is completely empty (0 bytes), the tools catch `EmptyDatasetError` gracefully.
2. **Corrupted Records**: Malformed rows (negative quantities, unparseable dates, missing prices) are automatically skipped with warning logs without interrupting analysis.
3. **Zero-Division Arithmetic Safety**: When baseline revenue is 0, `compare_periods` sets `percentage_growth: null` and explains the mathematical limitation in plain English.
4. **Ollama Service Downtime**: If Ollama is offline (`ConnectionRefusedError`), DataPilot raises `OllamaUnavailableError` with clear instructions to run `ollama serve`.
5. **LLM Decision Failures**: If the model outputs unparseable JSON or times out, the agent loop records an error step and halts gracefully.
6. **Unknown Tool Requests**: If the LLM requests an unregistered tool, DataPilot reports the error and supplies the list of valid MCP tools.

---

## 9. Project Directory Structure

```
Agentic DataPilot/
├── .gitignore                     # Git ignore rules (.venv, caches, logs)
├── requirements.txt               # Pinned Python dependencies
├── config.py                      # Local Ollama & dataset configuration
├── demo.py                        # Live CLI end-to-end demonstration script
├── demo_transcript.md             # Recorded transcript of live benchmark run
├── app.py                         # Streamlit interactive web application
├── README.md                      # Comprehensive project documentation
│
├── data/
│   ├── sales_data.csv             # 26 sales records (Sep - Nov 2024)
│   └── README.md                  # Dataset schema and benchmark truth table
│
├── tools/
│   ├── __init__.py                # Package init
│   └── sales_tools.py             # Deterministic sales calculation tools
│
├── mcp_server/
│   ├── __init__.py                # Package init
│   ├── server.py                  # Genuine local MCP stdio server
│   └── client.py                  # Async MCP Client wrapping ClientSession
│
├── agent/
│   ├── __init__.py                # Package init
│   ├── schemas.py                 # Pydantic schemas (AgentDecision, Response)
│   ├── llm.py                     # Local Ollama client & prompt engine
│   ├── core.py                    # Multi-step agent loop & cycle guards
│   ├── evidence.py                # EvidenceRecord & GroundTruthFact tracker
│   └── verifier.py                # Deterministic AnswerVerifier & grounding engine
│
└── tests/
    ├── test_sales_tools.py        # 14 unit tests for analytical tools
    ├── test_mcp_integration.py    # 5 integration tests for MCP server/client
    ├── test_agent_decisions.py    # 11 tests for agent decision schemas & loops
    ├── test_verification.py       # 7 tests for grounding verifier & correction
    └── test_reliability.py        # 6 tests for corrupted CSVs, outages & edge cases
```

---

## 10. Evaluation Checklist & Rubric Fulfillment

| Evaluation Requirement | Implementation Evidence |
| :--- | :--- |
| **100% Local Execution** | Local Ollama (`llama3.2:latest`), no cloud endpoints, no API keys. |
| **Official MCP SDK** | Official Python `mcp` SDK (v2.3+) over `stdio` transport in `mcp_server/`. |
| **Dynamic Tool Selection** | Model selects tools dynamically based on MCP tool discovery schemas. |
| **Deterministic Arithmetic** | Pandas/Python computes exact sums, rankings, and percentage growth. |
| **Deterministic Grounding Verification** | `AnswerVerifier` validates entity, revenue, and percentage claims before presentation. |
| **Zero-Baseline Handling** | `compare_periods` returns `null` percentage growth with descriptive explanation. |
| **Refusal Over Hallucination** | Ungrounded claims trigger correction or refusal (`status="unverified"`). |
| **Transparent Execution Trace** | Step-by-step breakdown of decisions, tool names, arguments, and observations. |
| **Automated Testing** | 43 comprehensive Pytest tests covering all components and edge cases. |
| **Streamlit Interface** | Modern web dashboard in `app.py` with real-time MCP tool inspection. |
