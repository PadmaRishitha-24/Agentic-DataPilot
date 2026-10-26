"""Configuration settings for Agentic DataPilot."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_CSV_PATH = DATA_DIR / "sales_data.csv"
SALES_DATA_PATH = DEFAULT_CSV_PATH

# Local Ollama Settings (100% local, no cloud APIs)
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2:latest"
OLLAMA_TIMEOUT_SECONDS = 60.0

# Agent Execution Limits
MAX_AGENT_ITERATIONS = 5
MAX_CORRECTION_ATTEMPTS = 1

# MCP Settings
MCP_SERVER_MODULE = "mcp_server.server"
