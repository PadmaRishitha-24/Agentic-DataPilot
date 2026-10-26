"""Official MCP Server for Agentic DataPilot.

Exposes deterministic sales analysis tools over standard input/output (stdio) transport:
- get_sales_data: Filter and load CSV sales records.
- calculate_metrics: Calculate deterministic revenue, units, and product rankings.
- compare_periods: Compare revenue and growth between two distinct date periods.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.mcpserver import MCPServer
from tools.sales_tools import calculate_metrics as calc_metrics_fn
from tools.sales_tools import compare_periods as compare_periods_fn
from tools.sales_tools import get_sales_data as get_sales_data_fn

# Initialize official MCP Server
server = MCPServer(
    name="AgenticDataPilotServer",
    version="1.0.0",
    instructions="Provides local deterministic tools for sales analysis and reporting.",
)


@server.tool(
    name="get_sales_data",
    description="Read and filter sales records from the local CSV file by optional date range (YYYY-MM-DD) and product name.",
)
def get_sales_data(
    csv_path: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    product: Optional[str] = None,
) -> str:
    """Filter sales records from local CSV file.

    Args:
        csv_path: Optional custom path to CSV file.
        start_date: Optional start date in YYYY-MM-DD format (inclusive).
        end_date: Optional end date in YYYY-MM-DD format (inclusive).
        product: Optional product name filter (case-insensitive).
    """
    result = get_sales_data_fn(
        csv_path=csv_path,
        start_date=start_date,
        end_date=end_date,
        product=product,
    )
    return json.dumps(result, indent=2)


@server.tool(
    name="calculate_metrics",
    description="Calculate deterministic sales metrics including total revenue, order count, total units sold, and product rankings.",
)
def calculate_metrics(
    csv_path: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    product: Optional[str] = None,
) -> str:
    """Calculate deterministic sales metrics and rankings.

    Args:
        csv_path: Optional custom path to CSV file.
        start_date: Optional start date in YYYY-MM-DD format (inclusive).
        end_date: Optional end date in YYYY-MM-DD format (inclusive).
        product: Optional product name filter.
    """
    result = calc_metrics_fn(
        csv_path=csv_path,
        start_date=start_date,
        end_date=end_date,
        product=product,
    )
    return json.dumps(result, indent=2)


@server.tool(
    name="compare_periods",
    description="Compare sales revenue and growth between two distinct date periods (Period A vs Period B) with zero-baseline handling.",
)
def compare_periods(
    period_a_start: str,
    period_a_end: str,
    period_b_start: str,
    period_b_end: str,
    product: Optional[str] = None,
    csv_path: Optional[str] = None,
) -> str:
    """Compare revenue between two explicit date periods.

    Args:
        period_a_start: Start date of baseline period (YYYY-MM-DD).
        period_a_end: End date of baseline period (YYYY-MM-DD).
        period_b_start: Start date of comparison period (YYYY-MM-DD).
        period_b_end: End date of comparison period (YYYY-MM-DD).
        product: Optional product name filter.
        csv_path: Optional custom path to CSV file.
    """
    result = compare_periods_fn(
        period_a_start=period_a_start,
        period_a_end=period_a_end,
        period_b_start=period_b_start,
        period_b_end=period_b_end,
        product=product,
        csv_path=csv_path,
    )
    return json.dumps(result, indent=2)


def main():
    """Run the MCP server over stdio transport."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
