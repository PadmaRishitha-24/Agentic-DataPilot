"""Sales analysis business logic tools for Agentic DataPilot.

Deterministic Python calculations for:
1. get_sales_data: Filter and load CSV sales records with robust validation.
2. calculate_metrics: Deterministic arithmetic for revenue, orders, and product rankings.
3. compare_periods: Compare revenue between two explicit periods with zero-baseline handling.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REQUIRED_COLUMNS = {
    "order_id",
    "date",
    "product",
    "category",
    "units_sold",
    "unit_price",
    "revenue",
}

DEFAULT_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"


def _validate_date(date_str: str, field_name: str) -> Optional[datetime]:
    """Validate ISO-8601 YYYY-MM-DD date format."""
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"Invalid {field_name} format: '{date_str}'. Expected format is YYYY-MM-DD (e.g. '2024-10-01')."
        )


def get_sales_data(
    csv_path: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    product: Optional[str] = None,
) -> Dict[str, Any]:
    """Read and filter sales records from a local CSV file.

    Args:
        csv_path: Optional path to CSV file. Defaults to data/sales_data.csv.
        start_date: Optional start date in YYYY-MM-DD format (inclusive).
        end_date: Optional end date in YYYY-MM-DD format (inclusive).
        product: Optional product name filter (case-insensitive).

    Returns:
        Structured dictionary with status, record list, total count, or error details.
    """
    target_path = Path(csv_path) if csv_path else DEFAULT_CSV_PATH

    # Check file existence
    if not target_path.exists():
        return {
            "status": "error",
            "error_type": "FileNotFoundError",
            "message": f"CSV file not found at: {target_path}",
            "records": [],
            "total_records": 0,
        }

    # Validate date formats if provided
    start_dt = None
    end_dt = None
    try:
        if start_date:
            start_dt = _validate_date(start_date, "start_date")
        if end_date:
            end_dt = _validate_date(end_date, "end_date")
    except ValueError as e:
        return {
            "status": "error",
            "error_type": "InvalidDateFormat",
            "message": str(e),
            "records": [],
            "total_records": 0,
        }

    if start_dt and end_dt and start_dt > end_dt:
        return {
            "status": "error",
            "error_type": "InvalidDateRange",
            "message": f"start_date ('{start_date}') must be less than or equal to end_date ('{end_date}').",
            "records": [],
            "total_records": 0,
        }

    # Validate product filter format if provided
    if product is not None:
        if isinstance(product, (list, tuple, set, dict)):
            return {
                "status": "error",
                "error_type": "InvalidProductFilter",
                "message": (
                    f"Product filter must be a single string, but received {type(product).__name__}: {product}. "
                    "To analyze or rank all products, leave product empty."
                ),
                "records": [],
                "total_records": 0,
            }
        product_str = str(product).strip()
        if (product_str.startswith("[") and product_str.endswith("]")) or (product_str.startswith("(") and product_str.endswith(")")):
            return {
                "status": "error",
                "error_type": "InvalidProductFilter",
                "message": (
                    f"Product filter must be a single string, but received a list representation: '{product_str}'. "
                    "To analyze or rank all products, leave product empty."
                ),
                "records": [],
                "total_records": 0,
            }

        prod_clean = product_str.lower().strip()
        analytical_terms = {
            "lowest_revenue_product", "highest_revenue_product", "top_product", "lowest_product",
            "bottom_product", "best_product", "most_revenue_product", "least_revenue_product",
            "all_products", "all products", "all", "rank_all_products", "rank all products",
            "all_product", "every_product", "products", "everything",
        }
        if (
            prod_clean in analytical_terms
            or any(term in prod_clean for term in [
                "lowest_revenue", "highest_revenue", "top_product", "lowest_product",
                "all_product", "all product", "rank_all", "rank all"
            ])
        ):
            return {
                "status": "error",
                "error_type": "InvalidProductFilter",
                "message": (
                    f"'{product}' is an analytical operation or aggregate request, not a literal product name. "
                    "To rank all products, analyze all products, or find the lowest/highest revenue product, "
                    "leave product empty (omit product filter) and call calculate_metrics."
                ),
                "records": [],
                "total_records": 0,
            }

    # Parse and validate CSV
    records: List[Dict[str, Any]] = []
    try:
        with open(target_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return {
                    "status": "error",
                    "error_type": "EmptyFileError",
                    "message": f"CSV file at {target_path} is empty or has no header.",
                    "records": [],
                    "total_records": 0,
                }

            file_headers = {h.strip() for h in reader.fieldnames if h}
            missing = REQUIRED_COLUMNS - file_headers
            if missing:
                return {
                    "status": "error",
                    "error_type": "MissingColumnsError",
                    "message": f"CSV is missing required column(s): {sorted(list(missing))}",
                    "records": [],
                    "total_records": 0,
                }

            normalized_product_filter = product.strip().lower() if product else None

            for line_no, row in enumerate(reader, start=2):
                row_date_str = row.get("date", "").strip()
                try:
                    row_dt = datetime.strptime(row_date_str, "%Y-%m-%d")
                except ValueError:
                    # Ignore or flag malformed date in a row
                    continue

                # Filter by start_date
                if start_dt and row_dt < start_dt:
                    continue

                # Filter by end_date
                if end_dt and row_dt > end_dt:
                    continue

                row_product = row.get("product", "").strip()
                # Filter by product name
                if normalized_product_filter and row_product.lower() != normalized_product_filter:
                    continue

                try:
                    units_sold = int(row.get("units_sold", 0))
                    unit_price = float(row.get("unit_price", 0.0))
                    revenue = float(row.get("revenue", units_sold * unit_price))
                except (ValueError, TypeError):
                    continue

                records.append({
                    "order_id": row.get("order_id", "").strip(),
                    "date": row_date_str,
                    "product": row_product,
                    "category": row.get("category", "").strip(),
                    "units_sold": units_sold,
                    "unit_price": round(unit_price, 2),
                    "revenue": round(revenue, 2),
                })

    except Exception as e:
        return {
            "status": "error",
            "error_type": "CSVReadError",
            "message": f"Failed to read CSV file: {str(e)}",
            "records": [],
            "total_records": 0,
        }

    return {
        "status": "success",
        "total_records": len(records),
        "records": records,
        "filters_applied": {
            "start_date": start_date,
            "end_date": end_date,
            "product": product,
            "csv_path": str(target_path),
        },
    }


def calculate_metrics(
    records: Optional[List[Dict[str, Any]]] = None,
    csv_path: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    product: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate deterministic summary metrics, totals, and product rankings.

    Args:
        records: Optional pre-filtered list of records. If None, records are loaded via get_sales_data.
        csv_path: Optional CSV file path (used if records is None).
        start_date: Optional start date filter (used if records is None).
        end_date: Optional end date filter (used if records is None).
        product: Optional product filter (used if records is None).

    Returns:
        Structured dictionary with total revenue, units, rankings, and top performer.
    """
    if records is None:
        fetch_result = get_sales_data(
            csv_path=csv_path,
            start_date=start_date,
            end_date=end_date,
            product=product,
        )
        if fetch_result.get("status") == "error":
            return fetch_result
        records = fetch_result.get("records", [])

    if not records:
        return {
            "status": "success",
            "total_records": 0,
            "total_revenue": 0.0,
            "total_units": 0,
            "average_order_revenue": 0.0,
            "product_rankings": [],
            "top_product": None,
            "top_product_revenue": 0.0,
            "message": "No sales records matched the specified criteria.",
        }

    total_revenue = 0.0
    total_units = 0
    product_stats: Dict[str, Dict[str, Any]] = {}

    for r in records:
        rev = float(r.get("revenue", 0.0))
        units = int(r.get("units_sold", 0))
        prod_name = str(r.get("product", "Unknown"))

        total_revenue += rev
        total_units += units

        if prod_name not in product_stats:
            product_stats[prod_name] = {
                "product": prod_name,
                "total_revenue": 0.0,
                "units_sold": 0,
                "order_count": 0,
            }
        product_stats[prod_name]["total_revenue"] += rev
        product_stats[prod_name]["units_sold"] += units
        product_stats[prod_name]["order_count"] += 1

    total_revenue = round(total_revenue, 2)
    avg_order_rev = round(total_revenue / len(records), 2) if records else 0.0

    rankings: List[Dict[str, Any]] = []
    for stat in product_stats.values():
        prod_rev = round(stat["total_revenue"], 2)
        share_pct = round((prod_rev / total_revenue) * 100.0, 2) if total_revenue > 0 else 0.0
        rankings.append({
            "product": stat["product"],
            "total_revenue": prod_rev,
            "units_sold": stat["units_sold"],
            "order_count": stat["order_count"],
            "share_of_total_pct": share_pct,
        })

    rankings.sort(key=lambda x: x["total_revenue"], reverse=True)

    top_product = rankings[0]["product"] if rankings else None
    top_product_revenue = rankings[0]["total_revenue"] if rankings else 0.0
    lowest_product = rankings[-1]["product"] if rankings else None
    lowest_product_revenue = rankings[-1]["total_revenue"] if rankings else 0.0

    return {
        "status": "success",
        "total_records": len(records),
        "total_revenue": total_revenue,
        "total_units": total_units,
        "average_order_revenue": avg_order_rev,
        "product_rankings": rankings,
        "top_product": top_product,
        "top_product_revenue": top_product_revenue,
        "lowest_product": lowest_product,
        "lowest_product_revenue": lowest_product_revenue,
    }


def compare_periods(
    period_a_start: str,
    period_a_end: str,
    period_b_start: str,
    period_b_end: str,
    product: Optional[str] = None,
    csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare revenue and sales metrics between two distinct date periods.

    Args:
        period_a_start: Start date of baseline period (YYYY-MM-DD).
        period_a_end: End date of baseline period (YYYY-MM-DD).
        period_b_start: Start date of comparison period (YYYY-MM-DD).
        period_b_end: End date of comparison period (YYYY-MM-DD).
        product: Optional specific product to compare across both periods.
        csv_path: Optional CSV file path.

    Returns:
        Structured dictionary comparing revenue, absolute change, percentage growth, and notes.
    """
    # Fetch metrics for Period A
    metrics_a = calculate_metrics(
        csv_path=csv_path,
        start_date=period_a_start,
        end_date=period_a_end,
        product=product,
    )
    if metrics_a.get("status") == "error":
        return {
            "status": "error",
            "error_type": "PeriodAError",
            "message": f"Error in baseline period: {metrics_a.get('message')}",
        }

    # Fetch metrics for Period B
    metrics_b = calculate_metrics(
        csv_path=csv_path,
        start_date=period_b_start,
        end_date=period_b_end,
        product=product,
    )
    if metrics_b.get("status") == "error":
        return {
            "status": "error",
            "error_type": "PeriodBError",
            "message": f"Error in comparison period: {metrics_b.get('message')}",
        }

    rev_a = float(metrics_a.get("total_revenue", 0.0))
    rev_b = float(metrics_b.get("total_revenue", 0.0))
    abs_change = round(rev_b - rev_a, 2)

    if rev_a == 0.0:
        zero_baseline = True
        pct_growth = None
        if rev_b == 0.0:
            interpretation = "Both periods had $0.00 revenue. Growth is 0%."
        else:
            interpretation = f"Baseline revenue was $0.00. Revenue reached ${rev_b:,.2f} (percentage growth undefined due to zero baseline)."
    else:
        zero_baseline = False
        pct_growth = round(((rev_b - rev_a) / rev_a) * 100.0, 2)
        direction = "+" if pct_growth >= 0 else ""
        interpretation = f"{direction}{pct_growth:.2f}% revenue change from ${rev_a:,.2f} to ${rev_b:,.2f}"

    return {
        "status": "success",
        "product_filter": product,
        "period_a": {
            "start_date": period_a_start,
            "end_date": period_a_end,
            "total_revenue": rev_a,
            "total_units": metrics_a.get("total_units", 0),
            "record_count": metrics_a.get("total_records", 0),
        },
        "period_b": {
            "start_date": period_b_start,
            "end_date": period_b_end,
            "total_revenue": rev_b,
            "total_units": metrics_b.get("total_units", 0),
            "record_count": metrics_b.get("total_records", 0),
        },
        "comparison": {
            "absolute_change": abs_change,
            "percentage_growth": pct_growth,
            "zero_baseline_detected": zero_baseline,
            "interpretation": interpretation,
        },
    }
