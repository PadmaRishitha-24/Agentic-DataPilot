"""Unit tests for deterministic sales tools."""

import tempfile
from pathlib import Path
import pytest

from tools.sales_tools import calculate_metrics, compare_periods, get_sales_data


class TestGetSalesData:
    """Tests for get_sales_data tool."""

    def test_load_all_records_success(self):
        result = get_sales_data()
        assert result["status"] == "success"
        assert result["total_records"] > 0
        first = result["records"][0]
        assert "order_id" in first
        assert "revenue" in first
        assert "product" in first

    def test_filter_by_date_range(self):
        # September 2024
        res_sep = get_sales_data(start_date="2024-09-01", end_date="2024-09-30")
        assert res_sep["status"] == "success"
        assert all("2024-09-01" <= r["date"] <= "2024-09-30" for r in res_sep["records"])

        # October 2024
        res_oct = get_sales_data(start_date="2024-10-01", end_date="2024-10-31")
        assert res_oct["status"] == "success"
        assert all("2024-10-01" <= r["date"] <= "2024-10-31" for r in res_oct["records"])

    def test_filter_by_product_case_insensitive(self):
        res = get_sales_data(product="quantum laptop")
        assert res["status"] == "success"
        assert len(res["records"]) > 0
        assert all(r["product"] == "Quantum Laptop" for r in res["records"])

    def test_filter_nonexistent_product(self):
        res = get_sales_data(product="NonExistentWidget999")
        assert res["status"] == "success"
        assert res["total_records"] == 0
        assert res["records"] == []

    def test_missing_file_error(self):
        res = get_sales_data(csv_path="data/does_not_exist_xyz.csv")
        assert res["status"] == "error"
        assert res["error_type"] == "FileNotFoundError"

    def test_invalid_date_format_error(self):
        res = get_sales_data(start_date="2024/09/01")
        assert res["status"] == "error"
        assert res["error_type"] == "InvalidDateFormat"

    def test_inverted_date_range_error(self):
        res = get_sales_data(start_date="2024-10-31", end_date="2024-10-01")
        assert res["status"] == "error"
        assert res["error_type"] == "InvalidDateRange"

    def test_missing_required_columns(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("order_id,date,some_col\n1,2024-10-01,val\n")
            temp_path = f.name

        try:
            res = get_sales_data(csv_path=temp_path)
            assert res["status"] == "error"
            assert res["error_type"] == "MissingColumnsError"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_reject_list_of_products_as_filter(self):
        """Reject list or bracketed list representations as single product filters."""
        # 1. Actual Python list
        res_list = get_sales_data(product=["Quantum Laptop", "CyberPhone Pro"])
        assert res_list["status"] == "error"
        assert res_list["error_type"] == "InvalidProductFilter"
        assert "must be a single string" in res_list["message"]

        # 2. Stringified list
        res_str = get_sales_data(product="['Quantum Laptop', 'CyberPhone Pro']")
        assert res_str["status"] == "error"
        assert res_str["error_type"] == "InvalidProductFilter"
        assert "list representation" in res_str["message"]

    def test_reject_all_products_as_literal_filter(self):
        # Passing 'all_products' or 'all products' must be rejected as an analytical operation
        res1 = get_sales_data(product="all_products")
        assert res1["status"] == "error"
        assert res1["error_type"] == "InvalidProductFilter"
        assert "analytical operation" in res1["message"] or "aggregate request" in res1["message"]

        res2 = get_sales_data(product="all products")
        assert res2["status"] == "error"
        assert res2["error_type"] == "InvalidProductFilter"

        res3 = calculate_metrics(product="all_products")
        assert res3["status"] == "error"
        assert res3["error_type"] == "InvalidProductFilter"


class TestCalculateMetrics:
    """Tests for calculate_metrics tool."""

    def test_october_top_product(self):
        # Benchmark verification: Quantum Laptop should be #1 in October
        res = calculate_metrics(start_date="2024-10-01", end_date="2024-10-31")
        assert res["status"] == "success"
        assert res["top_product"] == "Quantum Laptop"
        assert res["top_product_revenue"] == 68400.0
        assert res["total_revenue"] > 68400.0

        # Verify product rankings are ordered descending
        rankings = res["product_rankings"]
        assert len(rankings) >= 5
        revenues = [p["total_revenue"] for p in rankings]
        assert revenues == sorted(revenues, reverse=True)

    def test_empty_period_metrics(self):
        # Date range with zero sales
        res = calculate_metrics(start_date="2020-01-01", end_date="2020-01-15")
        assert res["status"] == "success"
        assert res["total_records"] == 0
        assert res["total_revenue"] == 0.0
        assert res["top_product"] is None
        assert res["product_rankings"] == []

    def test_specific_product_metrics(self):
        res = calculate_metrics(
            start_date="2024-09-01",
            end_date="2024-09-30",
            product="Quantum Laptop",
        )
        assert res["status"] == "success"
        assert res["total_revenue"] == 44400.0
        assert res["top_product"] == "Quantum Laptop"

    def test_october_lowest_revenue_product(self):
        """Verify deterministic calculation of lowest-revenue product in October."""
        res = calculate_metrics(start_date="2024-10-01", end_date="2024-10-31")
        assert res["status"] == "success"
        assert res["lowest_product"] == "SolarKey Mechanical Keyboard"
        assert res["lowest_product_revenue"] == 4000.0
        assert res["product_rankings"][-1]["product"] == "SolarKey Mechanical Keyboard"
        assert res["product_rankings"][-1]["total_revenue"] == 4000.0

    def test_revenue_aggregation_across_multiple_records(self):
        """Verify that revenue and units aggregate across multiple orders per product."""
        res = calculate_metrics(start_date="2024-10-01", end_date="2024-10-31")
        assert res["status"] == "success"
        by_prod = {p["product"]: p for p in res["product_rankings"]}
        # Quantum Laptop: 3 orders (21600 + 24000 + 22800 = 68400.0, 57 units)
        assert by_prod["Quantum Laptop"]["order_count"] == 3
        assert by_prod["Quantum Laptop"]["total_revenue"] == 68400.0
        assert by_prod["Quantum Laptop"]["units_sold"] == 57
        # CyberPhone Pro: 2 orders (17600 + 14400 = 32000.0, 40 units)
        assert by_prod["CyberPhone Pro"]["order_count"] == 2
        assert by_prod["CyberPhone Pro"]["total_revenue"] == 32000.0
        assert by_prod["CyberPhone Pro"]["units_sold"] == 40
        # EchoPods Wireless: 2 orders (3750 + 2250 = 6000.0, 40 units)
        assert by_prod["EchoPods Wireless"]["order_count"] == 2
        assert by_prod["EchoPods Wireless"]["total_revenue"] == 6000.0
        # SolarKey Mechanical Keyboard: 1 order (4000.0, 40 units)
        assert by_prod["SolarKey Mechanical Keyboard"]["order_count"] == 1
        assert by_prod["SolarKey Mechanical Keyboard"]["total_revenue"] == 4000.0


class TestComparePeriods:
    """Tests for compare_periods tool."""

    def test_benchmark_growth_calculation(self):
        # Benchmark: Quantum Laptop growth from September to October
        res = compare_periods(
            period_a_start="2024-09-01",
            period_a_end="2024-09-30",
            period_b_start="2024-10-01",
            period_b_end="2024-10-31",
            product="Quantum Laptop",
        )
        assert res["status"] == "success"
        assert res["period_a"]["total_revenue"] == 44400.0
        assert res["period_b"]["total_revenue"] == 68400.0
        assert res["comparison"]["absolute_change"] == 24000.0
        assert res["comparison"]["percentage_growth"] == 54.05
        assert res["comparison"]["zero_baseline_detected"] is False

    def test_zero_baseline_handling(self):
        # SolarKey Mechanical Keyboard has 0 sales in September, 4000 in October
        res = compare_periods(
            period_a_start="2024-09-01",
            period_a_end="2024-09-30",
            period_b_start="2024-10-01",
            period_b_end="2024-10-31",
            product="SolarKey Mechanical Keyboard",
        )
        assert res["status"] == "success"
        assert res["period_a"]["total_revenue"] == 0.0
        assert res["period_b"]["total_revenue"] == 4000.0
        assert res["comparison"]["zero_baseline_detected"] is True
        assert res["comparison"]["percentage_growth"] is None

    def test_invalid_date_error_propagation(self):
        res = compare_periods(
            period_a_start="invalid-date",
            period_a_end="2024-09-30",
            period_b_start="2024-10-01",
            period_b_end="2024-10-31",
        )
        assert res["status"] == "error"
        assert "error_type" in res
