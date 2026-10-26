"""Evidence collection and indexing for grounded verification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field


class GroundTruthFact(BaseModel):
    """An established factual data point extracted from an MCP tool observation."""

    fact_type: str  # e.g., 'top_product', 'revenue', 'growth_pct', 'absolute_change'
    subject: str    # e.g., product name or period name
    numeric_value: Optional[float] = None
    string_value: Optional[str] = None
    source_tool: str
    details: Dict[str, Any] = Field(default_factory=dict)


class EvidenceRecord:
    """Maintains and organizes factual observations collected during tool execution."""

    def __init__(self):
        self.raw_observations: List[Dict[str, Any]] = []
        self.facts: List[GroundTruthFact] = []
        self.known_products: Set[str] = set()
        self.revenue_by_product_and_period: Dict[str, float] = {}
        self.top_products_by_period: Dict[str, str] = {}
        self.period_comparisons: List[Dict[str, Any]] = []

    def add_observation(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: Dict[str, Any],
    ) -> None:
        """Parse and index facts from a tool execution result."""
        self.raw_observations.append({
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": tool_result,
        })

        if tool_result.get("status") == "error":
            return

        if tool_name == "calculate_metrics":
            top_prod = tool_result.get("top_product")
            top_rev = tool_result.get("top_product_revenue")
            tot_rev = tool_result.get("total_revenue")
            start = tool_args.get("start_date", "all")
            end = tool_args.get("end_date", "all")
            period_key = f"{start}_to_{end}"

            if top_prod:
                self.known_products.add(top_prod)
                self.top_products_by_period[period_key] = top_prod
                self.facts.append(
                    GroundTruthFact(
                        fact_type="top_product",
                        subject=period_key,
                        string_value=top_prod,
                        numeric_value=float(top_rev) if top_rev is not None else None,
                        source_tool="calculate_metrics",
                        details={"start_date": start, "end_date": end, "revenue": top_rev},
                    )
                )

            if tot_rev is not None:
                self.facts.append(
                    GroundTruthFact(
                        fact_type="total_revenue",
                        subject=period_key,
                        numeric_value=float(tot_rev),
                        source_tool="calculate_metrics",
                        details={"start_date": start, "end_date": end},
                    )
                )

            for item in tool_result.get("product_rankings", []):
                p_name = item.get("product")
                p_rev = item.get("total_revenue")
                if p_name and p_rev is not None:
                    self.known_products.add(p_name)
                    rev_key = f"{p_name}::{period_key}"
                    self.revenue_by_product_and_period[rev_key] = float(p_rev)
                    self.facts.append(
                        GroundTruthFact(
                            fact_type="product_revenue",
                            subject=p_name,
                            numeric_value=float(p_rev),
                            source_tool="calculate_metrics",
                            details={"period": period_key, "units_sold": item.get("units_sold")},
                        )
                    )

        elif tool_name == "compare_periods":
            comp = tool_result.get("comparison", {})
            p_a = tool_result.get("period_a", {})
            p_b = tool_result.get("period_b", {})
            prod = tool_result.get("product_filter") or tool_args.get("product") or "all"

            rev_a = p_a.get("total_revenue")
            rev_b = p_b.get("total_revenue")
            abs_change = comp.get("absolute_change")
            pct_growth = comp.get("percentage_growth")

            if prod and prod != "all":
                self.known_products.add(prod)

            self.period_comparisons.append({
                "product": prod,
                "period_a": p_a,
                "period_b": p_b,
                "comparison": comp,
            })

            if pct_growth is not None:
                self.facts.append(
                    GroundTruthFact(
                        fact_type="percentage_growth",
                        subject=prod,
                        numeric_value=float(pct_growth),
                        source_tool="compare_periods",
                        details={
                            "period_a_revenue": rev_a,
                            "period_b_revenue": rev_b,
                            "absolute_change": abs_change,
                        },
                    )
                )

            if abs_change is not None:
                self.facts.append(
                    GroundTruthFact(
                        fact_type="absolute_change",
                        subject=prod,
                        numeric_value=float(abs_change),
                        source_tool="compare_periods",
                        details={"period_a_revenue": rev_a, "period_b_revenue": rev_b},
                    )
                )

            if rev_a is not None:
                self.facts.append(
                    GroundTruthFact(
                        fact_type="period_a_revenue",
                        subject=prod,
                        numeric_value=float(rev_a),
                        source_tool="compare_periods",
                    )
                )
            if rev_b is not None:
                self.facts.append(
                    GroundTruthFact(
                        fact_type="period_b_revenue",
                        subject=prod,
                        numeric_value=float(rev_b),
                        source_tool="compare_periods",
                    )
                )

        elif tool_name == "get_sales_data":
            records = tool_result.get("records", [])
            for r in records:
                p = r.get("product")
                if p:
                    self.known_products.add(p)
            if records:
                tot_rev = sum(float(r.get("revenue", 0.0)) for r in records)
                start = tool_args.get("start_date", "all")
                end = tool_args.get("end_date", "all")
                period_key = f"{start}_to_{end}"
                self.facts.append(
                    GroundTruthFact(
                        fact_type="total_revenue",
                        subject=period_key,
                        numeric_value=round(tot_rev, 2),
                        source_tool="get_sales_data",
                        details={"record_count": len(records)},
                    )
                )
                prod_totals: Dict[str, float] = {}
                for r in records:
                    p = r.get("product")
                    if p:
                        prod_totals[p] = prod_totals.get(p, 0.0) + float(r.get("revenue", 0.0))
                if prod_totals:
                    top_p = max(prod_totals, key=prod_totals.get)
                    self.top_products_by_period[period_key] = top_p
                    self.facts.append(
                        GroundTruthFact(
                            fact_type="top_product",
                            subject=period_key,
                            string_value=top_p,
                            numeric_value=round(prod_totals[top_p], 2),
                            source_tool="get_sales_data",
                            details={"revenue": round(prod_totals[top_p], 2)},
                        )
                    )
                    for prod_name, p_rev in prod_totals.items():
                        self.facts.append(
                            GroundTruthFact(
                                fact_type="product_revenue",
                                subject=prod_name,
                                numeric_value=round(p_rev, 2),
                                source_tool="get_sales_data",
                                details={"period": period_key},
                            )
                        )

    def get_summary_text(self) -> str:
        """Return formatted bullet points of established ground truth facts."""
        if not self.facts:
            return "No verified facts established yet."

        lines = []
        for f in self.facts:
            if f.fact_type == "top_product":
                lines.append(f"- Top product for {f.subject}: '{f.string_value}' (${f.numeric_value:,.2f})")
            elif f.fact_type == "product_revenue":
                lines.append(f"- Revenue for '{f.subject}' in {f.details.get('period', 'period')}: ${f.numeric_value:,.2f}")
            elif f.fact_type == "percentage_growth":
                lines.append(f"- Growth for '{f.subject}': {f.numeric_value:+.2f}% (Absolute: ${f.details.get('absolute_change', 0):+,.2f})")
            elif f.fact_type == "total_revenue":
                lines.append(f"- Total revenue for {f.subject}: ${f.numeric_value:,.2f}")
        return "\n".join(lines)
