"""Cross-board analytics: sales pipeline vs operational workload.

Join policy
-----------
The only dimension that is reliably shared by both boards is the **normalised
sector**; ``owner`` is a usable secondary dimension when both boards carry the
same personnel codes. Customer identifiers are *not* joinable -- the Deals board
uses ``COMPANY###`` while the Work Orders board uses ``WOCOMPANY_###``, and no
mapping table exists. Deal names are masked aliases that repeat across rows, so
they are not a key either.

The module therefore aggregates each board independently and compares the
aggregates. It never claims a specific deal produced a specific work order, and
every output is labelled as an indication, not a causal link.
"""
from __future__ import annotations

import pandas as pd

from analytics.deals import _money, _pct
from analytics.normalization import NormalizedDataset
from analytics.quality import DataQualityReport, Severity
from utils.numbers import safe_ratio

__all__ = [
    "JOIN_POLICY",
    "shared_dimension_overlap",
    "pipeline_vs_workload",
    "capacity_signals",
    "analyze_cross_board",
]

JOIN_POLICY = {
    "join_key": "normalised sector",
    "secondary_key": "owner code (when present on both boards)",
    "rejected_keys": {
        "client": (
            "Deals use COMPANY### codes and Work Orders use WOCOMPANY_### codes; "
            "there is no reliable mapping between them, so customer-level joins "
            "are not attempted."
        ),
        "deal_name": (
            "Deal names are masked aliases that repeat across many rows, so they "
            "cannot uniquely link a deal to a work order."
        ),
    },
    "interpretation": (
        "Figures are aggregate comparisons by sector, not record-level links. "
        "They indicate where sales and delivery are out of balance; they do not "
        "prove that pipeline caused workload or vice versa."
    ),
}


def shared_dimension_overlap(
    deals: NormalizedDataset, work_orders: NormalizedDataset, *, dimension: str = "sector"
) -> dict:
    """Report how well the join dimension actually overlaps between the boards."""
    if dimension not in deals.frame.columns or dimension not in work_orders.frame.columns:
        return {"dimension": dimension, "joinable": False,
                "reason": f"'{dimension}' is not present on both boards."}

    deal_values = set(deals.frame[dimension].dropna().unique().tolist())
    wo_values = set(work_orders.frame[dimension].dropna().unique().tolist())
    shared = sorted(deal_values & wo_values)
    return {
        "dimension": dimension,
        "joinable": bool(shared),
        "shared_values": shared,
        "deals_only": sorted(deal_values - wo_values),
        "work_orders_only": sorted(wo_values - deal_values),
        "coverage_pct": _pct(safe_ratio(len(shared), len(deal_values | wo_values))),
    }


def pipeline_vs_workload(
    deals: NormalizedDataset,
    work_orders: NormalizedDataset,
    *,
    dimension: str = "sector",
    limit: int = 15,
) -> dict:
    """Compare open pipeline against active operational workload per sector."""
    overlap = shared_dimension_overlap(deals, work_orders, dimension=dimension)
    report = DataQualityReport(dataset="cross_board", row_count=0)

    if dimension not in deals.frame.columns or dimension not in work_orders.frame.columns:
        return {"available": False, "reason": overlap.get("reason"), "join_policy": JOIN_POLICY}

    deal_frame = deals.frame
    open_deals = deal_frame[deal_frame["is_open"]] if "is_open" in deal_frame else deal_frame
    wo_frame = work_orders.frame
    active_wo = wo_frame[wo_frame["is_active"]] if "is_active" in wo_frame else wo_frame

    deal_value_available = "value" in open_deals.columns and open_deals["value"].notna().any()
    wo_value_available = "order_value" in active_wo.columns and active_wo["order_value"].notna().any()

    # ``min_count=1`` keeps a group whose values are all missing as NaN. A plain
    # sum would report 0.0, which would read as "this sector has zero pipeline"
    # when the truth is "no value was recorded for it".
    def _nan_safe_sum(series: pd.Series) -> float:
        return series.sum(min_count=1)

    deal_agg = open_deals.groupby(dimension, dropna=False).agg(
        open_deal_count=("item_id", "size"),
        open_pipeline_value=("value", _nan_safe_sum) if deal_value_available
        else ("item_id", "size"),
        open_deals_missing_value=("value", lambda v: int(v.isna().sum()))
        if deal_value_available else ("item_id", "size"),
    )
    if not deal_value_available:
        deal_agg["open_pipeline_value"] = float("nan")
        deal_agg["open_deals_missing_value"] = 0

    wo_agg = active_wo.groupby(dimension, dropna=False).agg(
        active_work_order_count=("item_id", "size"),
        active_order_value=("order_value", _nan_safe_sum) if wo_value_available
        else ("item_id", "size"),
    )
    if not wo_value_available:
        wo_agg["active_order_value"] = float("nan")

    delayed_agg = None
    if "is_delayed" in wo_frame.columns:
        delayed_agg = (
            wo_frame[wo_frame["is_delayed"].fillna(False)]
            .groupby(dimension, dropna=False)
            .size()
            .rename("delayed_work_order_count")
        )

    combined = deal_agg.join(wo_agg, how="outer")
    if delayed_agg is not None:
        combined = combined.join(delayed_agg, how="left")
    combined = combined.fillna({
        "open_deal_count": 0, "active_work_order_count": 0, "delayed_work_order_count": 0
    })

    total_pipeline = float(open_deals["value"].sum()) if deal_value_available else None
    total_workload = float(active_wo["order_value"].sum()) if wo_value_available else None
    total_open_deals = int(len(open_deals))
    total_active_wo = int(len(active_wo))

    rows = []
    for key, row in combined.iterrows():
        label = "Unknown" if pd.isna(key) else str(key)
        pipeline_value = row.get("open_pipeline_value")
        workload_value = row.get("active_order_value")
        pipeline_share = safe_ratio(pipeline_value, total_pipeline) if deal_value_available else None
        workload_share = safe_ratio(workload_value, total_workload) if wo_value_available else None

        record = {
            dimension: label,
            "open_deal_count": int(row.get("open_deal_count", 0)),
            "active_work_order_count": int(row.get("active_work_order_count", 0)),
            "delayed_work_order_count": int(row.get("delayed_work_order_count", 0) or 0),
            "open_deal_share_pct": _pct(safe_ratio(row.get("open_deal_count", 0), total_open_deals)),
            "active_work_order_share_pct": _pct(
                safe_ratio(row.get("active_work_order_count", 0), total_active_wo)
            ),
        }
        if deal_value_available and pd.notna(pipeline_value):
            record["open_pipeline_value"] = _money(float(pipeline_value))
            record["pipeline_share_pct"] = _pct(pipeline_share)
            record["open_deals_missing_value"] = int(row.get("open_deals_missing_value", 0) or 0)
        elif deal_value_available:
            record["open_pipeline_value"] = None
            record["value_note"] = "no deal value recorded for this group"
        if wo_value_available and pd.notna(workload_value):
            record["active_order_value"] = _money(float(workload_value))
            record["workload_share_pct"] = _pct(workload_share)

        # Balance = how much bigger this sector's share of pipeline is than its
        # share of current delivery workload. Share-based so it is unit-free.
        if pipeline_share is not None and workload_share is not None:
            record["pipeline_minus_workload_share_pts"] = round(
                (pipeline_share - workload_share) * 100, 1
            )
            if pipeline_share > 0 and workload_share > 0:
                record["pipeline_to_workload_ratio"] = round(pipeline_share / workload_share, 2)
        elif total_open_deals and total_active_wo:
            count_pipeline_share = safe_ratio(row.get("open_deal_count", 0), total_open_deals) or 0
            count_workload_share = safe_ratio(row.get("active_work_order_count", 0), total_active_wo) or 0
            record["pipeline_minus_workload_share_pts"] = round(
                (count_pipeline_share - count_workload_share) * 100, 1
            )
            record["basis"] = "deal / work-order counts (values unavailable)"
        rows.append(record)

    sort_value = "open_pipeline_value" if deal_value_available else "open_deal_count"

    def _sort_key(record: dict) -> float:
        """Rank by value when known; a group with no recorded value sorts last."""
        raw = record.get(sort_value)
        if isinstance(raw, dict):
            raw = raw.get("amount")
        if raw is None:
            return 0.0
        try:
            return -float(raw)
        except (TypeError, ValueError):
            return 0.0

    rows.sort(key=_sort_key)

    if not deal_value_available or not wo_value_available:
        report.add(
            "cross_board_value_unavailable",
            "Value-based comparison was not possible on at least one board; the "
            "comparison falls back to record counts.",
            1, Severity.INFO, total=1,
        )
    if overlap.get("deals_only"):
        report.add(
            "sectors_without_workload",
            f"{len(overlap['deals_only'])} sector(s) appear in the pipeline but have no "
            f"work orders at all: {', '.join(overlap['deals_only'][:6])}",
            len(overlap["deals_only"]), Severity.INFO, total=1,
        )
    if overlap.get("work_orders_only"):
        report.add(
            "sectors_without_pipeline",
            f"{len(overlap['work_orders_only'])} sector(s) have work orders but no deals "
            f"on the Deals board: {', '.join(overlap['work_orders_only'][:6])}",
            len(overlap["work_orders_only"]), Severity.INFO, total=1,
        )

    return {
        "available": True,
        "dimension": dimension,
        "join_policy": JOIN_POLICY,
        "overlap": overlap,
        "totals": {
            "open_deal_count": total_open_deals,
            "open_pipeline_value": _money(total_pipeline),
            "active_work_order_count": total_active_wo,
            "active_order_value": _money(total_workload),
        },
        "rows": rows[:limit],
        "data_quality": report.as_dict(),
    }


def capacity_signals(comparison: dict, *, threshold_pts: float = 8.0) -> dict:
    """Turn the sector comparison into balance signals.

    ``threshold_pts`` is the share-point gap at which a sector is called out.
    Signals are deliberately phrased as indications requiring judgement.
    """
    if not comparison.get("available"):
        return {"available": False, "reason": comparison.get("reason")}

    rows = comparison.get("rows", [])
    pipeline_heavy, delivery_heavy, balanced, no_workload, no_pipeline = [], [], [], [], []

    for row in rows:
        gap = row.get("pipeline_minus_workload_share_pts")
        dimension = comparison["dimension"]
        label = row.get(dimension)
        if row["open_deal_count"] > 0 and row["active_work_order_count"] == 0:
            no_workload.append(label)
        elif row["active_work_order_count"] > 0 and row["open_deal_count"] == 0:
            no_pipeline.append(label)
        if gap is None:
            continue
        if gap >= threshold_pts:
            pipeline_heavy.append({"sector": label, "gap_share_pts": gap,
                                   "open_deal_count": row["open_deal_count"],
                                   "active_work_order_count": row["active_work_order_count"]})
        elif gap <= -threshold_pts:
            delivery_heavy.append({"sector": label, "gap_share_pts": gap,
                                   "open_deal_count": row["open_deal_count"],
                                   "active_work_order_count": row["active_work_order_count"]})
        else:
            balanced.append(label)

    return {
        "available": True,
        "threshold_share_points": threshold_pts,
        "pipeline_ahead_of_delivery": pipeline_heavy,
        "delivery_ahead_of_pipeline": delivery_heavy,
        "broadly_balanced": balanced,
        "pipeline_but_no_active_work_orders": no_workload,
        "active_work_orders_but_no_open_pipeline": no_pipeline,
        "interpretation_note": (
            "A sector whose share of open pipeline materially exceeds its share of "
            "active delivery workload may need delivery capacity if those deals "
            "convert. The reverse suggests current execution load is not being "
            "replenished by new sales. These are indications from aggregate data, "
            "not proven cause and effect."
        ),
    }


def analyze_cross_board(
    deals: NormalizedDataset,
    work_orders: NormalizedDataset,
    *,
    dimension: str = "sector",
) -> dict:
    """Full cross-board bundle used by the agent orchestrator."""
    if deals.empty or work_orders.empty:
        return {
            "available": False,
            "reason": (
                "Cross-board analysis needs data on both boards; "
                f"deals={len(deals.frame)} rows, work orders={len(work_orders.frame)} rows."
            ),
            "join_policy": JOIN_POLICY,
        }

    comparison = pipeline_vs_workload(deals, work_orders, dimension=dimension)
    report = DataQualityReport(dataset="cross_board", row_count=0)
    report.extend(deals.quality)
    report.extend(work_orders.quality)

    return {
        "available": comparison.get("available", False),
        "comparison": comparison,
        "signals": capacity_signals(comparison),
        "data_quality": report.as_dict(limit=5),
    }
