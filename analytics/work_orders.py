"""Deterministic work-order / operations analytics.

Mirrors :mod:`analytics.deals`: pandas does all the arithmetic and every metric
that cannot be supported by the board's actual columns is reported as
unavailable rather than estimated.
"""
from __future__ import annotations

import pandas as pd

from analytics.deals import _money, _pct, _sum
from analytics.normalization import (
    NormalizedDataset,
    filter_by_date_range,
    filter_by_sector,
    sector_matches,
)
from analytics.quality import DataQualityReport, Severity
from utils.dates import DateRange, describe_range
from utils.numbers import safe_ratio
from utils.text import UNKNOWN

__all__ = [
    "work_order_scope",
    "operations_summary",
    "work_orders_by_dimension",
    "delayed_work_orders",
    "upcoming_completions",
    "billing_summary",
    "analyze_work_orders",
]


def work_order_scope(
    dataset: NormalizedDataset,
    *,
    sector: str | None = None,
    owner: str | None = None,
    date_range: DateRange | None = None,
    date_field: str = "end_date",
    status_filter: str | None = None,
) -> dict:
    frame = dataset.frame
    report = DataQualityReport(dataset="work_orders_scope", row_count=len(frame))
    applied: list[str] = []

    sectors: list[str] = []
    if sector and "sector" in frame.columns:
        sectors = sector_matches(sector, sorted(frame["sector"].dropna().unique().tolist()))
        if sectors:
            frame = filter_by_sector(frame, sectors)
            applied.append(f"sector = {', '.join(sectors)}")
        else:
            # The user named a sector that is not on the board: return an empty
            # scope rather than silently answering about every sector.
            frame = frame.iloc[0:0]
            applied.append(f"sector '{sector}' (no matching work orders on the board)")

    if owner and "owner" in frame.columns:
        owner_key = str(owner).strip().lower()
        matches = frame["owner"].fillna("").str.lower().str.contains(owner_key, regex=False)
        if matches.any():
            frame = frame[matches]
            applied.append(f"owner ~ {owner}")

    status_map = {
        "active": ["Not Started", "In Progress", "Blocked"],
        "open": ["Not Started", "In Progress", "Blocked"],
        "in_progress": ["In Progress"],
        "completed": ["Completed"],
        "closed": ["Completed"],
        "not_started": ["Not Started"],
        "blocked": ["Blocked"],
    }
    key = (status_filter or "").strip().lower().replace(" ", "_")
    if key in status_map and "execution_status" in frame.columns:
        frame = frame[frame["execution_status"].isin(status_map[key])]
        applied.append(f"execution status = {', '.join(status_map[key])}")

    excluded_missing_date = 0
    if date_range is not None:
        frame, excluded_missing_date = filter_by_date_range(frame, date_field, date_range)
        applied.append(f"{date_field.replace('_', ' ')} within {date_range.label}")
        if excluded_missing_date:
            report.add(
                "date_filter_excluded",
                f"{excluded_missing_date} work orders have no {date_field.replace('_', ' ')} "
                f"and could not be placed in {date_range.label}",
                excluded_missing_date, Severity.EXCLUDED, total=len(dataset.frame),
                field_name=date_field,
            )

    return {
        "frame": frame,
        "sectors": sectors,
        "filters_applied": applied or ["none (all work orders)"],
        "period": describe_range(date_range),
        "excluded_missing_date": excluded_missing_date,
        "report": report,
        "row_count": len(frame),
    }


def operations_summary(dataset: NormalizedDataset, frame: pd.DataFrame | None = None) -> dict:
    """Headline operational counts and values."""
    frame = dataset.frame if frame is None else frame
    unavailable: dict[str, str] = {}
    total = int(len(frame))
    if total == 0:
        return {"work_order_count": 0, "note": "No work orders match this scope.",
                "unavailable_metrics": {}}

    summary: dict = {"work_order_count": total}
    for flag, label in (
        ("is_active", "active_work_orders"),
        ("is_completed", "completed_work_orders"),
        ("is_in_progress", "in_progress_work_orders"),
        ("is_not_started", "not_started_work_orders"),
        ("is_blocked", "blocked_work_orders"),
        ("is_delayed", "delayed_work_orders"),
    ):
        if flag in frame.columns:
            summary[label] = int(frame[flag].fillna(False).sum())

    if "execution_status" in frame.columns:
        summary["unknown_status_work_orders"] = int((frame["execution_status"] == UNKNOWN).sum())

    if summary.get("active_work_orders") is not None:
        summary["active_share_pct"] = _pct(safe_ratio(summary["active_work_orders"], total))
    if summary.get("delayed_work_orders") is not None:
        assessable = int(
            (frame["is_active"] & frame["end_date"].notna()).sum()
        ) if "end_date" in frame.columns else 0
        summary["delayed_share_of_assessable_pct"] = _pct(
            safe_ratio(summary["delayed_work_orders"], assessable)
        )
        summary["open_work_orders_assessable_for_delay"] = assessable

    if dataset.available_fields.get("order_value") and "order_value" in frame.columns:
        summary["total_order_value"] = _money(_sum(frame, "order_value"))
        active = frame[frame["is_active"]] if "is_active" in frame.columns else frame.iloc[0:0]
        summary["active_order_value"] = _money(_sum(active, "order_value"))
        completed = frame[frame["is_completed"]] if "is_completed" in frame.columns else frame.iloc[0:0]
        summary["completed_order_value"] = _money(_sum(completed, "order_value"))
        summary["work_orders_missing_value"] = int(frame["order_value"].isna().sum())
        if "is_delayed" in frame.columns:
            delayed = frame[frame["is_delayed"].fillna(False)]
            summary["delayed_order_value"] = _money(_sum(delayed, "order_value"))
    else:
        unavailable["order_value"] = (
            "The Work Orders board has no usable order-value column, so operational "
            "value cannot be quantified. Counts remain accurate."
        )

    if "days_overdue" in frame.columns and frame["days_overdue"].notna().any():
        summary["average_days_overdue"] = round(float(frame["days_overdue"].mean()), 1)
        summary["max_days_overdue"] = int(frame["days_overdue"].max())

    summary["unavailable_metrics"] = unavailable
    return summary


def work_orders_by_dimension(
    dataset: NormalizedDataset,
    frame: pd.DataFrame | None = None,
    *,
    dimension: str = "sector",
    limit: int = 12,
) -> dict:
    """Group work orders by sector / status / owner / nature of work."""
    frame = dataset.frame if frame is None else frame
    if dimension not in frame.columns or frame.empty:
        return {"dimension": dimension, "rows": [],
                "note": f"'{dimension}' is not available on this board."}

    has_value = "order_value" in frame.columns and frame["order_value"].notna().any()
    total_count = int(len(frame))
    total_value = float(frame["order_value"].sum()) if has_value else None

    records = []
    for key, group in frame.groupby(dimension, dropna=False):
        record = {
            dimension: UNKNOWN if pd.isna(key) else str(key),
            "work_order_count": int(len(group)),
            "count_share_pct": _pct(safe_ratio(len(group), total_count)),
        }
        for flag, label in (
            ("is_active", "active"), ("is_completed", "completed"), ("is_delayed", "delayed")
        ):
            if flag in group.columns:
                record[label] = int(group[flag].fillna(False).sum())
        if has_value and group["order_value"].notna().any():
            value = float(group["order_value"].sum())
            record["order_value"] = _money(value)
            record["value_share_pct"] = _pct(safe_ratio(value, total_value))
        records.append(record)

    records.sort(
        key=(lambda r: -(r.get("order_value", {}) or {}).get("amount", 0)) if has_value
        else (lambda r: -r["work_order_count"])
    )
    result = {
        "dimension": dimension,
        "group_count": len(records),
        "total_work_order_count": total_count,
        "rows": records[:limit],
    }
    if total_value is not None:
        result["total_order_value"] = _money(total_value)
    if len(records) > limit:
        result["truncated"] = f"showing top {limit} of {len(records)} groups"
    return result


def delayed_work_orders(
    dataset: NormalizedDataset, frame: pd.DataFrame | None = None, *, limit: int = 10
) -> dict:
    """Work orders past their planned end date and not yet complete."""
    frame = dataset.frame if frame is None else frame
    if "is_delayed" not in frame.columns or frame.empty:
        return {"rows": [], "note": "Delay cannot be assessed without a planned end date column."}

    delayed = frame[frame["is_delayed"].fillna(False)]
    if delayed.empty:
        assessable = int((frame["is_active"] & frame["end_date"].notna()).sum())
        return {
            "rows": [], "delayed_count": 0,
            "assessable_count": assessable,
            "note": "No open work order is past its planned end date in this scope.",
        }

    ranked = delayed.sort_values("days_overdue", ascending=False)
    rows = []
    for _, row in ranked.head(limit).iterrows():
        rows.append({
            "work_order": row.get("work_order_id") or row.get("deal_name") or "(unidentified)",
            "deal": row.get("deal_name"),
            "client": row.get("client"),
            "sector": row.get("sector"),
            "status": row.get("execution_status"),
            "owner": row.get("owner"),
            "planned_end": (
                row["end_date"].date().isoformat() if pd.notna(row.get("end_date")) else None
            ),
            "days_overdue": int(row["days_overdue"]) if pd.notna(row.get("days_overdue")) else None,
            "order_value": _money(row.get("order_value")),
        })

    assessable = int((frame["is_active"] & frame["end_date"].notna()).sum())
    result = {
        "rows": rows,
        "delayed_count": int(len(delayed)),
        "assessable_count": assessable,
        "delayed_share_pct": _pct(safe_ratio(len(delayed), assessable)),
        "shown": len(rows),
    }
    if "order_value" in delayed.columns and delayed["order_value"].notna().any():
        result["delayed_value"] = _money(float(delayed["order_value"].sum()))
        result["delayed_missing_value_count"] = int(delayed["order_value"].isna().sum())
    if "days_overdue" in delayed.columns and delayed["days_overdue"].notna().any():
        result["average_days_overdue"] = round(float(delayed["days_overdue"].mean()), 1)
    by_sector = work_orders_by_dimension(dataset, delayed, dimension="sector", limit=6)
    result["delayed_by_sector"] = by_sector.get("rows", [])
    return result


def upcoming_completions(
    dataset: NormalizedDataset,
    frame: pd.DataFrame | None = None,
    *,
    days: int = 30,
    limit: int = 10,
) -> dict:
    """Open work orders scheduled to finish within ``days``."""
    frame = dataset.frame if frame is None else frame
    if "days_to_end" not in frame.columns or frame.empty:
        return {"rows": [], "note": "No planned end date column available."}

    upcoming = frame[
        frame["is_active"].fillna(False)
        & frame["days_to_end"].between(0, days).fillna(False)
    ].sort_values("days_to_end")
    rows = []
    for _, row in upcoming.head(limit).iterrows():
        rows.append({
            "work_order": row.get("work_order_id") or row.get("deal_name"),
            "client": row.get("client"),
            "sector": row.get("sector"),
            "planned_end": (
                row["end_date"].date().isoformat() if pd.notna(row.get("end_date")) else None
            ),
            "days_remaining": int(row["days_to_end"]) if pd.notna(row.get("days_to_end")) else None,
            "order_value": _money(row.get("order_value")),
        })
    result = {"window_days": days, "count": int(len(upcoming)), "rows": rows}
    if "order_value" in upcoming.columns and upcoming["order_value"].notna().any():
        result["value"] = _money(float(upcoming["order_value"].sum()))
    return result


def billing_summary(dataset: NormalizedDataset, frame: pd.DataFrame | None = None) -> dict:
    """Order-book vs billed vs collected, when those columns exist."""
    frame = dataset.frame if frame is None else frame
    if frame.empty:
        return {"available": False, "reason": "No work orders in scope."}

    fields = {
        "order_value": "total_order_value",
        "billed_value": "total_billed_value",
        "collected_value": "total_collected_value",
        "receivable_value": "total_receivable_value",
        "to_be_billed_value": "total_to_be_billed_value",
    }
    present = {
        target: _money(_sum(frame, source))
        for source, target in fields.items()
        if dataset.available_fields.get(source) and source in frame.columns
    }
    present = {k: v for k, v in present.items() if v is not None}
    if not present:
        return {"available": False, "reason": "No billing columns are available on this board."}

    result: dict = {"available": True, **present}
    order = present.get("total_order_value", {}).get("amount")
    billed = present.get("total_billed_value", {}).get("amount")
    collected = present.get("total_collected_value", {}).get("amount")
    if order and billed is not None:
        result["billed_share_of_order_book_pct"] = _pct(safe_ratio(billed, order))
    if billed and collected is not None:
        result["collected_share_of_billed_pct"] = _pct(safe_ratio(collected, billed))
    result["caveat"] = (
        "Billed and collected figures are recorded inclusive of GST on this board "
        "while order value is exclusive of GST, so the ratios are indicative only."
    )
    return result


def analyze_work_orders(
    dataset: NormalizedDataset,
    *,
    sector: str | None = None,
    owner: str | None = None,
    date_range: DateRange | None = None,
    status_filter: str | None = None,
    group_by: str | None = "sector",
    today: pd.Timestamp | None = None,
) -> dict:
    """Full work-order analysis bundle used by the agent orchestrator."""
    scope = work_order_scope(
        dataset, sector=sector, owner=owner, date_range=date_range, status_filter=status_filter
    )
    frame = scope["frame"]

    report = DataQualityReport(dataset="work_orders", row_count=len(dataset.frame))
    report.extend(dataset.quality)
    report.extend(scope["report"])

    facts: dict = {
        "board": "work_orders",
        "scope": {
            "filters_applied": scope["filters_applied"],
            "period": scope["period"],
            "work_orders_in_scope": scope["row_count"],
            "work_orders_on_board": int(len(dataset.frame)),
        },
        "summary": operations_summary(dataset, frame),
        "delayed": delayed_work_orders(dataset, frame),
        "upcoming_completions": upcoming_completions(dataset, frame),
        "billing": billing_summary(dataset, frame),
    }
    if group_by and group_by != "none":
        facts["breakdown"] = work_orders_by_dimension(dataset, frame, dimension=group_by)
    facts["by_status"] = work_orders_by_dimension(dataset, frame, dimension="execution_status")
    facts["data_quality"] = report.as_dict(limit=6)
    return facts
