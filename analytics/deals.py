"""Deterministic deal / pipeline analytics.

Every figure returned by this module is computed in pandas. Functions return
plain JSON-serialisable dicts so they can be handed to the LLM as *facts to
narrate*, never as *maths to perform*.

A metric is omitted (``None``) rather than guessed whenever its input field is
unavailable, and the reason is recorded under ``unavailable_metrics``.
"""
from __future__ import annotations

import pandas as pd

from analytics.normalization import (
    NormalizedDataset,
    filter_by_date_range,
    filter_by_sector,
    sector_matches,
)
from analytics.quality import DataQualityReport, Severity
from utils.dates import DateRange, describe_range
from utils.numbers import format_inr, safe_ratio
from utils.text import PROBABILITY_BANDS, UNKNOWN

__all__ = [
    "deal_scope",
    "pipeline_summary",
    "pipeline_by_dimension",
    "top_opportunities",
    "deals_at_risk",
    "win_rate",
    "expected_close_analysis",
    "analyze_deals",
]


def _money(value: float | None) -> dict | None:
    """Money is emitted as both a raw number and a pre-formatted string.

    The formatted string is what the LLM is told to quote, which removes any
    opportunity for it to re-derive or mis-round a figure.
    """
    if value is None or pd.isna(value):
        return None
    return {"amount": round(float(value), 2), "display": format_inr(float(value))}


def _pct(value: float | None, *, digits: int = 1) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value) * 100, digits)


def _sum(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns or frame.empty:
        return None
    series = pd.to_numeric(frame[column], errors="coerce")
    return float(series.sum()) if series.notna().any() else None


def deal_scope(
    dataset: NormalizedDataset,
    *,
    sector: str | None = None,
    owner: str | None = None,
    date_range: DateRange | None = None,
    date_field: str = "expected_close_date",
    status_filter: str | None = None,
) -> dict:
    """Apply the planner's filters and describe exactly what was scoped.

    Returns the filtered frame plus a human-readable description of the scope and
    the number of rows excluded because of a missing date.
    """
    frame = dataset.frame
    report = DataQualityReport(dataset="deals_scope", row_count=len(frame))
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
            applied.append(f"sector '{sector}' (no matching deals on the board)")

    if owner and "owner" in frame.columns:
        owner_key = str(owner).strip().lower()
        matches = frame["owner"].fillna("").str.lower().str.contains(owner_key, regex=False)
        if matches.any():
            frame = frame[matches]
            applied.append(f"owner ~ {owner}")

    status_map = {
        "open": ["Open"],
        "active": ["Open"],
        "won": ["Won"],
        "lost": ["Lost"],
        "on_hold": ["On Hold"],
        "closed": ["Won", "Lost"],
        "open_and_hold": ["Open", "On Hold"],
    }
    normalized_status = (status_filter or "").strip().lower().replace(" ", "_")
    if normalized_status in status_map and "status" in frame.columns:
        frame = frame[frame["status"].isin(status_map[normalized_status])]
        applied.append(f"status = {', '.join(status_map[normalized_status])}")

    excluded_missing_date = 0
    if date_range is not None:
        frame, excluded_missing_date = filter_by_date_range(frame, date_field, date_range)
        applied.append(f"{date_field.replace('_', ' ')} within {date_range.label}")
        if excluded_missing_date:
            report.add(
                "date_filter_excluded",
                f"{excluded_missing_date} deals have no {date_field.replace('_', ' ')} and "
                f"could not be placed in {date_range.label}; the period figure may be understated",
                excluded_missing_date, Severity.EXCLUDED, total=len(dataset.frame),
                field_name=date_field,
            )

    return {
        "frame": frame,
        "sectors": sectors,
        "filters_applied": applied or ["none (all deals)"],
        "period": describe_range(date_range),
        "date_field": date_field,
        "excluded_missing_date": excluded_missing_date,
        "report": report,
        "row_count": len(frame),
    }


def pipeline_summary(dataset: NormalizedDataset, frame: pd.DataFrame | None = None) -> dict:
    """Headline pipeline / revenue metrics for a scoped set of deals."""
    frame = dataset.frame if frame is None else frame
    unavailable: dict[str, str] = {}
    total_deals = int(len(frame))

    if total_deals == 0:
        return {
            "deal_count": 0,
            "note": "No deals match this scope.",
            "unavailable_metrics": {},
        }

    has_value = dataset.available_fields.get("value", False) and "value" in frame.columns
    open_frame = frame[frame["is_open"]] if "is_open" in frame.columns else frame.iloc[0:0]
    won_frame = frame[frame["is_won"]] if "is_won" in frame.columns else frame.iloc[0:0]
    lost_frame = frame[frame["is_lost"]] if "is_lost" in frame.columns else frame.iloc[0:0]
    hold_frame = frame[frame["is_on_hold"]] if "is_on_hold" in frame.columns else frame.iloc[0:0]
    late_frame = open_frame[open_frame["is_late_stage"]] if "is_late_stage" in open_frame.columns else open_frame.iloc[0:0]

    summary: dict = {
        "deal_count": total_deals,
        "open_deal_count": int(len(open_frame)),
        "won_deal_count": int(len(won_frame)),
        "lost_deal_count": int(len(lost_frame)),
        "on_hold_deal_count": int(len(hold_frame)),
        "late_stage_open_deal_count": int(len(late_frame)),
    }

    if not has_value:
        unavailable["pipeline_value"] = (
            "The Deals board has no usable deal-value column, so value-based metrics "
            "cannot be calculated. Counts are still accurate."
        )
        summary["unavailable_metrics"] = unavailable
        return summary

    valued = frame["value"].notna()
    summary["deals_with_value"] = int(valued.sum())
    summary["deals_missing_value"] = int((~valued).sum())
    summary["value_coverage_pct"] = _pct(safe_ratio(int(valued.sum()), total_deals))

    summary["total_value"] = _money(_sum(frame, "value"))
    summary["open_pipeline_value"] = _money(_sum(open_frame, "value"))
    summary["won_value"] = _money(_sum(won_frame, "value"))
    summary["lost_value"] = _money(_sum(lost_frame, "value"))
    summary["on_hold_value"] = _money(_sum(hold_frame, "value"))
    summary["late_stage_open_value"] = _money(_sum(late_frame, "value"))

    open_value = _sum(open_frame, "value")
    late_value = _sum(late_frame, "value")
    summary["late_stage_share_of_open_pct"] = _pct(safe_ratio(late_value, open_value))
    summary["open_deals_missing_value"] = int(open_frame["value"].isna().sum()) if len(open_frame) else 0

    if len(open_frame) and open_frame["value"].notna().any():
        summary["average_open_deal_value"] = _money(float(open_frame["value"].mean()))
        summary["median_open_deal_value"] = _money(float(open_frame["value"].median()))

    # Weighted pipeline requires the categorical closure-probability field.
    if dataset.available_fields.get("probability") and "weighted_value" in open_frame.columns:
        weighted = open_frame["weighted_value"]
        if weighted.notna().any():
            summary["weighted_open_pipeline_value"] = _money(float(weighted.sum()))
            summary["weighted_pipeline_basis"] = {
                "method": "categorical closure probability mapped to a fixed weight",
                "weights": PROBABILITY_BANDS,
                "deals_weighted": int(weighted.notna().sum()),
                "deals_without_probability": int(weighted.isna().sum()),
            }
        else:
            unavailable["weighted_pipeline"] = (
                "No open deal has a closure probability recorded, so a weighted "
                "pipeline cannot be produced."
            )
    else:
        unavailable["weighted_pipeline"] = (
            "The board has no closure-probability field, so the pipeline cannot be "
            "probability-weighted."
        )

    summary["unavailable_metrics"] = unavailable
    return summary


def pipeline_by_dimension(
    dataset: NormalizedDataset,
    frame: pd.DataFrame | None = None,
    *,
    dimension: str = "sector",
    status: str | None = "Open",
    limit: int = 12,
) -> dict:
    """Group open (or any-status) pipeline by sector / stage / owner / product."""
    frame = dataset.frame if frame is None else frame
    if dimension not in frame.columns or frame.empty:
        return {
            "dimension": dimension,
            "rows": [],
            "note": f"'{dimension}' is not available on this board.",
        }

    scoped = frame[frame["status"] == status] if status and "status" in frame.columns else frame
    if scoped.empty:
        return {"dimension": dimension, "rows": [], "note": "No deals in scope."}

    has_value = "value" in scoped.columns and scoped["value"].notna().any()
    grouped = scoped.groupby(dimension, dropna=False)

    records = []
    total_value = float(scoped["value"].sum()) if has_value else None
    total_count = int(len(scoped))

    for key, group in grouped:
        value = float(group["value"].sum()) if has_value and group["value"].notna().any() else None
        record = {
            dimension: UNKNOWN if pd.isna(key) else str(key),
            "deal_count": int(len(group)),
            "count_share_pct": _pct(safe_ratio(len(group), total_count)),
            "deals_missing_value": int(group["value"].isna().sum()) if "value" in group else None,
        }
        if value is not None:
            record["value"] = _money(value)
            record["value_share_pct"] = _pct(safe_ratio(value, total_value))
        if "is_late_stage" in group.columns:
            record["late_stage_deal_count"] = int(group["is_late_stage"].sum())
        records.append(record)

    sort_key = (lambda r: -(r.get("value", {}) or {}).get("amount", 0)) if has_value else (
        lambda r: -r["deal_count"]
    )
    records.sort(key=sort_key)

    result = {
        "dimension": dimension,
        "status_scope": status or "all statuses",
        "group_count": len(records),
        "total_deal_count": total_count,
        "rows": records[:limit],
    }
    if total_value is not None:
        result["total_value"] = _money(total_value)
        top = records[:3]
        top_value = sum((r.get("value") or {}).get("amount", 0) for r in top)
        result["top3_concentration_pct"] = _pct(safe_ratio(top_value, total_value))
    if len(records) > limit:
        result["truncated"] = f"showing top {limit} of {len(records)} groups"
    return result


def top_opportunities(
    dataset: NormalizedDataset, frame: pd.DataFrame | None = None, *, limit: int = 5
) -> dict:
    """Largest open deals by value, with concentration analysis."""
    frame = dataset.frame if frame is None else frame
    if "value" not in frame.columns or frame.empty or not frame["value"].notna().any():
        return {"rows": [], "note": "Deal values are unavailable, so opportunities cannot be ranked."}

    open_frame = frame[frame["is_open"]] if "is_open" in frame.columns else frame
    open_frame = open_frame[open_frame["value"].notna()]
    if open_frame.empty:
        return {"rows": [], "note": "No open deals with a recorded value in this scope."}

    ranked = open_frame.sort_values("value", ascending=False).head(limit)
    total = float(open_frame["value"].sum())
    rows = []
    for _, row in ranked.iterrows():
        rows.append({
            "deal": row.get("deal_name") or "(unnamed)",
            "client": row.get("client"),
            "sector": row.get("sector"),
            "stage": row.get("stage"),
            "owner": row.get("owner"),
            "value": _money(row.get("value")),
            "share_of_open_pipeline_pct": _pct(safe_ratio(row.get("value"), total)),
            "expected_close": (
                row["expected_close_date"].date().isoformat()
                if pd.notna(row.get("expected_close_date")) else None
            ),
            "probability": row.get("probability"),
        })
    top_sum = float(ranked["value"].sum())
    return {
        "rows": rows,
        "open_deals_considered": int(len(open_frame)),
        "open_pipeline_value": _money(total),
        f"top{len(rows)}_value": _money(top_sum),
        f"top{len(rows)}_concentration_pct": _pct(safe_ratio(top_sum, total)),
    }


def deals_at_risk(
    dataset: NormalizedDataset,
    frame: pd.DataFrame | None = None,
    *,
    today: pd.Timestamp | None = None,
    stale_days: int = 90,
    limit: int = 10,
) -> dict:
    """Identify open deals carrying a defensible, data-supported risk signal.

    Risk signals used (all derived from recorded fields, none inferred):
    overdue expected close, low closure probability on a large deal, long time in
    funnel without a close date, and on-hold status.
    """
    frame = dataset.frame if frame is None else frame
    if frame.empty:
        return {"rows": [], "criteria": [], "note": "No deals in scope."}

    now = today or pd.Timestamp.today().normalize()
    open_frame = frame[frame["is_open"]] if "is_open" in frame.columns else frame
    hold_frame = frame[frame["is_on_hold"]] if "is_on_hold" in frame.columns else frame.iloc[0:0]

    criteria: list[str] = []
    signals: dict[int, list[str]] = {}

    def _flag(mask: pd.Series, label: str) -> None:
        if mask is None or not len(mask):
            return
        matched = mask[mask.fillna(False)].index
        if len(matched):
            criteria.append(f"{label} ({len(matched)} deals)")
        for idx in matched:
            signals.setdefault(idx, []).append(label)

    if "expected_close_date" in open_frame.columns:
        _flag(open_frame["expected_close_date"] < now, "expected close date has passed")
        _flag(open_frame["expected_close_date"].isna(), "no expected close date recorded")

    if "probability" in open_frame.columns and dataset.available_fields.get("probability"):
        _flag(open_frame["probability"] == "Low", "low closure probability")

    if "age_days" in open_frame.columns:
        _flag(open_frame["age_days"] > stale_days, f"open for more than {stale_days} days")

    if "value" in open_frame.columns:
        _flag(open_frame["value"].isna(), "no deal value recorded")

    if len(hold_frame):
        for idx in hold_frame.index:
            signals.setdefault(idx, []).append("deal is on hold")
        criteria.append(f"on hold ({len(hold_frame)} deals)")

    if not signals:
        return {"rows": [], "criteria": criteria, "at_risk_count": 0,
                "note": "No open deal triggered a risk signal in this scope."}

    at_risk = frame.loc[sorted(signals)]
    scored = at_risk.assign(
        risk_signal_count=[len(signals[i]) for i in at_risk.index],
        _sort_value=at_risk["value"].fillna(0) if "value" in at_risk.columns else 0,
    ).sort_values(["risk_signal_count", "_sort_value"], ascending=[False, False])

    rows = []
    for idx, row in scored.head(limit).iterrows():
        rows.append({
            "deal": row.get("deal_name") or "(unnamed)",
            "client": row.get("client"),
            "sector": row.get("sector"),
            "stage": row.get("stage"),
            "status": row.get("status"),
            "owner": row.get("owner"),
            "value": _money(row.get("value")),
            "expected_close": (
                row["expected_close_date"].date().isoformat()
                if pd.notna(row.get("expected_close_date")) else None
            ),
            "risk_signals": signals[idx],
        })

    value_at_risk = None
    if "value" in at_risk.columns and at_risk["value"].notna().any():
        value_at_risk = float(at_risk["value"].sum())

    return {
        "rows": rows,
        "criteria": criteria,
        "at_risk_count": int(len(at_risk)),
        "at_risk_value": _money(value_at_risk),
        "deals_considered": int(len(frame)),
        "shown": len(rows),
    }


def win_rate(dataset: NormalizedDataset, frame: pd.DataFrame | None = None) -> dict:
    """Win rate by count and by value over *closed* deals only.

    Returns ``available: False`` when the closed sample is too small to be
    meaningful rather than quoting a misleading percentage.
    """
    frame = dataset.frame if frame is None else frame
    if frame.empty or "is_won" not in frame.columns:
        return {"available": False, "reason": "Deal status is unavailable."}

    won = int(frame["is_won"].sum())
    lost = int(frame["is_lost"].sum())
    closed = won + lost
    if closed < 5:
        return {
            "available": False,
            "won_count": won,
            "lost_count": lost,
            "reason": (
                f"Only {closed} closed deals in scope; a win rate from this sample "
                f"would not be reliable."
            ),
        }

    result = {
        "available": True,
        "won_count": won,
        "lost_count": lost,
        "closed_count": closed,
        "win_rate_by_count_pct": _pct(safe_ratio(won, closed)),
        "basis": "closed deals only (Won vs Lost); open and on-hold deals excluded",
    }
    if "value" in frame.columns and frame["value"].notna().any():
        won_value = _sum(frame[frame["is_won"]], "value") or 0.0
        lost_value = _sum(frame[frame["is_lost"]], "value") or 0.0
        if won_value + lost_value > 0:
            result["won_value"] = _money(won_value)
            result["lost_value"] = _money(lost_value)
            result["win_rate_by_value_pct"] = _pct(safe_ratio(won_value, won_value + lost_value))
            missing = int(frame[frame["is_won"] | frame["is_lost"]]["value"].isna().sum())
            result["closed_deals_missing_value"] = missing
    return result


def expected_close_analysis(
    dataset: NormalizedDataset,
    frame: pd.DataFrame | None = None,
    *,
    today: pd.Timestamp | None = None,
    fy_start_month: int = 4,
    quarters: int = 4,
) -> dict:
    """Distribute open pipeline across the coming quarters by expected close date."""
    from utils.dates import quarter_range  # noqa: PLC0415 - keeps module import light

    frame = dataset.frame if frame is None else frame
    if "expected_close_date" not in frame.columns or frame.empty:
        return {"available": False, "reason": "No expected close date column on this board."}

    open_frame = frame[frame["is_open"]] if "is_open" in frame.columns else frame
    if open_frame.empty:
        return {"available": False, "reason": "No open deals in scope."}

    now = today or pd.Timestamp.today().normalize()
    buckets = []
    for offset in range(quarters):
        window = quarter_range(now, fy_start_month=fy_start_month, offset_quarters=offset)
        in_window = open_frame[
            open_frame["expected_close_date"].between(window.start, window.end).fillna(False)
        ]
        bucket = {
            "quarter": window.label,
            "deal_count": int(len(in_window)),
        }
        if "value" in in_window.columns and in_window["value"].notna().any():
            bucket["value"] = _money(float(in_window["value"].sum()))
            bucket["deals_missing_value"] = int(in_window["value"].isna().sum())
        buckets.append(bucket)

    overdue = open_frame[(open_frame["expected_close_date"] < now)]
    no_date = open_frame[open_frame["expected_close_date"].isna()]

    result = {
        "available": True,
        "quarters": buckets,
        "overdue_open_deal_count": int(len(overdue)),
        "open_deals_without_close_date": int(len(no_date)),
    }
    if "value" in open_frame.columns:
        if overdue["value"].notna().any():
            result["overdue_open_value"] = _money(float(overdue["value"].sum()))
        if no_date["value"].notna().any():
            result["value_without_close_date"] = _money(float(no_date["value"].sum()))
    return result


def analyze_deals(
    dataset: NormalizedDataset,
    *,
    sector: str | None = None,
    owner: str | None = None,
    date_range: DateRange | None = None,
    status_filter: str | None = None,
    group_by: str | None = "sector",
    today: pd.Timestamp | None = None,
    fy_start_month: int = 4,
) -> dict:
    """Full deal analysis bundle used by the agent orchestrator."""
    scope = deal_scope(
        dataset,
        sector=sector,
        owner=owner,
        date_range=date_range,
        status_filter=status_filter,
    )
    frame = scope["frame"]

    report = DataQualityReport(dataset="deals", row_count=len(dataset.frame))
    report.extend(dataset.quality)
    report.extend(scope["report"])

    facts: dict = {
        "board": "deals",
        "scope": {
            "filters_applied": scope["filters_applied"],
            "period": scope["period"],
            "deals_matching_all_filters": scope["row_count"],
            "total_deals_on_board_all_statuses": int(len(dataset.frame)),
            "open_deals_on_board_all_periods": int(dataset.frame["is_open"].sum())
            if "is_open" in dataset.frame.columns else None,
        },
        "summary": pipeline_summary(dataset, frame),
        "top_opportunities": top_opportunities(dataset, frame),
        "win_rate": win_rate(dataset, frame),
    }

    if group_by and group_by != "none":
        facts["breakdown"] = pipeline_by_dimension(dataset, frame, dimension=group_by)
    if group_by != "stage":
        facts["by_stage"] = pipeline_by_dimension(dataset, frame, dimension="stage_group")
    facts["expected_close"] = expected_close_analysis(
        dataset, frame, today=today, fy_start_month=fy_start_month
    )
    if scope["row_count"] == 0 and len(dataset.frame) > 0:
        facts["empty_scope_context"] = _empty_scope_context(
            dataset, sector=sector, date_range=date_range, today=today
        )
    facts["data_quality"] = report.as_dict(limit=6)
    return facts


def _empty_scope_context(
    dataset: NormalizedDataset,
    *,
    sector: str | None,
    date_range: DateRange | None,
    today: pd.Timestamp | None,
) -> dict:
    """Explain *why* a filtered scope is empty and where the data actually sits.

    Returning a bare zero is technically correct but useless to a founder: it
    cannot be distinguished from a broken connection. This block lets the answer
    say "nothing closes in that window, here is where the open book actually is".
    """
    frame = dataset.frame
    context: dict = {
        "reason": "No deals matched every filter applied to this question.",
        "total_deals_on_board_all_statuses": int(len(frame)),
    }
    if "is_open" in frame.columns:
        context["open_deals_on_board_all_periods"] = int(frame["is_open"].sum())

    if sector and "sector" in frame.columns:
        context["sectors_present_on_board"] = sorted(
            frame["sector"].dropna().unique().tolist()
        )[:15]

    if date_range is not None and "expected_close_date" in frame.columns:
        open_frame = frame[frame["is_open"]] if "is_open" in frame.columns else frame
        dated = open_frame["expected_close_date"].dropna()
        context["requested_window"] = date_range.as_dict()
        if len(dated):
            context["open_expected_close_dates_actually_span"] = {
                "earliest": dated.min().date().isoformat(),
                "latest": dated.max().date().isoformat(),
            }
            # Where the open book really falls, month by month.
            dated_open = open_frame.dropna(subset=["expected_close_date"])
            grouped = dated_open.groupby(dated_open["expected_close_date"].dt.to_period("M"))
            # min_count=1 keeps an all-missing month as NaN instead of summing to
            # 0.0 -- a month with no recorded values is not a zero-rupee month.
            by_month = grouped.agg(
                deal_count=("item_id", "size"),
                value=("value", lambda v: v.sum(min_count=1)),
                deals_missing_value=("value", lambda v: int(v.isna().sum())),
            ).sort_index()
            context["open_pipeline_by_expected_close_month"] = [
                {
                    "month": str(period),
                    "deal_count": int(row["deal_count"]),
                    "value": _money(row["value"]) if pd.notna(row["value"]) else None,
                    "deals_missing_value": int(row["deals_missing_value"]),
                }
                for period, row in by_month.tail(12).iterrows()
            ]
        context["open_deals_without_expected_close_date"] = int(
            open_frame["expected_close_date"].isna().sum()
        )
    return context
