"""Turn raw Monday.com board rows into clean, typed, analysis-ready DataFrames.

Guarantees
----------
* Original values are preserved. Every derived column is prefixed (``value_``,
  ``sector_norm``, ``expected_close_date``...) and the raw column stays intact.
* Nothing is coerced to zero. Unparseable numbers become ``NaN`` and are counted.
* Nothing raises. Malformed rows degrade into ``Unknown``/``NaN`` plus an entry
  in the :class:`~analytics.quality.DataQualityReport`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from analytics.quality import DataQualityReport, Severity
from monday.column_map import ColumnMapping
from utils.dates import parse_date_series
from utils.logging import get_logger
from utils.numbers import is_missing, parse_amount
from utils.text import (
    PROBABILITY_BANDS,
    SECTOR_UMBRELLAS,
    UNKNOWN,
    canonical_deal_stage,
    canonical_deal_status,
    canonical_execution_status,
    canonical_probability,
    canonical_sector,
    slugify,
)

logger = get_logger(__name__)

__all__ = [
    "NormalizedDataset",
    "normalize_deals",
    "normalize_work_orders",
    "sector_matches",
    "filter_by_sector",
    "filter_by_date_range",
]

#: Columns whose presence in *every* cell of a row means the row is a repeated
#: header accidentally pasted into the data (observed in the seed workbook).
_HEADER_ECHO_RATIO = 0.5


@dataclass
class NormalizedDataset:
    """A normalised board plus its quality report and field availability."""

    name: str
    frame: pd.DataFrame
    quality: DataQualityReport
    #: canonical field -> whether usable data actually exists for it
    available_fields: dict[str, bool] = field(default_factory=dict)
    source: dict[str, object] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return self.frame.empty

    def has(self, *fields: str) -> bool:
        return all(self.available_fields.get(f, False) for f in fields)

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self.frame)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _series(frame: pd.DataFrame, mapping: ColumnMapping, field_name: str) -> pd.Series | None:
    """Fetch the raw series backing a canonical field, or ``None`` if unmapped."""
    column = mapping.get(field_name)
    if not column or column not in frame.columns:
        return None
    return frame[column]


def _drop_header_echo_rows(frame: pd.DataFrame, report: DataQualityReport) -> pd.DataFrame:
    """Remove rows that are a repeat of the header (cell value == column title).

    The supplied workbook contains two such rows; a spreadsheet import carries
    them straight into Monday.
    """
    if frame.empty:
        return frame
    data_columns = [c for c in frame.columns if not c.startswith("__")]
    if not data_columns:
        return frame

    def _is_echo(row: pd.Series) -> bool:
        matches = sum(
            1 for c in data_columns
            if not is_missing(row.get(c)) and slugify(row.get(c)) == slugify(c)
        )
        populated = sum(1 for c in data_columns if not is_missing(row.get(c)))
        return populated > 0 and matches / populated >= _HEADER_ECHO_RATIO and matches >= 3

    mask = frame.apply(_is_echo, axis=1)
    removed = int(mask.sum())
    if removed:
        report.add(
            "header_rows_removed",
            f"{removed} row(s) repeated the column headers and were excluded",
            removed,
            Severity.EXCLUDED,
        )
    return frame.loc[~mask].copy()


def _flag_duplicates(
    frame: pd.DataFrame,
    key_columns: list[str],
    report: DataQualityReport,
    *,
    label: str,
) -> pd.DataFrame:
    """Mark (but do not drop) duplicate business records.

    Duplicates are kept in the totals because a repeated row can legitimately be
    two similar deals with the same masked name; dropping them would understate
    the pipeline. The count is surfaced so the user can judge.
    """
    frame = frame.copy()
    usable = [c for c in key_columns if c in frame.columns]
    if not usable:
        frame["is_duplicate"] = False
        return frame
    frame["is_duplicate"] = frame.duplicated(subset=usable, keep="first")
    count = int(frame["is_duplicate"].sum())
    if count:
        report.add(
            "duplicate_records",
            f"{count} {label} look like exact duplicates of an earlier row "
            f"(matched on {', '.join(usable)}); they are still counted",
            count,
            Severity.INFO,
        )
    return frame


def _parse_amount_column(
    raw: pd.Series | None,
    frame: pd.DataFrame,
    target: str,
    report: DataQualityReport,
    *,
    label: str,
    severity: Severity = Severity.EXCLUDED,
) -> bool:
    """Parse a money column into ``target``; report missing / unparseable / FX rows."""
    if raw is None:
        frame[target] = pd.Series([float("nan")] * len(frame), index=frame.index, dtype="float64")
        frame[f"{target}_missing_reason"] = "column_not_on_board"
        return False

    parsed = [parse_amount(v) for v in raw]
    values, reasons = [], []
    foreign = 0
    for item in parsed:
        if item.ok and not item.is_base_currency:
            # Parsed, but in a currency we refuse to convert.
            foreign += 1
            values.append(float("nan"))
            reasons.append(f"non_inr_currency:{item.currency}")
        elif item.ok:
            values.append(float(item.value))
            reasons.append(None)
        else:
            values.append(float("nan"))
            reasons.append(item.reason)

    frame[target] = pd.Series(values, index=frame.index, dtype="float64")
    frame[f"{target}_missing_reason"] = pd.Series(reasons, index=frame.index, dtype="object")

    total = len(frame)
    missing = sum(1 for r in reasons if r == "missing")
    unparseable = sum(1 for r in reasons if r and r not in {"missing"} and not r.startswith("non_inr"))
    field_label = label_field(target)
    report.add(
        f"missing_{target}",
        f"{missing} of {total} {label} have no {field_label} recorded "
        f"(they are excluded from {field_label} totals, not counted as ₹0)",
        missing, severity, total=total, field_name=target,
    )
    report.add(
        f"unparseable_{target}",
        f"{unparseable} {label} have a {field_label} that could not be parsed safely",
        unparseable, Severity.EXCLUDED, total=total, field_name=target,
    )
    report.add(
        f"non_inr_{target}",
        f"{foreign} {label} record their {field_label} in a non-INR currency and are "
        f"excluded "
        f"(no exchange rate is assumed)",
        foreign, Severity.EXCLUDED, total=total, field_name=target,
    )
    return bool(frame[target].notna().any())


def _parse_date_column(
    raw: pd.Series | None,
    frame: pd.DataFrame,
    target: str,
    report: DataQualityReport,
    *,
    label: str,
    severity: Severity = Severity.INCLUDED_WITH_GAP,
) -> bool:
    if raw is None:
        frame[target] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
        return False
    values, reasons = parse_date_series(raw)
    frame[target] = values
    frame[f"{target}_missing_reason"] = reasons

    total = len(frame)
    missing = int((reasons == "missing").sum())
    invalid = int((reasons == "invalid").sum())
    report.add(
        f"missing_{target}",
        f"{missing} of {total} {label} have no {label_field(target)}",
        missing, severity, total=total, field_name=target,
    )
    report.add(
        f"invalid_{target}",
        f"{invalid} {label} have an unreadable {label_field(target)} value",
        invalid, Severity.EXCLUDED, total=total, field_name=target,
    )
    return bool(frame[target].notna().any())


def label_field(target: str) -> str:
    return target.replace("_", " ")


def _normalize_labels(
    raw: pd.Series | None,
    frame: pd.DataFrame,
    target: str,
    fn,
    report: DataQualityReport,
    *,
    label: str,
    severity: Severity = Severity.INCLUDED_WITH_GAP,
) -> bool:
    """Apply a canonicaliser to a text column, tracking unknown/missing counts."""
    if raw is None:
        frame[target] = UNKNOWN
        return False
    canonical = [fn(v) for v in raw]
    frame[target] = [c.value for c in canonical]
    total = len(frame)
    missing = sum(1 for c in canonical if c.method == "missing")
    unknown = sum(1 for c in canonical if c.value == UNKNOWN and c.method != "missing")
    fuzzy = sum(1 for c in canonical if c.method == "fuzzy")

    report.add(
        f"missing_{target}",
        f"{missing} of {total} {label} have no {label_field(target).replace(' norm', '')} recorded",
        missing, severity, total=total, field_name=target,
    )
    report.add(
        f"unrecognised_{target}",
        f"{unknown} {label} have a {label_field(target).replace(' norm', '')} value that "
        f"did not map to a known category",
        unknown, Severity.INCLUDED_WITH_GAP, total=total, field_name=target,
    )
    report.add(
        f"fuzzy_{target}",
        f"{fuzzy} {label} had their {label_field(target).replace(' norm', '')} matched "
        f"approximately (possible typos in the source board)",
        fuzzy, Severity.INFO, total=total, field_name=target,
    )
    return bool((frame[target] != UNKNOWN).any())


def _copy_raw(raw: pd.Series | None, frame: pd.DataFrame, target: str) -> bool:
    if raw is None:
        frame[target] = None
        return False
    frame[target] = [None if is_missing(v) else str(v).strip() for v in raw]
    return bool(frame[target].notna().any())


# ---------------------------------------------------------------------------
# Deals
# ---------------------------------------------------------------------------

def normalize_deals(
    raw_frame: pd.DataFrame,
    mapping: ColumnMapping,
    *,
    today: pd.Timestamp | None = None,
) -> NormalizedDataset:
    """Normalise the Deals board into an analysis-ready dataset."""
    report = DataQualityReport(dataset="deals", row_count=len(raw_frame))
    if raw_frame.empty:
        report.add(
            "empty_board",
            "The Deals board returned no items.",
            1, Severity.EXCLUDED, total=1,
        )
        return NormalizedDataset("deals", _empty_deals_frame(), report, {}, {})

    frame = _drop_header_echo_rows(raw_frame, report)
    report.row_count = len(frame)
    frame = frame.reset_index(drop=True)
    out = pd.DataFrame(index=frame.index)
    out["item_id"] = frame.get("__item_id")
    available: dict[str, bool] = {}

    available["deal_name"] = _copy_raw(_series(frame, mapping, "deal_name"), out, "deal_name")
    if out["deal_name"].isna().all() and "__item_name" in frame.columns:
        out["deal_name"] = frame["__item_name"]
        available["deal_name"] = bool(out["deal_name"].notna().any())
    available["owner"] = _copy_raw(_series(frame, mapping, "owner"), out, "owner")
    available["client"] = _copy_raw(_series(frame, mapping, "client"), out, "client")
    available["product"] = _copy_raw(_series(frame, mapping, "product"), out, "product")
    out["status_raw"] = _series(frame, mapping, "status") if mapping.has("status") else None
    out["stage_raw"] = _series(frame, mapping, "stage") if mapping.has("stage") else None
    out["sector_raw"] = _series(frame, mapping, "sector") if mapping.has("sector") else None

    available["sector"] = _normalize_labels(
        _series(frame, mapping, "sector"), out, "sector", canonical_sector, report,
        label="deals",
    )
    available["status"] = _normalize_labels(
        _series(frame, mapping, "status"), out, "status", canonical_deal_status, report,
        label="deals",
    )
    available["probability"] = _normalize_labels(
        _series(frame, mapping, "probability"), out, "probability", canonical_probability,
        report, label="deals", severity=Severity.INFO,
    )

    # Stage carries three derived columns (label / funnel group / ordering).
    stage_raw = _series(frame, mapping, "stage")
    if stage_raw is None:
        out["stage"] = UNKNOWN
        out["stage_group"] = "unknown"
        out["stage_order"] = 99.0
        available["stage"] = False
    else:
        stages = [canonical_deal_stage(v) for v in stage_raw]
        out["stage"] = [s.label for s in stages]
        out["stage_group"] = [s.group for s in stages]
        out["stage_order"] = [s.order for s in stages]
        unknown_stage = sum(1 for s in stages if not s.is_known)
        report.add(
            "unknown_stage",
            f"{unknown_stage} deals have a funnel stage that could not be classified",
            unknown_stage, Severity.INCLUDED_WITH_GAP, total=len(out), field_name="stage",
        )
        available["stage"] = bool(unknown_stage < len(stages))

    available["value"] = _parse_amount_column(
        _series(frame, mapping, "value"), out, "value", report, label="deals",
    )
    available["expected_close_date"] = _parse_date_column(
        _series(frame, mapping, "expected_close_date"), out, "expected_close_date", report,
        label="deals",
    )
    available["actual_close_date"] = _parse_date_column(
        _series(frame, mapping, "actual_close_date"), out, "actual_close_date", report,
        label="deals", severity=Severity.INFO,
    )
    available["created_date"] = _parse_date_column(
        _series(frame, mapping, "created_date"), out, "created_date", report,
        label="deals", severity=Severity.INFO,
    )

    # Status is authoritative; fall back to the funnel stage when it is absent.
    fallback = out["status"] == UNKNOWN
    stage_implied = out["stage_group"].map(
        {"won": "Won", "lost": "Lost", "on_hold": "On Hold"}
    ).fillna("Open")
    filled = int((fallback & (out["stage_group"] != "unknown")).sum())
    out.loc[fallback & (out["stage_group"] != "unknown"), "status"] = stage_implied[
        fallback & (out["stage_group"] != "unknown")
    ]
    out["status_inferred"] = False
    out.loc[fallback & (out["stage_group"] != "unknown"), "status_inferred"] = True
    report.add(
        "status_inferred_from_stage",
        f"{filled} deals had no status and were classified from their funnel stage",
        filled, Severity.INCLUDED_WITH_GAP, total=len(out), field_name="status",
    )

    out["is_open"] = out["status"] == "Open"
    out["is_won"] = out["status"] == "Won"
    out["is_lost"] = out["status"] == "Lost"
    out["is_on_hold"] = out["status"] == "On Hold"
    out["is_late_stage"] = out["stage_group"] == "late"
    out["probability_weight"] = out["probability"].map(PROBABILITY_BANDS)
    out["weighted_value"] = out["value"] * out["probability_weight"]

    now = today or pd.Timestamp.today().normalize()
    out["days_to_expected_close"] = (out["expected_close_date"] - now).dt.days
    out["is_overdue_close"] = out["is_open"] & (out["expected_close_date"] < now)
    if "created_date" in out:
        out["age_days"] = (now - out["created_date"]).dt.days

    out = _flag_duplicates(
        out,
        ["deal_name", "client", "value", "expected_close_date", "stage"],
        report,
        label="deals",
    )

    stale = int(out["is_overdue_close"].fillna(False).sum())
    report.add(
        "overdue_open_deals",
        f"{stale} open deals have an expected close date in the past "
        f"(the forecast date may be stale)",
        stale, Severity.INCLUDED_WITH_GAP, total=len(out), field_name="expected_close_date",
    )

    source = {
        "board_id": mapping.board_id,
        "board_name": mapping.board_name,
        "mapped_fields": mapping.describe(),
        "unmapped_fields": mapping.unmapped_fields,
    }
    return NormalizedDataset("deals", out, report, available, source)


def _empty_deals_frame() -> pd.DataFrame:
    columns = [
        "item_id", "deal_name", "owner", "client", "product", "sector", "status",
        "stage", "stage_group", "stage_order", "probability", "value",
        "expected_close_date", "actual_close_date", "created_date", "is_open",
        "is_won", "is_lost", "is_on_hold", "is_late_stage", "weighted_value",
        "is_duplicate", "is_overdue_close",
    ]
    return pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# Work orders
# ---------------------------------------------------------------------------

def normalize_work_orders(
    raw_frame: pd.DataFrame,
    mapping: ColumnMapping,
    *,
    today: pd.Timestamp | None = None,
    delay_grace_days: int = 0,
) -> NormalizedDataset:
    """Normalise the Work Orders board into an analysis-ready dataset."""
    report = DataQualityReport(dataset="work_orders", row_count=len(raw_frame))
    if raw_frame.empty:
        report.add(
            "empty_board", "The Work Orders board returned no items.", 1,
            Severity.EXCLUDED, total=1,
        )
        return NormalizedDataset("work_orders", _empty_wo_frame(), report, {}, {})

    frame = _drop_header_echo_rows(raw_frame, report)
    report.row_count = len(frame)
    frame = frame.reset_index(drop=True)
    out = pd.DataFrame(index=frame.index)
    out["item_id"] = frame.get("__item_id")
    available: dict[str, bool] = {}

    available["work_order_id"] = _copy_raw(
        _series(frame, mapping, "work_order_id"), out, "work_order_id"
    )
    if out["work_order_id"].isna().all() and "__item_name" in frame.columns:
        out["work_order_id"] = frame["__item_name"]
    available["deal_name"] = _copy_raw(_series(frame, mapping, "deal_name"), out, "deal_name")
    available["client"] = _copy_raw(_series(frame, mapping, "client"), out, "client")
    available["owner"] = _copy_raw(_series(frame, mapping, "owner"), out, "owner")
    available["nature_of_work"] = _copy_raw(
        _series(frame, mapping, "nature_of_work"), out, "nature_of_work"
    )
    available["type_of_work"] = _copy_raw(
        _series(frame, mapping, "type_of_work"), out, "type_of_work"
    )
    available["invoice_status"] = _copy_raw(
        _series(frame, mapping, "invoice_status"), out, "invoice_status"
    )
    available["wo_status"] = _copy_raw(_series(frame, mapping, "wo_status"), out, "wo_status")
    out["sector_raw"] = _series(frame, mapping, "sector") if mapping.has("sector") else None
    out["execution_status_raw"] = (
        _series(frame, mapping, "execution_status") if mapping.has("execution_status") else None
    )

    available["sector"] = _normalize_labels(
        _series(frame, mapping, "sector"), out, "sector", canonical_sector, report,
        label="work orders",
    )
    available["execution_status"] = _normalize_labels(
        _series(frame, mapping, "execution_status"), out, "execution_status",
        canonical_execution_status, report, label="work orders",
    )

    available["order_value"] = _parse_amount_column(
        _series(frame, mapping, "order_value"), out, "order_value", report,
        label="work orders",
    )
    available["billed_value"] = _parse_amount_column(
        _series(frame, mapping, "billed_value"), out, "billed_value", report,
        label="work orders", severity=Severity.INFO,
    )
    available["collected_value"] = _parse_amount_column(
        _series(frame, mapping, "collected_value"), out, "collected_value", report,
        label="work orders", severity=Severity.INFO,
    )
    available["receivable_value"] = _parse_amount_column(
        _series(frame, mapping, "receivable_value"), out, "receivable_value", report,
        label="work orders", severity=Severity.INFO,
    )
    available["to_be_billed_value"] = _parse_amount_column(
        _series(frame, mapping, "to_be_billed_value"), out, "to_be_billed_value", report,
        label="work orders", severity=Severity.INFO,
    )

    available["start_date"] = _parse_date_column(
        _series(frame, mapping, "start_date"), out, "start_date", report,
        label="work orders", severity=Severity.INFO,
    )
    available["end_date"] = _parse_date_column(
        _series(frame, mapping, "end_date"), out, "end_date", report, label="work orders",
    )
    available["po_date"] = _parse_date_column(
        _series(frame, mapping, "po_date"), out, "po_date", report,
        label="work orders", severity=Severity.INFO,
    )
    available["delivery_date"] = _parse_date_column(
        _series(frame, mapping, "delivery_date"), out, "delivery_date", report,
        label="work orders", severity=Severity.INFO,
    )

    out["is_completed"] = out["execution_status"] == "Completed"
    out["is_blocked"] = out["execution_status"] == "Blocked"
    out["is_not_started"] = out["execution_status"] == "Not Started"
    out["is_in_progress"] = out["execution_status"] == "In Progress"
    # "Active" = still consuming (or about to consume) delivery capacity.
    out["is_active"] = out["is_in_progress"] | out["is_not_started"] | out["is_blocked"]

    now = today or pd.Timestamp.today().normalize()
    cutoff = now - pd.Timedelta(days=int(delay_grace_days))
    # Delayed = still open and the planned end date has passed.
    # Delay is only assessable when the status is known and not complete; an
    # unrecognised status tells us nothing about whether work is still running.
    out["is_delayed"] = out["is_active"] & out["end_date"].notna() & (out["end_date"] < cutoff)
    out["days_overdue"] = (now - out["end_date"]).dt.days.where(out["is_delayed"])
    out["days_to_end"] = (out["end_date"] - now).dt.days

    if available.get("order_value") and available.get("billed_value"):
        out["unbilled_value"] = (out["order_value"] - out["billed_value"]).clip(lower=0)
    else:
        out["unbilled_value"] = pd.Series([float("nan")] * len(out), index=out.index, dtype="float64")

    out = _flag_duplicates(out, ["work_order_id"], report, label="work orders")

    incomplete_mask = out["execution_status"].eq(UNKNOWN) | out["end_date"].isna()
    if available.get("order_value"):
        incomplete_mask = incomplete_mask | out["order_value"].isna()
    incomplete = int(incomplete_mask.sum())
    report.add(
        "incomplete_work_orders",
        f"{incomplete} work orders are missing at least one core field "
        f"(status, planned end date or order value)",
        incomplete, Severity.INCLUDED_WITH_GAP, total=len(out),
    )

    no_end_date_open = int((out["is_active"] & out["end_date"].isna()).sum())
    report.add(
        "open_wo_without_end_date",
        f"{no_end_date_open} open work orders have no planned end date, so they cannot "
        f"be assessed for delay",
        no_end_date_open, Severity.EXCLUDED, total=len(out), field_name="end_date",
    )

    source = {
        "board_id": mapping.board_id,
        "board_name": mapping.board_name,
        "mapped_fields": mapping.describe(),
        "unmapped_fields": mapping.unmapped_fields,
    }
    return NormalizedDataset("work_orders", out, report, available, source)


def _empty_wo_frame() -> pd.DataFrame:
    columns = [
        "item_id", "work_order_id", "deal_name", "client", "owner", "sector",
        "execution_status", "nature_of_work", "type_of_work", "order_value",
        "billed_value", "collected_value", "receivable_value", "start_date",
        "end_date", "po_date", "delivery_date", "is_completed", "is_active",
        "is_delayed", "is_blocked", "days_overdue", "is_duplicate",
    ]
    return pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# filtering helpers shared by the analytics modules
# ---------------------------------------------------------------------------

def sector_matches(requested: str | None, available_sectors: list[str]) -> list[str]:
    """Resolve a user-supplied sector token to the canonical sectors present.

    Handles umbrella terms: asking about "energy" on a board whose sectors are
    ``Renewables`` and ``Powerline`` returns both, rather than nothing.
    """
    if not requested:
        return []
    canonical = canonical_sector(requested).value
    present = {s: s for s in available_sectors}
    matched = [s for s in available_sectors if s == canonical]

    if canonical in SECTOR_UMBRELLAS:
        for member in SECTOR_UMBRELLAS[canonical]:
            if member in present and member not in matched:
                matched.append(member)
    if matched:
        return matched

    key = slugify(requested)
    loose = [s for s in available_sectors if key and key in slugify(s)]
    return loose


def filter_by_sector(frame: pd.DataFrame, sectors: list[str]) -> pd.DataFrame:
    if not sectors or "sector" not in frame.columns:
        return frame
    return frame[frame["sector"].isin(sectors)]


def filter_by_date_range(frame: pd.DataFrame, column: str, date_range) -> tuple[pd.DataFrame, int]:
    """Filter rows into ``date_range`` on ``column``.

    Returns ``(filtered_frame, rows_excluded_for_missing_date)`` so callers can
    warn that a period-specific figure may be understated.
    """
    if date_range is None or column not in frame.columns:
        return frame, 0
    missing = int(frame[column].isna().sum())
    mask = frame[column].between(date_range.start, date_range.end)
    return frame[mask.fillna(False)], missing
