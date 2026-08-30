"""Column mapping layer.

Monday.com boards created by importing a spreadsheet keep the spreadsheet's
column *titles* but generate opaque column *IDs* (``text0``, ``date4``,
``numbers``...). The analytics layer needs stable canonical field names, so this
module resolves ``canonical field -> actual board column title`` at runtime by:

1. exact (slugified) title match against a curated alias table built from the
   supplied seed files,
2. substring / token-overlap scoring,
3. fuzzy string similarity,
4. an optional column-type preference (a date field should prefer a ``date``
   column over a text column of the same name).

Anything unresolved is reported so the UI can show which fields were found and
which analytics are consequently unavailable -- we never silently invent a column.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from monday.schemas import BoardColumn
from utils.logging import get_logger
from utils.text import slugify

logger = get_logger(__name__)

__all__ = ["FieldSpec", "ColumnMapping", "DEAL_FIELDS", "WORK_ORDER_FIELDS", "resolve_columns"]


@dataclass(frozen=True)
class FieldSpec:
    """Definition of one canonical field the analytics layer wants."""

    name: str
    #: Candidate titles, best first. Matched case/punctuation-insensitively.
    aliases: tuple[str, ...]
    #: ``date`` | ``numeric`` | ``text`` -- used to break ties between columns.
    kind: str = "text"
    required: bool = False
    description: str = ""


#: Canonical fields for the Deals board (aliases seeded from "Deal funnel Data.xlsx").
DEAL_FIELDS: tuple[FieldSpec, ...] = (
    # Not marked required: Monday always supplies an item name, which the
    # normaliser falls back to when no dedicated deal-name column exists.
    FieldSpec("deal_name", ("Deal Name", "Deal", "Opportunity", "Name", "Deal name masked"),
              "text", False, "Deal identifier"),
    FieldSpec("owner", ("Owner code", "Owner", "Sales Owner", "Deal Owner", "Account Owner",
                        "BD/KAM Personnel code", "Sales Rep", "Rep", "Assigned To"),
              "text", False, "Sales owner / rep"),
    FieldSpec("client", ("Client Code", "Client", "Customer", "Customer Name Code",
                         "Account", "Company", "Customer Name"),
              "text", False, "Client / account"),
    FieldSpec("status", ("Deal Status", "Status", "Stage Status", "Deal State"),
              "text", False, "Open / Won / Lost / On Hold"),
    FieldSpec("stage", ("Deal Stage", "Stage", "Funnel Stage", "Pipeline Stage", "Sales Stage"),
              "text", False, "Funnel stage"),
    FieldSpec("value", ("Masked Deal value", "Deal Value", "Value", "Amount", "Deal Amount",
                        "Deal Size", "Opportunity Value", "Revenue"),
              "numeric", False, "Deal value"),
    FieldSpec("probability", ("Closure Probability", "Probability", "Win Probability",
                              "Confidence", "Likelihood"),
              "text", False, "Closure probability"),
    FieldSpec("expected_close_date", ("Tentative Close Date", "Expected Close Date",
                                      "Expected Close", "Close Date (E)", "Target Close Date",
                                      "Projected Close Date", "Estimated Close Date"),
              "date", False, "Expected close date"),
    FieldSpec("actual_close_date", ("Close Date (A)", "Actual Close Date", "Closed Date",
                                    "Won Date", "Close Date"),
              "date", False, "Actual close date"),
    FieldSpec("created_date", ("Created Date", "Created On", "Creation Date", "Date Created",
                               "Opened Date"),
              "date", False, "Deal creation date"),
    FieldSpec("sector", ("Sector/service", "Sector", "Industry", "Vertical", "Segment",
                         "Sector / Service"),
              "text", False, "Sector / service line"),
    FieldSpec("product", ("Product deal", "Product", "Product Line", "Offering", "Solution"),
              "text", False, "Product mix"),
)

#: Canonical fields for the Work Orders board (aliases seeded from the WO tracker).
WORK_ORDER_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("work_order_id", ("Serial #", "Serial No", "Work Order ID", "WO ID", "WO Number",
                                "Order ID", "Project ID", "Serial"),
              "text", False, "Work order identifier"),
    FieldSpec("deal_name", ("Deal name masked", "Deal Name", "Deal", "Project Name", "Name"),
              "text", False, "Originating deal"),
    FieldSpec("client", ("Customer Name Code", "Customer", "Client", "Client Code", "Account",
                         "Company", "Customer Name"),
              "text", False, "Customer"),
    FieldSpec("sector", ("Sector", "Sector/service", "Industry", "Vertical", "Segment"),
              "text", False, "Sector"),
    FieldSpec("owner", ("BD/KAM Personnel code", "Owner code", "Owner", "Account Manager",
                        "KAM", "Project Manager", "Assigned To"),
              "text", False, "BD / KAM owner"),
    FieldSpec("execution_status", ("Execution Status", "Status", "Project Status",
                                   "Delivery Status", "Work Status"),
              "text", False, "Execution status"),
    FieldSpec("nature_of_work", ("Nature of Work", "Engagement Type", "Contract Type",
                                 "Work Nature", "Project Type"),
              "text", False, "One-time / contract"),
    FieldSpec("type_of_work", ("Type of Work", "Work Type", "Service Type", "Scope"),
              "text", False, "Service delivered"),
    FieldSpec("start_date", ("Probable Start Date", "Start Date", "Planned Start Date",
                             "Kickoff Date", "Actual Start Date"),
              "date", False, "Planned start"),
    FieldSpec("end_date", ("Probable End Date", "End Date", "Planned End Date", "Due Date",
                           "Target Completion Date", "Completion Date"),
              "date", False, "Planned completion"),
    FieldSpec("po_date", ("Date of PO/LOI", "PO Date", "Purchase Order Date", "LOI Date",
                          "Order Date"),
              "date", False, "PO / LOI date"),
    FieldSpec("delivery_date", ("Data Delivery Date", "Delivery Date", "Actual Delivery Date",
                                "Handover Date"),
              "date", False, "Actual delivery"),
    FieldSpec("order_value", ("Amount in Rupees (Excl of GST) (Masked)",
                              "Amount in Rupees (Excl of GST)", "Order Value", "PO Value",
                              "Project Value", "Contract Value", "Amount"),
              "numeric", False, "Order value (excl. GST)"),
    FieldSpec("billed_value", ("Billed Value in Rupees (Excl of GST.) (Masked)",
                               "Billed Value in Rupees (Excl of GST.)", "Billed Value",
                               "Invoiced Value", "Billed Amount"),
              "numeric", False, "Billed value (excl. GST)"),
    FieldSpec("collected_value", ("Collected Amount in Rupees (Incl of GST.) (Masked)",
                                  "Collected Amount", "Collections", "Amount Collected"),
              "numeric", False, "Collected amount (incl. GST)"),
    FieldSpec("receivable_value", ("Amount Receivable (Masked)", "Amount Receivable",
                                   "Receivables", "Outstanding Amount"),
              "numeric", False, "Receivable"),
    FieldSpec("to_be_billed_value", ("Amount to be billed in Rs. (Exl. of GST) (Masked)",
                                     "Amount to be billed in Rs. (Exl. of GST)",
                                     "Amount to be billed", "Unbilled Amount"),
              "numeric", False, "Yet to bill (excl. GST)"),
    FieldSpec("invoice_status", ("Invoice Status", "Billing Status", "Invoicing Status"),
              "text", False, "Invoice status"),
    FieldSpec("wo_status", ("WO Status (billed)", "WO Status", "Work Order Status"),
              "text", False, "Open / Closed work order"),
    FieldSpec("quantity_po", ("Quantities as per PO", "Quantity as per PO", "PO Quantity",
                              "Ordered Quantity"),
              "text", False, "Quantity ordered"),
    FieldSpec("quantity_balance", ("Balance in quantity", "Balance Quantity",
                                   "Remaining Quantity"),
              "text", False, "Quantity remaining"),
    FieldSpec("platform", ("Is any Skylark software platform part of the client deliverables in this deal?",
                           "Software Platform", "Platform"),
              "text", False, "Software platform attached"),
)


@dataclass
class ColumnMapping:
    """Resolved ``canonical field -> board column title`` mapping for one board."""

    board_id: str
    board_name: str
    mapping: dict[str, str] = field(default_factory=dict)
    #: How each field was matched, for the "Data sources" UI panel.
    match_method: dict[str, str] = field(default_factory=dict)
    unmapped_fields: list[str] = field(default_factory=list)
    unused_columns: list[str] = field(default_factory=list)
    column_types: dict[str, str] = field(default_factory=dict)

    def get(self, field_name: str) -> str | None:
        return self.mapping.get(field_name)

    def has(self, field_name: str) -> bool:
        return field_name in self.mapping

    def describe(self) -> list[dict[str, str]]:
        return [
            {
                "field": name,
                "monday_column": title,
                "matched_by": self.match_method.get(name, "?"),
                "type": self.column_types.get(title, "?"),
            }
            for name, title in sorted(self.mapping.items())
        ]


_TYPE_PREFERENCES = {
    "date": {"date", "timeline", "creation_log", "last_updated"},
    "numeric": {"numbers", "numeric", "formula", "mirror"},
    "text": {"text", "long_text", "status", "dropdown", "color", "people", "name"},
}


def _score(spec: FieldSpec, column: BoardColumn) -> tuple[float, str]:
    """Score how well ``column`` satisfies ``spec``. Returns ``(score, method)``."""
    title_key = slugify(column.title)
    if not title_key:
        return 0.0, "none"

    best = 0.0
    method = "none"
    for rank, alias in enumerate(spec.aliases):
        alias_key = slugify(alias)
        if not alias_key:
            continue
        # Earlier aliases are preferred; the penalty is small enough that a much
        # better textual match on a later alias still wins.
        rank_penalty = rank * 0.012

        if title_key == alias_key:
            candidate, how = 1.0 - rank_penalty, "exact"
        elif title_key.startswith(alias_key) or alias_key.startswith(title_key):
            candidate, how = 0.86 - rank_penalty, "prefix"
        elif alias_key in title_key or title_key in alias_key:
            candidate, how = 0.80 - rank_penalty, "substring"
        else:
            alias_tokens = set(alias_key.split("_"))
            title_tokens = set(title_key.split("_"))
            overlap = len(alias_tokens & title_tokens)
            if overlap and overlap >= max(1, len(alias_tokens) - 1):
                candidate = 0.72 + 0.02 * overlap - rank_penalty
                how = "tokens"
            else:
                ratio = difflib.SequenceMatcher(None, alias_key, title_key).ratio()
                candidate, how = (ratio - rank_penalty, "fuzzy") if ratio >= 0.84 else (0.0, "none")

        if candidate > best:
            best, method = candidate, how

    if best <= 0:
        return 0.0, "none"

    preferred_types = _TYPE_PREFERENCES.get(spec.kind, set())
    if column.type in preferred_types:
        best += 0.05
    elif spec.kind == "date" and column.type in {"text", "long_text"}:
        best -= 0.02
    elif spec.kind == "numeric" and column.type in {"text", "long_text"}:
        best -= 0.02
    return best, method


def resolve_columns(
    board_id: str,
    board_name: str,
    columns: list[BoardColumn],
    specs: tuple[FieldSpec, ...],
    *,
    overrides: dict[str, str] | None = None,
    threshold: float = 0.70,
) -> ColumnMapping:
    """Resolve canonical fields against a board's real columns.

    ``overrides`` lets an operator pin ``field -> column title`` explicitly (via
    ``COLUMN_OVERRIDES_*`` env JSON) when auto-detection picks the wrong column.
    """
    available = {slugify(c.title): c for c in columns if c.title}
    mapping: dict[str, str] = {}
    methods: dict[str, str] = {}
    claimed: set[str] = set()

    # 1. Explicit operator overrides win outright.
    for field_name, column_title in (overrides or {}).items():
        column = available.get(slugify(column_title))
        if column:
            mapping[field_name] = column.title
            methods[field_name] = "override"
            claimed.add(column.title)
        else:
            logger.warning(
                "Column override for '%s' points at '%s', which is not on board %s",
                field_name, column_title, board_id,
            )

    # 2. Score every (field, column) pair and assign greedily by descending score
    #    so a strong match cannot be stolen by a weaker one for another field.
    candidates: list[tuple[float, str, str, str]] = []
    for spec in specs:
        if spec.name in mapping:
            continue
        for column in columns:
            if not column.title:
                continue
            score, method = _score(spec, column)
            if score >= threshold:
                candidates.append((score, spec.name, column.title, method))

    for score, field_name, column_title, method in sorted(candidates, key=lambda x: -x[0]):
        if field_name in mapping or column_title in claimed:
            continue
        mapping[field_name] = column_title
        methods[field_name] = method
        claimed.add(column_title)

    unmapped = [s.name for s in specs if s.name not in mapping]
    if unmapped:
        logger.info("Board %s (%s): unmapped fields %s", board_id, board_name, unmapped)

    missing_required = [s.name for s in specs if s.required and s.name not in mapping]
    if missing_required:
        logger.warning(
            "Board %s is missing required field(s) %s; analytics will be limited",
            board_id, missing_required,
        )

    return ColumnMapping(
        board_id=str(board_id),
        board_name=board_name,
        mapping=mapping,
        match_method=methods,
        unmapped_fields=unmapped,
        unused_columns=[c.title for c in columns if c.title and c.title not in claimed],
        column_types={c.title: c.type for c in columns},
    )
