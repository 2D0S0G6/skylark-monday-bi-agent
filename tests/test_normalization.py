"""Normalisation: text canonicalisation, dates, currency, missing values, malformed rows."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from analytics.normalization import (
    normalize_deals,
    normalize_work_orders,
    sector_matches,
)
from analytics.quality import Severity
from monday.column_map import DEAL_FIELDS, WORK_ORDER_FIELDS, resolve_columns
from monday.schemas import BoardColumn
from tests.conftest import DEAL_COLUMNS, WO_COLUMNS, make_mapping
from utils.dates import parse_date, resolve_date_range
from utils.numbers import format_inr, is_missing, parse_amount, parse_percentage, parse_quantity
from utils.text import (
    canonical_deal_stage,
    canonical_deal_status,
    canonical_execution_status,
    canonical_probability,
    canonical_sector,
)

# --------------------------------------------------------------------------
# 1. sector normalisation
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Energy", "Energy"),
        ("energy", "Energy"),
        ("ENERGY", "Energy"),
        ("Energy Sector", "Energy"),
        ("  energy sector  ", "Energy"),
        ("Renewables", "Renewables"),
        ("renewable", "Renewables"),
        ("SOLAR", "Renewables"),
        ("Mining", "Mining"),
        ("mining sector", "Mining"),
        ("powerline", "Powerline"),
        ("Power Line", "Powerline"),
        ("railways", "Railways"),
        ("DSP", "DSP"),
        ("Security and Surveillance", "Security and Surveillance"),
    ],
)
def test_sector_normalisation_variants(raw, expected):
    assert canonical_sector(raw).value == expected


def test_sector_typo_is_fuzzy_matched():
    result = canonical_sector("Enrgy")
    assert result.value == "Energy"
    assert result.method == "fuzzy"


def test_unknown_sector_is_preserved_not_bucketed():
    """An unrecognised sector must stay visible rather than becoming 'Others'."""
    result = canonical_sector("Deep Sea Robotics")
    assert result.value == "Deep Sea Robotics"
    assert result.method == "passthrough"


@pytest.mark.parametrize("raw", [None, "", "N/A", "  ", "-", float("nan"), "Unknown"])
def test_missing_sector_becomes_unknown(raw):
    result = canonical_sector(raw)
    assert result.value == "Unknown"
    assert result.method == "missing"


def test_sector_umbrella_matching():
    """'energy' should reach the renewables/powerline sectors actually on the board."""
    available = ["Renewables", "Mining", "Powerline", "Railways"]
    assert set(sector_matches("energy", available)) == {"Renewables", "Powerline"}
    assert sector_matches("mining", available) == ["Mining"]
    assert sector_matches("aviation", available) == []


# --------------------------------------------------------------------------
# 2. status normalisation
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Closed Won", "Won"), ("Won", "Won"), ("won", "Won"), ("CLOSED WON", "Won"),
        ("Open", "Open"), ("open", "Open"), ("Active", "Open"),
        ("Dead", "Lost"), ("Closed Lost", "Lost"), ("cancelled", "Lost"),
        ("On Hold", "On Hold"), ("on hold", "On Hold"), ("PAUSED", "On Hold"),
    ],
)
def test_deal_status_normalisation(raw, expected):
    assert canonical_deal_status(raw).value == expected


@pytest.mark.parametrize("raw", [None, "", "N/A", "-"])
def test_missing_status_is_unknown(raw):
    assert canonical_deal_status(raw).value == "Unknown"


@pytest.mark.parametrize(
    "raw,label,group",
    [
        ("A. Lead Generated", "Lead Generated", "early"),
        ("E. Proposal/Commercials Sent", "Proposal/Commercials Sent", "late"),
        ("F. Negotiations", "Negotiations", "late"),
        ("G. Project Won", "Project Won", "won"),
        ("L. Project Lost", "Project Lost", "lost"),
        ("M. Projects On Hold", "Projects On Hold", "on_hold"),
        ("Project Completed", "Project Completed", "won"),
    ],
)
def test_deal_stage_normalisation(raw, label, group):
    stage = canonical_deal_stage(raw)
    assert stage.label == label
    assert stage.group == group


def test_stage_ordering_is_monotonic_for_lettered_stages():
    early = canonical_deal_stage("A. Lead Generated")
    late = canonical_deal_stage("F. Negotiations")
    assert early.order < late.order


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Completed", "Completed"), ("Ongoing", "In Progress"),
        ("Executed until current month", "In Progress"),
        ("Partial Completed", "In Progress"), ("Not Started", "Not Started"),
        ("Pause / struck", "Blocked"), ("Details pending from Client", "Blocked"),
    ],
)
def test_execution_status_normalisation(raw, expected):
    assert canonical_execution_status(raw).value == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("High", "High"), ("high", "High"), ("MEDIUM", "Medium"), ("low", "Low"),
     ("80%", "High"), ("0.4", "Medium"), ("10%", "Low")],
)
def test_probability_normalisation(raw, expected):
    assert canonical_probability(raw).value == expected


# --------------------------------------------------------------------------
# 3. date parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("12/08/2026", "2026-08-12"),
        ("2026-08-12", "2026-08-12"),
        ("12-Aug-26", "2026-08-12"),
        ("Aug 12, 2026", "2026-08-12"),
        ("August 12, 2026", "2026-08-12"),
        ("12 Aug 2026", "2026-08-12"),
        ("2026-08-12 00:00:00", "2026-08-12"),
        ("31/12/2025", "2025-12-31"),
        ("2026/08/12", "2026-08-12"),
    ],
)
def test_date_formats(raw, expected):
    parsed = parse_date(raw)
    assert parsed.ok
    assert parsed.value.date().isoformat() == expected


def test_ambiguous_date_resolves_by_day_first_hint():
    """05/03/2026 is 5 March under the Indian day-first convention."""
    assert parse_date("05/03/2026").value.date().isoformat() == "2026-03-05"
    # Unambiguous: 25 cannot be a month.
    assert parse_date("25/03/2026").value.date().isoformat() == "2026-03-25"


@pytest.mark.parametrize("raw", ["not a date", "TBD", "12/45/2026", "???"])
def test_invalid_dates_do_not_raise(raw):
    parsed = parse_date(raw)
    assert parsed.value is None
    assert parsed.reason in {"invalid", "missing"}


@pytest.mark.parametrize("raw", [None, "", "N/A", "-", float("nan")])
def test_missing_dates_reported_as_missing(raw):
    assert parse_date(raw).reason == "missing"


def test_quarter_resolution_uses_indian_fiscal_year():
    today = pd.Timestamp("2026-02-15")
    current = resolve_date_range("current_quarter", today=today, fy_start_month=4)
    assert current.start.date().isoformat() == "2026-01-01"
    assert current.end.date().isoformat() == "2026-03-31"
    assert "Q4 FY2025-26" in current.label


def test_calendar_quarter_mode():
    today = pd.Timestamp("2026-02-15")
    current = resolve_date_range("current_quarter", today=today, fy_start_month=1)
    assert current.start.date().isoformat() == "2026-01-01"
    assert current.label.startswith("Q1 2026")


def test_all_time_returns_no_range():
    assert resolve_date_range("all_time") is None
    assert resolve_date_range(None) is None


# --------------------------------------------------------------------------
# 4. currency / numeric parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("₹25 Lakhs", 2_500_000), ("25 L", 2_500_000), ("₹2.5 Cr", 25_000_000),
        ("2.5 Cr", 25_000_000), ("2500000", 2_500_000), ("2,500,000", 2_500_000),
        ("1,50,00,000", 15_000_000), ("₹ 1,20,000", 120_000), ("50 lakh", 5_000_000),
        ("3 crore", 30_000_000), ("1.2 M", 1_200_000), ("15K", 15_000),
        (489360, 489_360), (2984097.36, 2984097.36),
    ],
)
def test_currency_parsing(raw, expected):
    parsed = parse_amount(raw)
    assert parsed.ok
    assert math.isclose(parsed.value, expected, rel_tol=1e-9)


def test_negative_and_parenthesised_values():
    assert parse_amount("(1,200)").value == -1200
    assert parse_amount("-82907.29").value == pytest.approx(-82907.29)


def test_foreign_currency_is_parsed_but_flagged_not_converted():
    """We never invent an FX rate: the magnitude is kept, the currency is recorded."""
    parsed = parse_amount("$100000")
    assert parsed.ok
    assert parsed.value == 100_000
    assert parsed.currency == "USD"
    assert parsed.is_base_currency is False


@pytest.mark.parametrize("raw", ["abc", "twelve lakhs", "12 bananas", "#REF!"])
def test_unparseable_values_are_not_zero(raw):
    parsed = parse_amount(raw)
    assert parsed.value is None
    assert parsed.reason in {"unparseable", "unrecognised_unit", "missing"}


@pytest.mark.parametrize("raw", [None, "", "N/A", "NA", "-", "Unknown", "Not Available",
                                 float("nan"), "TBD"])
def test_missing_values_are_missing_not_zero(raw):
    parsed = parse_amount(raw)
    assert parsed.value is None
    assert parsed.reason == "missing"
    assert is_missing(raw)


def test_quantity_parsing_keeps_units():
    parsed = parse_quantity("5360 HA")
    assert parsed.value == 5360
    assert parsed.currency == "ha"
    assert parse_quantity("59.33").value == pytest.approx(59.33)


def test_percentage_parsing():
    assert parse_percentage("75%").value == pytest.approx(0.75)
    assert parse_percentage(0.4).value == pytest.approx(0.4)
    assert parse_percentage("not a number").value is None


def test_inr_formatting_matches_indian_conventions():
    assert format_inr(124_000_000) == "₹12.40 Cr"
    assert format_inr(250_000) == "₹2.50 L"
    assert format_inr(None) == "n/a"
    assert format_inr(-25_000_000).startswith("-₹2.50 Cr")


# --------------------------------------------------------------------------
# 5. dataset-level normalisation
# --------------------------------------------------------------------------

def test_normalize_deals_from_fake_board(deals_dataset):
    frame = deals_dataset.frame
    # 14 raw rows, one of which echoes the header and is removed.
    assert len(frame) == 13
    assert set(frame["status"].unique()) <= {"Open", "Won", "Lost", "On Hold", "Unknown"}
    assert frame["sector"].tolist().count("Renewables") == 4


def test_missing_deal_value_is_nan_not_zero(deals_dataset):
    frame = deals_dataset.frame
    missing = frame[frame["value"].isna()]
    # "N/A", "-", "Unknown" (missing) plus "twelve lakhs" (unparseable)
    assert len(missing) == 4
    assert (frame["value"].fillna(-1) != 0).all()


def test_unparseable_value_recorded_as_quality_issue(deals_dataset):
    codes = {i.code for i in deals_dataset.quality.issues}
    assert "unparseable_value" in codes
    assert "missing_value" in codes


def test_header_echo_row_is_dropped_and_reported(deals_dataset):
    issue = next(i for i in deals_dataset.quality.issues if i.code == "header_rows_removed")
    assert issue.count == 1
    assert issue.severity == Severity.EXCLUDED


def test_duplicate_deals_are_flagged_but_retained(deals_dataset):
    frame = deals_dataset.frame
    assert int(frame["is_duplicate"].sum()) == 1
    # Retained: dropping it would understate the pipeline.
    assert len(frame) == 13


def test_status_is_inferred_from_stage_when_missing(deals_dataset):
    frame = deals_dataset.frame
    inferred = frame[frame["status_inferred"]]
    assert len(inferred) == 1
    assert inferred.iloc[0]["status"] == "Open"


def test_original_values_are_preserved(deals_dataset):
    frame = deals_dataset.frame
    raw_sectors = set(frame["sector_raw"].dropna())
    assert "RENEWABLES" in raw_sectors or "renewables" in raw_sectors
    assert "Renewables" in set(frame["sector"])


def test_normalize_work_orders_from_fake_board(work_orders_dataset):
    frame = work_orders_dataset.frame
    assert len(frame) == 10
    assert int(frame["is_delayed"].sum()) == 2
    assert int(frame["is_duplicate"].sum()) == 1


def test_unknown_execution_status_is_not_treated_as_active(work_orders_dataset):
    frame = work_orders_dataset.frame
    unknown = frame[frame["execution_status"] == "Unknown"]
    assert len(unknown) == 1
    assert bool(unknown.iloc[0]["is_active"]) is False
    assert bool(unknown.iloc[0]["is_delayed"]) is False


def test_work_order_without_end_date_cannot_be_delayed(work_orders_dataset):
    frame = work_orders_dataset.frame
    no_end = frame[frame["end_date"].isna()]
    assert len(no_end) == 1
    assert bool(no_end.iloc[0]["is_delayed"]) is False
    codes = {i.code for i in work_orders_dataset.quality.issues}
    assert "open_wo_without_end_date" in codes


# --------------------------------------------------------------------------
# 9/10. malformed data and empty datasets
# --------------------------------------------------------------------------

def test_empty_board_produces_empty_dataset_not_a_crash():
    mapping = make_mapping(DEAL_COLUMNS, DEAL_FIELDS)
    dataset = normalize_deals(pd.DataFrame(), mapping)
    assert dataset.empty
    assert len(dataset.frame) == 0
    codes = {i.code for i in dataset.quality.issues}
    assert "empty_board" in codes


def test_empty_work_order_board():
    mapping = make_mapping(WO_COLUMNS, WORK_ORDER_FIELDS)
    dataset = normalize_work_orders(pd.DataFrame(), mapping)
    assert dataset.empty
    assert {i.code for i in dataset.quality.issues} >= {"empty_board"}


def test_board_missing_every_expected_column_degrades_gracefully():
    """A board whose columns we cannot recognise must not raise."""
    columns = [BoardColumn(id="c1", title="Totally Unrelated", type="text")]
    mapping = resolve_columns("9", "Odd board", columns, DEAL_FIELDS)
    frame = pd.DataFrame([{"__item_id": "1", "__item_name": "A", "Totally Unrelated": "x"}])
    dataset = normalize_deals(frame, mapping)
    assert len(dataset.frame) == 1
    assert dataset.available_fields.get("value") is False
    assert dataset.frame["value"].isna().all()
    assert dataset.frame["sector"].iloc[0] == "Unknown"


def test_all_null_rows_do_not_crash_normalisation():
    mapping = make_mapping(DEAL_COLUMNS, DEAL_FIELDS)
    frame = pd.DataFrame([
        {"__item_id": "1", "__item_name": None, **{c["title"]: None for c in DEAL_COLUMNS}},
        {"__item_id": "2", "__item_name": None, **{c["title"]: "N/A" for c in DEAL_COLUMNS}},
    ])
    dataset = normalize_deals(frame, mapping)
    assert len(dataset.frame) == 2
    assert dataset.frame["value"].isna().all()
    assert set(dataset.frame["status"]) == {"Unknown"}


def test_wildly_malformed_cell_types_are_tolerated():
    mapping = make_mapping(DEAL_COLUMNS, DEAL_FIELDS)
    frame = pd.DataFrame([{
        "__item_id": "1", "__item_name": "Weird",
        "Masked Deal value": {"nested": "dict"},
        "Tentative Close Date": ["a", "list"],
        "Deal Status": 12345,
        "Sector/service": True,
        "Deal Stage": None,
    }])
    dataset = normalize_deals(frame, mapping)
    assert len(dataset.frame) == 1
    assert pd.isna(dataset.frame["value"].iloc[0])
