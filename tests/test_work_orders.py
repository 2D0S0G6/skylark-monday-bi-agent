"""Deterministic work-order analytics."""
from __future__ import annotations

import pandas as pd
import pytest

from analytics.normalization import normalize_work_orders
from analytics.work_orders import (
    analyze_work_orders,
    billing_summary,
    delayed_work_orders,
    operations_summary,
    upcoming_completions,
    work_order_scope,
    work_orders_by_dimension,
)
from monday.column_map import WORK_ORDER_FIELDS
from tests.conftest import TODAY, WO_COLUMNS, make_mapping
from utils.dates import resolve_date_range


def test_operations_summary_counts(work_orders_dataset):
    summary = operations_summary(work_orders_dataset)
    assert summary["work_order_count"] == 10
    assert summary["completed_work_orders"] == 2
    assert summary["active_work_orders"] == 7
    assert summary["blocked_work_orders"] == 1
    assert summary["not_started_work_orders"] == 1
    assert summary["unknown_status_work_orders"] == 1


def test_active_and_completed_never_exceed_the_total(work_orders_dataset):
    summary = operations_summary(work_orders_dataset)
    assert summary["active_work_orders"] + summary["completed_work_orders"] <= summary["work_order_count"]


def test_delayed_work_orders_are_identified_with_days_overdue(work_orders_dataset):
    delayed = delayed_work_orders(work_orders_dataset)
    assert delayed["delayed_count"] == 2
    ids = {r["work_order"] for r in delayed["rows"]}
    assert ids == {"SDPL-002", "SDPL-004"}
    assert all(r["days_overdue"] > 0 for r in delayed["rows"])
    # Ranked worst-first.
    overdue = [r["days_overdue"] for r in delayed["rows"]]
    assert overdue == sorted(overdue, reverse=True)


def test_delay_share_is_measured_against_assessable_orders_only(work_orders_dataset):
    delayed = delayed_work_orders(work_orders_dataset)
    # Six active work orders have a planned end date; one active order has none.
    assert delayed["assessable_count"] == 6
    assert delayed["delayed_share_pct"] == pytest.approx(33.3, abs=0.1)


def test_delayed_value_excludes_unparseable_amounts(work_orders_dataset):
    delayed = delayed_work_orders(work_orders_dataset)
    assert delayed["delayed_value"]["amount"] == pytest.approx(2_984_097 + 1_200_000)


def test_work_orders_by_sector(work_orders_dataset):
    breakdown = work_orders_by_dimension(work_orders_dataset, dimension="sector")
    sectors = {r["sector"]: r for r in breakdown["rows"]}
    assert sectors["Mining"]["work_order_count"] == 4
    assert sectors["Renewables"]["work_order_count"] == 2
    assert sum(r["work_order_count"] for r in breakdown["rows"]) == 10


def test_work_orders_by_status(work_orders_dataset):
    breakdown = work_orders_by_dimension(work_orders_dataset, dimension="execution_status")
    statuses = {r["execution_status"] for r in breakdown["rows"]}
    assert {"Completed", "In Progress", "Not Started", "Blocked", "Unknown"} <= statuses


def test_operational_value_totals(work_orders_dataset):
    summary = operations_summary(work_orders_dataset)
    # One order value is "N/A" and must be excluded, not counted as zero.
    assert summary["work_orders_missing_value"] == 1
    assert summary["total_order_value"]["amount"] == pytest.approx(
        184980 + 2984097 + 154150 + 1_200_000 + 3995568 + 500000 + 750000 + 1200000 + 1200000
    )


def test_sector_scope_filters_work_orders(work_orders_dataset):
    scope = work_order_scope(work_orders_dataset, sector="Mining")
    assert set(scope["frame"]["sector"]) == {"Mining"}
    assert len(scope["frame"]) == 4


def test_status_scope_filters_to_active(work_orders_dataset):
    scope = work_order_scope(work_orders_dataset, status_filter="active")
    assert len(scope["frame"]) == 7
    assert bool(scope["frame"]["is_active"].all())


def test_date_scope_reports_undated_exclusions(work_orders_dataset):
    date_range = resolve_date_range("current_fy", today=TODAY, fy_start_month=4)
    scope = work_order_scope(work_orders_dataset, date_range=date_range, date_field="end_date")
    assert scope["excluded_missing_date"] == 1
    assert {i.code for i in scope["report"].issues} >= {"date_filter_excluded"}


def test_upcoming_completions_window(work_orders_dataset):
    upcoming = upcoming_completions(work_orders_dataset, days=90)
    assert upcoming["window_days"] == 90
    # SDPL-003 ends 2026-04-15, which is within 90 days of 2026-02-15.
    assert any(r["work_order"] == "SDPL-003" for r in upcoming["rows"])


def test_billing_summary_reports_ratios_with_a_caveat(work_orders_dataset):
    billing = billing_summary(work_orders_dataset)
    assert billing["available"] is True
    assert billing["total_billed_value"]["amount"] == pytest.approx(184980 + 23959 + 3662604 + 750000)
    assert "caveat" in billing


def test_billing_unavailable_without_billing_columns():
    columns = [c for c in WO_COLUMNS if "Billed" not in c["title"] and "Collected" not in c["title"]]
    mapping = make_mapping(columns, WORK_ORDER_FIELDS)
    rows = [{
        "__item_id": "1", "__item_name": "WO1", "Serial #": "S1", "Sector": "Mining",
        "Execution Status": "Ongoing", "Probable End Date": "2026-01-01",
        "Amount in Rupees (Excl of GST) (Masked)": "100000",
    }]
    dataset = normalize_work_orders(pd.DataFrame(rows), mapping, today=TODAY)
    billing = billing_summary(dataset)
    assert billing["available"] is True
    assert "total_billed_value" not in billing


def test_value_metrics_unavailable_without_an_amount_column():
    columns = [c for c in WO_COLUMNS if "Amount in Rupees" not in c["title"]]
    mapping = make_mapping(columns, WORK_ORDER_FIELDS)
    rows = [{
        "__item_id": "1", "__item_name": "WO1", "Serial #": "S1", "Sector": "Mining",
        "Execution Status": "Ongoing", "Probable End Date": "2026-01-01",
    }]
    dataset = normalize_work_orders(pd.DataFrame(rows), mapping, today=TODAY)
    summary = operations_summary(dataset)
    assert summary["work_order_count"] == 1
    assert "order_value" in summary["unavailable_metrics"]


def test_delay_grace_period_is_respected():
    rows = [{
        "__item_id": "1", "__item_name": "WO1", "Serial #": "S1", "Sector": "Mining",
        "Execution Status": "Ongoing", "Probable End Date": "2026-02-10",
        "Amount in Rupees (Excl of GST) (Masked)": "100000",
    }]
    mapping = make_mapping(WO_COLUMNS, WORK_ORDER_FIELDS)
    strict = normalize_work_orders(pd.DataFrame(rows), mapping, today=TODAY, delay_grace_days=0)
    lenient = normalize_work_orders(pd.DataFrame(rows), mapping, today=TODAY, delay_grace_days=30)
    assert bool(strict.frame["is_delayed"].iloc[0]) is True
    assert bool(lenient.frame["is_delayed"].iloc[0]) is False


def test_analyze_work_orders_bundle_is_json_serialisable(work_orders_dataset):
    import json

    facts = analyze_work_orders(work_orders_dataset, today=TODAY)
    json.dumps(facts, default=str)
    assert facts["board"] == "work_orders"
    assert facts["summary"]["work_order_count"] == 10


def test_analyze_work_orders_on_empty_dataset():
    dataset = normalize_work_orders(pd.DataFrame(), make_mapping(WO_COLUMNS, WORK_ORDER_FIELDS))
    facts = analyze_work_orders(dataset, today=TODAY)
    assert facts["summary"]["work_order_count"] == 0


def test_no_delayed_orders_returns_a_clear_note():
    rows = [{
        "__item_id": "1", "__item_name": "WO1", "Serial #": "S1", "Sector": "Mining",
        "Execution Status": "Ongoing", "Probable End Date": "2026-12-31",
        "Amount in Rupees (Excl of GST) (Masked)": "100000",
    }]
    dataset = normalize_work_orders(
        pd.DataFrame(rows), make_mapping(WO_COLUMNS, WORK_ORDER_FIELDS), today=TODAY
    )
    delayed = delayed_work_orders(dataset)
    assert delayed["delayed_count"] == 0
    assert "No open work order" in delayed["note"]
