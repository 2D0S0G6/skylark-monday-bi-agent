"""Cross-board aggregation: pipeline vs operational workload."""
from __future__ import annotations

import pandas as pd
import pytest

from analytics.cross_board import (
    JOIN_POLICY,
    analyze_cross_board,
    capacity_signals,
    pipeline_vs_workload,
    shared_dimension_overlap,
)
from analytics.normalization import normalize_deals, normalize_work_orders
from monday.column_map import DEAL_FIELDS, WORK_ORDER_FIELDS
from tests.conftest import DEAL_COLUMNS, TODAY, WO_COLUMNS, make_mapping


def test_shared_sector_overlap_is_reported(deals_dataset, work_orders_dataset):
    overlap = shared_dimension_overlap(deals_dataset, work_orders_dataset)
    assert overlap["joinable"] is True
    assert set(overlap["shared_values"]) == {"Mining", "Renewables", "Powerline",
                                             "Railways", "Construction"}
    # A sector present only in the pipeline must be called out, not dropped silently.
    assert "Deep Sea Robotics" in overlap["deals_only"]


def test_pipeline_vs_workload_aggregates_both_boards(deals_dataset, work_orders_dataset):
    comparison = pipeline_vs_workload(deals_dataset, work_orders_dataset)
    assert comparison["available"] is True
    rows = {r["sector"]: r for r in comparison["rows"]}

    assert rows["Renewables"]["open_deal_count"] == 4
    assert rows["Renewables"]["active_work_order_count"] == 2
    assert rows["Mining"]["open_deal_count"] == 1
    assert rows["Mining"]["active_work_order_count"] == 3

    totals = comparison["totals"]
    assert totals["open_deal_count"] == 8
    assert totals["active_work_order_count"] == 7


def test_shares_are_computed_not_raw_values(deals_dataset, work_orders_dataset):
    comparison = pipeline_vs_workload(deals_dataset, work_orders_dataset)
    shares = [r["pipeline_share_pct"] for r in comparison["rows"]
              if r.get("pipeline_share_pct") is not None]
    assert sum(shares) == pytest.approx(100.0, abs=0.5)


def test_capacity_signals_identify_imbalance(deals_dataset, work_orders_dataset):
    comparison = pipeline_vs_workload(deals_dataset, work_orders_dataset)
    signals = capacity_signals(comparison)
    assert signals["available"] is True
    ahead = {s["sector"] for s in signals["pipeline_ahead_of_delivery"]}
    behind = {s["sector"] for s in signals["delivery_ahead_of_pipeline"]}
    # Renewables carries most of the pipeline but little of the current workload.
    assert "Renewables" in ahead
    assert "Mining" in behind
    assert ahead.isdisjoint(behind)


def test_signals_carry_an_explicit_correlation_caveat(deals_dataset, work_orders_dataset):
    signals = capacity_signals(pipeline_vs_workload(deals_dataset, work_orders_dataset))
    note = signals["interpretation_note"].lower()
    assert "not proven cause and effect" in note


def test_join_policy_rejects_unreliable_keys():
    assert JOIN_POLICY["join_key"] == "normalised sector"
    assert "client" in JOIN_POLICY["rejected_keys"]
    assert "deal_name" in JOIN_POLICY["rejected_keys"]


def test_sectors_present_on_only_one_board_are_reported(deals_dataset, work_orders_dataset):
    comparison = pipeline_vs_workload(deals_dataset, work_orders_dataset)
    codes = {i["code"] for i in comparison["data_quality"]["issues"]}
    assert "sectors_without_workload" in codes

    signals = capacity_signals(comparison)
    assert "Deep Sea Robotics" in signals["pipeline_but_no_active_work_orders"]


def test_cross_board_analysis_bundle(deals_dataset, work_orders_dataset):
    import json

    result = analyze_cross_board(deals_dataset, work_orders_dataset)
    json.dumps(result, default=str)
    assert result["available"] is True
    assert result["comparison"]["dimension"] == "sector"
    assert result["signals"]["available"] is True


def test_cross_board_unavailable_when_a_board_is_empty(work_orders_dataset):
    empty_deals = normalize_deals(pd.DataFrame(), make_mapping(DEAL_COLUMNS, DEAL_FIELDS))
    result = analyze_cross_board(empty_deals, work_orders_dataset)
    assert result["available"] is False
    assert "both boards" in result["reason"]


def test_cross_board_falls_back_to_counts_without_values():
    """With no value columns, the comparison must still work on counts."""
    deal_columns = [c for c in DEAL_COLUMNS if c["title"] != "Masked Deal value"]
    wo_columns = [c for c in WO_COLUMNS if "Amount in Rupees" not in c["title"]]

    deals = normalize_deals(
        pd.DataFrame([
            {"__item_id": "1", "__item_name": "D1", "Deal Status": "Open",
             "Sector/service": "Mining", "Deal Stage": "A. Lead Generated",
             "Tentative Close Date": "2026-03-01"},
            {"__item_id": "2", "__item_name": "D2", "Deal Status": "Open",
             "Sector/service": "Renewables", "Deal Stage": "F. Negotiations",
             "Tentative Close Date": "2026-03-01"},
        ]),
        make_mapping(deal_columns, DEAL_FIELDS), today=TODAY,
    )
    work_orders = normalize_work_orders(
        pd.DataFrame([
            {"__item_id": "1", "__item_name": "W1", "Serial #": "S1", "Sector": "Mining",
             "Execution Status": "Ongoing", "Probable End Date": "2026-12-31"},
        ]),
        make_mapping(wo_columns, WORK_ORDER_FIELDS), today=TODAY,
    )
    comparison = pipeline_vs_workload(deals, work_orders)
    assert comparison["available"] is True
    rows = {r["sector"]: r for r in comparison["rows"]}
    assert rows["Renewables"]["open_deal_count"] == 1
    assert rows["Renewables"]["active_work_order_count"] == 0
    assert rows["Mining"]["basis"] == "deal / work-order counts (values unavailable)"


def test_unknown_sectors_do_not_break_the_join(work_orders_dataset):
    deals = normalize_deals(
        pd.DataFrame([
            {"__item_id": "1", "__item_name": "D1", "Deal Status": "Open",
             "Sector/service": None, "Deal Stage": "A. Lead Generated",
             "Masked Deal value": "100000", "Tentative Close Date": "2026-03-01"},
        ]),
        make_mapping(DEAL_COLUMNS, DEAL_FIELDS), today=TODAY,
    )
    comparison = pipeline_vs_workload(deals, work_orders_dataset)
    assert comparison["available"] is True
    assert any(r["sector"] == "Unknown" for r in comparison["rows"])


def test_group_with_no_recorded_values_is_not_reported_as_zero(work_orders_dataset):
    """A sector whose deals all lack a value must read as 'unknown', never ₹0."""
    deals = normalize_deals(
        pd.DataFrame([
            {"__item_id": "1", "__item_name": "D1", "Deal Status": "Open",
             "Sector/service": "Railways", "Deal Stage": "A. Lead Generated",
             "Masked Deal value": "N/A", "Tentative Close Date": "2026-03-01"},
            {"__item_id": "2", "__item_name": "D2", "Deal Status": "Open",
             "Sector/service": "Mining", "Deal Stage": "F. Negotiations",
             "Masked Deal value": "1000000", "Tentative Close Date": "2026-03-01"},
        ]),
        make_mapping(DEAL_COLUMNS, DEAL_FIELDS), today=TODAY,
    )
    comparison = pipeline_vs_workload(deals, work_orders_dataset)
    rows = {r["sector"]: r for r in comparison["rows"]}
    assert rows["Railways"]["open_pipeline_value"] is None
    assert "no deal value recorded" in rows["Railways"]["value_note"]
    assert rows["Mining"]["open_pipeline_value"]["amount"] == pytest.approx(1_000_000)
