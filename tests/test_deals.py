"""Deterministic deal analytics."""
from __future__ import annotations

import pandas as pd
import pytest

from analytics.deals import (
    analyze_deals,
    deal_scope,
    deals_at_risk,
    expected_close_analysis,
    pipeline_by_dimension,
    pipeline_summary,
    top_opportunities,
    win_rate,
)
from analytics.normalization import normalize_deals
from monday.column_map import DEAL_FIELDS
from tests.conftest import DEAL_COLUMNS, TODAY, make_mapping
from utils.dates import resolve_date_range


def test_pipeline_summary_totals_only_parseable_values(deals_dataset):
    """Open pipeline = 25.0 + 25.0 + 15.0 + 8.0 + 4.5 + 25.0 (duplicate) Cr."""
    summary = pipeline_summary(deals_dataset)
    assert summary["open_deal_count"] == 8
    # Two open deals have no parseable value and must be excluded, not zeroed.
    assert summary["open_deals_missing_value"] == 2
    assert summary["open_pipeline_value"]["amount"] == pytest.approx(102_500_000)
    assert summary["open_pipeline_value"]["display"] == "₹10.25 Cr"


def test_won_and_lost_values_are_separated(deals_dataset):
    summary = pipeline_summary(deals_dataset)
    assert summary["won_deal_count"] == 2
    assert summary["won_value"]["amount"] == pytest.approx(17_000_000)
    assert summary["lost_deal_count"] == 2
    assert summary["lost_value"]["amount"] == pytest.approx(3_000_000)


def test_late_stage_pipeline_and_share(deals_dataset):
    summary = pipeline_summary(deals_dataset)
    # Deals 1, 2, 11 and the duplicate of 1 are late stage and open.
    assert summary["late_stage_open_deal_count"] == 4
    assert summary["late_stage_open_value"]["amount"] == pytest.approx(79_500_000)
    assert summary["late_stage_share_of_open_pct"] == pytest.approx(77.6, abs=0.1)


def test_weighted_pipeline_uses_documented_probability_weights(deals_dataset):
    summary = pipeline_summary(deals_dataset)
    weighted = summary["weighted_open_pipeline_value"]["amount"]
    # High=0.75 (25.0 + 25.0 + 4.5 Cr), Medium=0.45 (25.0 Cr), Low=0.20 (15.0 + 8.0 Cr)
    expected = 0.75 * (250_000_000 + 45_000_000) / 10 * 10  # kept explicit below
    expected = 0.75 * (25_000_000 + 25_000_000 + 4_500_000) + 0.45 * 25_000_000 + 0.20 * (15_000_000 + 8_000_000)
    assert weighted == pytest.approx(expected)
    basis = summary["weighted_pipeline_basis"]
    assert basis["weights"] == {"High": 0.75, "Medium": 0.45, "Low": 0.20}
    assert basis["deals_without_probability"] >= 0


def test_pipeline_by_sector_shares_sum_to_100(deals_dataset):
    breakdown = pipeline_by_dimension(deals_dataset, dimension="sector", status="Open")
    shares = [r["value_share_pct"] for r in breakdown["rows"] if r.get("value_share_pct")]
    assert sum(shares) == pytest.approx(100.0, abs=0.2)
    assert breakdown["rows"][0]["sector"] == "Renewables"


def test_pipeline_by_stage_group(deals_dataset):
    breakdown = pipeline_by_dimension(deals_dataset, dimension="stage_group", status="Open")
    groups = {r["stage_group"] for r in breakdown["rows"]}
    assert "late" in groups and "early" in groups


def test_pipeline_by_owner(deals_dataset):
    breakdown = pipeline_by_dimension(deals_dataset, dimension="owner", status="Open")
    assert breakdown["rows"]
    assert all(r["owner"].startswith("OWNER_") for r in breakdown["rows"])


def test_top_opportunities_ranked_with_concentration(deals_dataset):
    top = top_opportunities(deals_dataset, limit=3)
    values = [r["value"]["amount"] for r in top["rows"]]
    assert values == sorted(values, reverse=True)
    assert top["top3_concentration_pct"] > 50


def test_sector_filter_scopes_the_analysis(deals_dataset):
    scope = deal_scope(deals_dataset, sector="Mining", status_filter="open")
    assert set(scope["frame"]["sector"]) == {"Mining"}
    assert "sector = Mining" in scope["filters_applied"][0]


def test_energy_umbrella_query_reaches_renewables(deals_dataset):
    scope = deal_scope(deals_dataset, sector="energy", status_filter="open")
    assert set(scope["frame"]["sector"]) <= {"Renewables", "Powerline"}
    assert len(scope["frame"]) > 0


def test_unknown_sector_filter_returns_empty_scope_not_an_error(deals_dataset):
    scope = deal_scope(deals_dataset, sector="Hospitality")
    assert len(scope["frame"]) == 0
    summary = pipeline_summary(deals_dataset, scope["frame"])
    assert summary["deal_count"] == 0
    assert "note" in summary


def test_quarter_filter_excludes_undated_deals_and_reports_it(deals_dataset):
    date_range = resolve_date_range("current_quarter", today=TODAY, fy_start_month=4)
    scope = deal_scope(deals_dataset, date_range=date_range)
    assert scope["excluded_missing_date"] > 0
    codes = {i.code for i in scope["report"].issues}
    assert "date_filter_excluded" in codes
    # Only deals with an expected close inside Jan-Mar 2026 survive.
    assert scope["frame"]["expected_close_date"].between(
        date_range.start, date_range.end
    ).all()


def test_win_rate_needs_a_meaningful_sample(deals_dataset):
    result = win_rate(deals_dataset)
    assert result["available"] is False
    assert "closed deals" in result["reason"]


def test_win_rate_computed_when_sample_is_large_enough():
    rows = []
    for i in range(10):
        rows.append({
            "__item_id": str(i), "__item_name": f"D{i}",
            "Deal Status": "Won" if i < 6 else "Dead",
            "Masked Deal value": "1000000",
            "Sector/service": "Mining",
            "Deal Stage": "G. Project Won" if i < 6 else "L. Project Lost",
            "Tentative Close Date": "2025-06-01",
            "Closure Probability": "High",
            "Owner code": "OWNER_001", "Client Code": f"C{i}",
            "Close Date (A)": "", "Product deal": "", "Created Date": "2025-01-01",
        })
    dataset = normalize_deals(pd.DataFrame(rows), make_mapping(DEAL_COLUMNS, DEAL_FIELDS))
    result = win_rate(dataset)
    assert result["available"] is True
    assert result["win_rate_by_count_pct"] == pytest.approx(60.0)
    assert result["win_rate_by_value_pct"] == pytest.approx(60.0)


def test_deals_at_risk_uses_explicit_signals(deals_dataset):
    risk = deals_at_risk(deals_dataset, today=TODAY)
    assert risk["at_risk_count"] > 0
    assert risk["criteria"]
    for row in risk["rows"]:
        assert row["risk_signals"], "every listed deal must carry a stated signal"
    overdue = [r for r in risk["rows"] if "expected close date has passed" in r["risk_signals"]]
    assert overdue, "the deal with a past expected close must be flagged"


def test_expected_close_analysis_buckets_by_quarter(deals_dataset):
    result = expected_close_analysis(deals_dataset, today=TODAY, fy_start_month=4, quarters=3)
    assert result["available"] is True
    assert len(result["quarters"]) == 3
    assert result["quarters"][0]["quarter"].startswith("Q4 FY2025-26")
    assert result["overdue_open_deal_count"] >= 1


def test_metrics_are_omitted_when_the_value_column_is_absent():
    """No value column -> counts still work, value metrics are declared unavailable."""
    columns = [c for c in DEAL_COLUMNS if c["title"] != "Masked Deal value"]
    mapping = make_mapping(columns, DEAL_FIELDS)
    rows = [{
        "__item_id": "1", "__item_name": "D1", "Deal Status": "Open",
        "Sector/service": "Mining", "Deal Stage": "A. Lead Generated",
        "Tentative Close Date": "2026-03-01", "Owner code": "OWNER_001",
        "Client Code": "C1", "Closure Probability": "High",
        "Close Date (A)": "", "Product deal": "", "Created Date": "2025-01-01",
    }]
    dataset = normalize_deals(pd.DataFrame(rows), mapping)
    summary = pipeline_summary(dataset)
    assert summary["open_deal_count"] == 1
    assert "pipeline_value" in summary["unavailable_metrics"]
    assert "total_value" not in summary


def test_analyze_deals_bundle_is_json_serialisable(deals_dataset):
    import json

    facts = analyze_deals(deals_dataset, today=TODAY)
    json.dumps(facts, default=str)  # must not raise
    assert facts["board"] == "deals"
    assert facts["summary"]["open_deal_count"] == 8
    assert facts["data_quality"]["issues"]


def test_analyze_deals_on_empty_dataset():
    dataset = normalize_deals(pd.DataFrame(), make_mapping(DEAL_COLUMNS, DEAL_FIELDS))
    facts = analyze_deals(dataset, today=TODAY)
    assert facts["summary"]["deal_count"] == 0
    assert facts["scope"]["deals_matching_all_filters"] == 0


def test_empty_period_scope_explains_where_the_pipeline_actually_is(deals_dataset):
    """A zero result must be distinguishable from a broken connection."""
    from utils.dates import resolve_date_range

    # A quarter far beyond every expected close date on the board.
    far_future = resolve_date_range("current_quarter", today=pd.Timestamp("2030-05-01"),
                                    fy_start_month=4)
    facts = analyze_deals(deals_dataset, date_range=far_future, status_filter="open",
                          today=TODAY)
    assert facts["scope"]["deals_matching_all_filters"] == 0
    context = facts["empty_scope_context"]
    assert context["open_deals_on_board_all_periods"] == 8
    assert context["open_expected_close_dates_actually_span"]["latest"] < "2030-01-01"
    assert context["open_pipeline_by_expected_close_month"]


def test_scope_labels_are_unambiguous(deals_dataset):
    """Field names must not invite the narrator to conflate scoped and total counts."""
    facts = analyze_deals(deals_dataset, status_filter="open", today=TODAY)
    scope = facts["scope"]
    assert scope["deals_matching_all_filters"] == 8
    assert scope["total_deals_on_board_all_statuses"] == 13
    assert scope["open_deals_on_board_all_periods"] == 8
