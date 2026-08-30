"""Monday.com client: pagination, column discovery, value flattening, error handling."""
from __future__ import annotations

import httpx
import pytest

from monday.client import (
    MondayAuthError,
    MondayBoardNotFoundError,
    MondayClient,
    MondayError,
    MondayRateLimitError,
    MondayUnavailableError,
    snapshot_to_dataframe,
)
from monday.column_map import DEAL_FIELDS, WORK_ORDER_FIELDS, resolve_columns
from monday.schemas import BoardColumn
from tests.conftest import DEAL_COLUMNS, WO_COLUMNS, FakeMondayAPI, make_settings


def test_pagination_retrieves_every_item(settings, fake_api):
    """Page size is 2 in the fixtures, so 14 deals require several round-trips."""
    client = MondayClient(settings, transport=fake_api.transport(), max_retries=1)
    snapshot = client.get_board_items("1111")
    assert snapshot.item_count == 14
    assert len({i.id for i in snapshot.items}) == 14
    assert fake_api.request_count > 3


def test_column_ids_are_discovered_and_differ_from_titles(settings, fake_api):
    client = MondayClient(settings, transport=fake_api.transport(), max_retries=1)
    name, columns = client.get_board_columns("1111")
    assert name == "Deals"
    ids = {c.id for c in columns}
    titles = {c.title for c in columns}
    assert "numbers1" in ids and "Masked Deal value" in titles
    assert ids.isdisjoint(titles), "the fixture models Monday's opaque column IDs"


def test_items_are_flattened_by_column_title(settings, fake_api):
    client = MondayClient(settings, transport=fake_api.transport(), max_retries=1)
    snapshot = client.get_board_items("1111")
    first = snapshot.items[0]
    assert first.values["Sector/service"] == "Renewables"
    assert first.values["Masked Deal value"] == "25000000"
    assert first.group == "Deals"


def test_snapshot_to_dataframe_adds_reserved_columns(settings, fake_api):
    client = MondayClient(settings, transport=fake_api.transport(), max_retries=1)
    frame = snapshot_to_dataframe(client.get_board_items("2222"))
    assert {"__item_id", "__item_name", "__group"} <= set(frame.columns)
    assert "Serial #" in frame.columns
    assert len(frame) == 10


def test_missing_token_raises_a_configuration_error():
    with pytest.raises(MondayAuthError):
        MondayClient(make_settings(monday_api_token=None))


def test_invalid_token_maps_to_auth_error(settings, fake_api):
    fake_api.fail_with = (401, {"errors": [{"message": "unauthorized"}]})
    client = MondayClient(settings, transport=fake_api.transport(), max_retries=1)
    with pytest.raises(MondayAuthError) as exc:
        client.get_board_items("1111")
    assert "token" in exc.value.user_message.lower()


def test_board_not_found_maps_to_a_clear_error(settings, fake_api):
    client = MondayClient(settings, transport=fake_api.transport(), max_retries=1)
    with pytest.raises(MondayBoardNotFoundError) as exc:
        client.get_board_items("9999")
    assert "board" in exc.value.user_message.lower()


def test_rate_limit_is_retried_then_surfaced(settings, fake_api):
    fake_api.fail_with = (429, {"errors": [{"message": "rate limit exceeded"}]})
    client = MondayClient(settings, transport=fake_api.transport(), max_retries=2)
    with pytest.raises(MondayRateLimitError):
        client.get_board_items("1111")
    assert fake_api.request_count == 2, "the client must retry a rate-limited call"


def test_server_error_maps_to_unavailable(settings, fake_api):
    fake_api.fail_with = (503, {})
    client = MondayClient(settings, transport=fake_api.transport(), max_retries=1)
    with pytest.raises(MondayUnavailableError):
        client.get_board_items("1111")


def test_network_failure_maps_to_unavailable(settings):
    def _boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    client = MondayClient(settings, transport=httpx.MockTransport(_boom), max_retries=1)
    with pytest.raises(MondayUnavailableError):
        client.get_board_items("1111")


def test_non_json_response_is_reported_safely(settings):
    def _html(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>maintenance</html>")

    client = MondayClient(settings, transport=httpx.MockTransport(_html), max_retries=1)
    with pytest.raises(MondayError):
        client.get_board_items("1111")


def test_empty_board_returns_an_empty_snapshot(settings):
    api = FakeMondayAPI(deals=[])
    client = MondayClient(settings, transport=api.transport(), max_retries=1)
    snapshot = client.get_board_items("1111")
    assert snapshot.item_count == 0
    frame = snapshot_to_dataframe(snapshot)
    assert frame.empty
    assert "Deal Status" in frame.columns


def test_error_messages_never_leak_the_token(settings, fake_api):
    fake_api.fail_with = (401, {"errors": [{"message": "unauthorized"}]})
    client = MondayClient(settings, transport=fake_api.transport(), max_retries=1)
    with pytest.raises(MondayAuthError) as exc:
        client.get_board_items("1111")
    assert settings.monday_api_token not in str(exc.value)
    assert settings.monday_api_token not in exc.value.user_message


# --- column mapping ---------------------------------------------------------

def test_column_mapping_resolves_real_titles():
    mapping = resolve_columns(
        "1", "Deals", [BoardColumn(**c) for c in DEAL_COLUMNS], DEAL_FIELDS
    )
    assert mapping.get("value") == "Masked Deal value"
    assert mapping.get("expected_close_date") == "Tentative Close Date"
    assert mapping.get("actual_close_date") == "Close Date (A)"
    assert mapping.get("sector") == "Sector/service"
    assert "Internal Notes" in mapping.unused_columns


def test_column_mapping_handles_renamed_columns():
    """A board renamed by the evaluator must still map."""
    columns = [
        BoardColumn(id="a", title="Deal Value", type="numbers"),
        BoardColumn(id="b", title="Industry", type="text"),
        BoardColumn(id="c", title="Expected Close Date", type="date"),
        BoardColumn(id="d", title="Status", type="status"),
        BoardColumn(id="e", title="Sales Stage", type="status"),
    ]
    mapping = resolve_columns("1", "Renamed", columns, DEAL_FIELDS)
    assert mapping.get("value") == "Deal Value"
    assert mapping.get("sector") == "Industry"
    assert mapping.get("expected_close_date") == "Expected Close Date"
    assert mapping.get("status") == "Status"
    assert mapping.get("stage") == "Sales Stage"


def test_one_column_is_never_claimed_by_two_fields():
    mapping = resolve_columns(
        "2", "WOs", [BoardColumn(**c) for c in WO_COLUMNS], WORK_ORDER_FIELDS
    )
    assigned = list(mapping.mapping.values())
    assert len(assigned) == len(set(assigned))


def test_explicit_overrides_win():
    columns = [
        BoardColumn(id="a", title="Deal Value", type="numbers"),
        BoardColumn(id="b", title="Secondary Value", type="numbers"),
    ]
    mapping = resolve_columns(
        "1", "B", columns, DEAL_FIELDS, overrides={"value": "Secondary Value"}
    )
    assert mapping.get("value") == "Secondary Value"
    assert mapping.match_method["value"] == "override"


def test_unmappable_board_reports_unmapped_fields():
    columns = [BoardColumn(id="x", title="Colour", type="text")]
    mapping = resolve_columns("1", "Odd", columns, DEAL_FIELDS)
    assert "value" in mapping.unmapped_fields
    assert mapping.mapping == {}
