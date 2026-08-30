"""Test fixtures.

A fake Monday.com GraphQL endpoint is served through ``httpx.MockTransport`` so
the entire stack -- client, pagination, column discovery, normalisation,
analytics, orchestration -- can be exercised without a live API, a token, or a
network connection.

The fake board deliberately reproduces the messiness of the supplied seed files:
mixed date formats, ``N/A`` placeholders, currency strings, casing variants,
duplicate rows, an echoed header row and unmapped extra columns.
"""
from __future__ import annotations

import json

import httpx
import pandas as pd
import pytest

from config import Settings
from monday.column_map import DEAL_FIELDS, WORK_ORDER_FIELDS, resolve_columns
from monday.schemas import BoardColumn

TODAY = pd.Timestamp("2026-02-15")


def make_settings(**overrides) -> Settings:
    base = dict(
        monday_api_token="test-token",
        monday_deals_board_id="1111",
        monday_work_orders_board_id="2222",
        monday_api_url="https://api.monday.com/v2",
        monday_api_version="2024-10",
        groq_api_key=None,
        groq_model="llama-3.3-70b-versatile",
        cache_ttl_seconds=300,
        fiscal_year_start_month=4,
        page_size=2,
        request_timeout_seconds=10,
        delay_grace_days=0,
    )
    base.update(overrides)
    return Settings(**base)


# --- fake board definitions -------------------------------------------------
# Column IDs deliberately differ from titles, exactly as Monday generates them.

DEAL_COLUMNS = [
    {"id": "text_owner", "title": "Owner code", "type": "text"},
    {"id": "text8", "title": "Client Code", "type": "text"},
    {"id": "status", "title": "Deal Status", "type": "status"},
    {"id": "date4", "title": "Close Date (A)", "type": "date"},
    {"id": "dropdown9", "title": "Closure Probability", "type": "dropdown"},
    {"id": "numbers1", "title": "Masked Deal value", "type": "numbers"},
    {"id": "date__1", "title": "Tentative Close Date", "type": "date"},
    {"id": "status_1", "title": "Deal Stage", "type": "status"},
    {"id": "text_product", "title": "Product deal", "type": "text"},
    {"id": "text_sector", "title": "Sector/service", "type": "text"},
    {"id": "date_created", "title": "Created Date", "type": "date"},
    {"id": "long_text9", "title": "Internal Notes", "type": "long_text"},
]

WO_COLUMNS = [
    {"id": "text0", "title": "Deal name masked", "type": "text"},
    {"id": "text1", "title": "Customer Name Code", "type": "text"},
    {"id": "text2", "title": "Serial #", "type": "text"},
    {"id": "status5", "title": "Nature of Work", "type": "status"},
    {"id": "status6", "title": "Execution Status", "type": "status"},
    {"id": "date1", "title": "Date of PO/LOI", "type": "date"},
    {"id": "date2", "title": "Probable Start Date", "type": "date"},
    {"id": "date3", "title": "Probable End Date", "type": "date"},
    {"id": "text_bd", "title": "BD/KAM Personnel code", "type": "text"},
    {"id": "text_sec", "title": "Sector", "type": "text"},
    {"id": "text_tow", "title": "Type of Work", "type": "text"},
    {"id": "numbers4", "title": "Amount in Rupees (Excl of GST) (Masked)", "type": "numbers"},
    {"id": "numbers5", "title": "Billed Value in Rupees (Excl of GST.) (Masked)", "type": "numbers"},
    {"id": "numbers6", "title": "Collected Amount in Rupees (Incl of GST.) (Masked)", "type": "numbers"},
    {"id": "status7", "title": "WO Status (billed)", "type": "status"},
]


def _deal(idx, owner, client, status, close_a, prob, value, tentative, stage, product,
          sector, created):
    return {
        "id": str(1000 + idx),
        "name": f"Deal {idx}",
        "group": {"title": "Deals"},
        "column_values": [
            {"id": "text_owner", "text": owner, "type": "text", "value": None},
            {"id": "text8", "text": client, "type": "text", "value": None},
            {"id": "status", "text": status, "type": "status", "value": None},
            {"id": "date4", "text": close_a, "type": "date", "value": None},
            {"id": "dropdown9", "text": prob, "type": "dropdown", "value": None},
            {"id": "numbers1", "text": value, "type": "numbers", "value": None},
            {"id": "date__1", "text": tentative, "type": "date", "value": None},
            {"id": "status_1", "text": stage, "type": "status", "value": None},
            {"id": "text_product", "text": product, "type": "text", "value": None},
            {"id": "text_sector", "text": sector, "type": "text", "value": None},
            {"id": "date_created", "text": created, "type": "date", "value": None},
            {"id": "long_text9", "text": "", "type": "long_text", "value": None},
        ],
    }


#: 14 deals covering every messy pattern found in the seed workbook.
DEAL_ROWS = [
    # sector casing variants, currency strings, several date formats
    _deal(1, "OWNER_001", "COMPANY001", "Open", "", "High", "25000000",
          "2026-02-26", "E. Proposal/Commercials Sent", "Pure Service", "Renewables", "2025-12-26"),
    _deal(2, "OWNER_001", "COMPANY002", "open", None, "Medium", "₹2.5 Cr",
          "12/03/2026", "F. Negotiations", None, "renewables", "2025-09-15"),
    _deal(3, "OWNER_002", "COMPANY003", "Open", "", None, "N/A",
          "12-Mar-26", "B. Sales Qualified Leads", None, "RENEWABLES", "2025-11-12"),
    _deal(4, "OWNER_002", "COMPANY004", "Open", "", "Low", "1,50,00,000",
          "Mar 20, 2026", "A. Lead Generated", None, "Mining", "2025-10-14"),
    _deal(5, "OWNER_002", "COMPANY005", "Won", "2025-12-31", "High", "50 L",
          "2025-12-20", "G. Project Won", "Hardware", "Mining", "2025-06-10"),
    _deal(6, "OWNER_003", "COMPANY006", "won", "2025-11-21", None, "12000000",
          "2025-11-15", "Project Completed", None, "mining sector", "2025-05-01"),
    _deal(7, "OWNER_003", "COMPANY007", "Dead", "", "Low", "3000000",
          "2025-06-11", "L. Project Lost", None, "Powerline", "2024-11-17"),
    _deal(8, "OWNER_001", "COMPANY008", "Dead", "", None, "-",
          "not a date", "O. Not Relevant at all", None, "Powerline", "2024-08-09"),
    _deal(9, "OWNER_004", "COMPANY009", "On Hold", "", "Medium", "Unknown",
          "", "M. Projects On Hold", None, "Railways", "2024-11-17"),
    # open deal whose expected close is in the past -> overdue risk signal
    _deal(10, "OWNER_004", "COMPANY010", "Open", "", "Low", "8000000",
          "2025-09-30", "D. Feasibility", "Spectra Deal", "Railways", "2024-08-30"),
    # missing status -> inferred from the stage
    _deal(11, "OWNER_005", "COMPANY011", "", "", "High", "4500000",
          "2026-05-15", "E. Proposal/Commercials Sent", None, "Construction", "2026-01-05"),
    # unparseable value + unknown sector
    _deal(12, "OWNER_005", "COMPANY012", "Open", "", "Medium", "twelve lakhs",
          "2026-06-01", "C. Demo Done", None, "Deep Sea Robotics", "2026-01-20"),
    # exact duplicate of deal 1 (same name, client, value, date and stage)
    {**_deal(13, "OWNER_001", "COMPANY001", "Open", "", "High", "25000000",
             "2026-02-26", "E. Proposal/Commercials Sent", "Pure Service", "Renewables",
             "2025-12-26"), "name": "Deal 1"},
    # header echo row -- must be dropped
    {
        "id": "1014", "name": "Header echo", "group": {"title": "Deals"},
        "column_values": [
            {"id": "text_owner", "text": "Owner code", "type": "text", "value": None},
            {"id": "text8", "text": "Client Code", "type": "text", "value": None},
            {"id": "status", "text": "Deal Status", "type": "status", "value": None},
            {"id": "date4", "text": "Close Date (A)", "type": "date", "value": None},
            {"id": "dropdown9", "text": "Closure Probability", "type": "dropdown", "value": None},
            {"id": "numbers1", "text": "", "type": "numbers", "value": None},
            {"id": "date__1", "text": "Tentative Close Date", "type": "date", "value": None},
            {"id": "status_1", "text": "Deal Stage", "type": "status", "value": None},
            {"id": "text_product", "text": "Product deal", "type": "text", "value": None},
            {"id": "text_sector", "text": "Sector/service", "type": "text", "value": None},
            {"id": "date_created", "text": "Created Date", "type": "date", "value": None},
            {"id": "long_text9", "text": "", "type": "long_text", "value": None},
        ],
    },
]


def _wo(idx, deal, customer, serial, nature, status, po, start, end, bd, sector, tow,
        amount, billed, collected, wo_status):
    return {
        "id": str(2000 + idx),
        "name": serial or f"WO {idx}",
        "group": {"title": "Work Orders"},
        "column_values": [
            {"id": "text0", "text": deal, "type": "text", "value": None},
            {"id": "text1", "text": customer, "type": "text", "value": None},
            {"id": "text2", "text": serial, "type": "text", "value": None},
            {"id": "status5", "text": nature, "type": "status", "value": None},
            {"id": "status6", "text": status, "type": "status", "value": None},
            {"id": "date1", "text": po, "type": "date", "value": None},
            {"id": "date2", "text": start, "type": "date", "value": None},
            {"id": "date3", "text": end, "type": "date", "value": None},
            {"id": "text_bd", "text": bd, "type": "text", "value": None},
            {"id": "text_sec", "text": sector, "type": "text", "value": None},
            {"id": "text_tow", "text": tow, "type": "text", "value": None},
            {"id": "numbers4", "text": amount, "type": "numbers", "value": None},
            {"id": "numbers5", "text": billed, "type": "numbers", "value": None},
            {"id": "numbers6", "text": collected, "type": "numbers", "value": None},
            {"id": "status7", "text": wo_status, "type": "status", "value": None},
        ],
    }


#: 10 work orders, including delayed, blocked, undated and unparseable rows.
WO_ROWS = [
    _wo(1, "Deal 5", "WOCOMPANY_001", "SDPL-001", "One time Project", "Completed",
        "2025-05-16", "2025-05-20", "2025-05-31", "OWNER_003", "Mining",
        "Topography Survey: RGB", "184980", "184980", "160069", "Closed"),
    # ongoing and past its end date -> delayed
    _wo(2, "Deal 6", "WOCOMPANY_002", "SDPL-002", "Monthly Contract", "Ongoing",
        "2025-05-16", "2025-05-01", "2025-12-31", "OWNER_003", "Mining",
        "Volumetric survey", "2984097", "23959", "", "Open"),
    _wo(3, "Deal 1", "WOCOMPANY_003", "SDPL-003", "One time Project", "Not Started",
        "2025-07-31", "2026-03-11", "2026-04-15", "OWNER_002", "Renewables",
        "Hydrology", "154150", "", "", "Open"),
    # blocked and overdue
    _wo(4, "Deal 2", "WOCOMPANY_004", "SDPL-004", "Proof of Concept", "Pause / struck",
        "2025-06-01", "2025-06-10", "2025-08-15", "OWNER_002", "renewables",
        "Powerline Inspection", "₹12 Lakhs", "", "", "Open"),
    _wo(5, "Deal 7", "WOCOMPANY_005", "SDPL-005", "Annual Rate Contract",
        "Executed until current month", "2025-05-15", "2025-05-19", "2026-11-19",
        "OWNER_005", "Powerline", "Others", "3995568", "3662604", "1405108", "Open"),
    # no end date -> cannot be assessed for delay
    _wo(6, "Deal 9", "WOCOMPANY_006", "SDPL-006", "One time Project", "Ongoing",
        "2025-09-01", "2025-09-10", "", "OWNER_001", "Railways",
        "Raw images/videography", "500000", "", "", "Open"),
    # unknown execution status + unparseable amount
    _wo(7, "Deal 10", "WOCOMPANY_007", "SDPL-007", "One time Project", "???",
        "2025-10-01", "2025-10-05", "2025-11-30", "OWNER_001", "Railways",
        "Others", "N/A", "", "", "Open"),
    _wo(8, "Deal 11", "WOCOMPANY_008", "SDPL-008", "One time Project", "Completed",
        "2025-11-01", "2025-11-05", "2025-12-20", "OWNER_004", "Construction",
        "Topography Survey: RGB", "750000", "750000", "750000", "Closed"),
    _wo(9, "Deal 12", "WOCOMPANY_009", "SDPL-009", "Monthly Contract", "Ongoing",
        "2026-01-05", "2026-01-10", "2026-06-30", "OWNER_004", "Mining",
        "Volumetric survey", "1200000", "", "", "Open"),
    # duplicate serial number
    _wo(10, "Deal 12", "WOCOMPANY_009", "SDPL-009", "Monthly Contract", "Ongoing",
        "2026-01-05", "2026-01-10", "2026-06-30", "OWNER_004", "Mining",
        "Volumetric survey", "1200000", "", "", "Open"),
]


class FakeMondayAPI:
    """In-memory Monday.com GraphQL server backed by ``httpx.MockTransport``."""

    def __init__(self, deals=None, work_orders=None, *, deal_columns=None, wo_columns=None):
        self.boards = {
            "1111": {
                "name": "Deals",
                "columns": deal_columns if deal_columns is not None else DEAL_COLUMNS,
                "items": deals if deals is not None else DEAL_ROWS,
            },
            "2222": {
                "name": "Work Orders",
                "columns": wo_columns if wo_columns is not None else WO_COLUMNS,
                "items": work_orders if work_orders is not None else WO_ROWS,
            },
        }
        self.request_count = 0
        self.fail_with: tuple[int, dict] | None = None
        self._cursors: dict[str, tuple[str, int]] = {}

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.request_count += 1
        if self.fail_with:
            status, body = self.fail_with
            return httpx.Response(status, json=body)

        payload = json.loads(request.content.decode())
        query = payload.get("query", "")
        variables = payload.get("variables", {})

        if "columns" in query and "items_page" not in query:
            board_id = str(variables["boardId"][0])
            board = self.boards.get(board_id)
            if not board:
                return httpx.Response(
                    200, json={"errors": [{"message": f"Board {board_id} does not exist"}]}
                )
            return httpx.Response(200, json={"data": {"boards": [{
                "id": board_id, "name": board["name"], "columns": board["columns"],
            }]}})

        if "next_items_page" in query:
            cursor = variables["cursor"]
            limit = int(variables.get("limit", 100))
            board_id, offset = self._cursors[cursor]
            items = self.boards[board_id]["items"][offset : offset + limit]
            next_cursor = None
            if offset + limit < len(self.boards[board_id]["items"]):
                next_cursor = f"cursor-{board_id}-{offset + limit}"
                self._cursors[next_cursor] = (board_id, offset + limit)
            return httpx.Response(
                200, json={"data": {"next_items_page": {"cursor": next_cursor, "items": items}}}
            )

        if "items_page(" in query:
            board_id = str(variables["boardId"][0])
            limit = int(variables.get("limit", 100))
            board = self.boards.get(board_id)
            if not board:
                return httpx.Response(
                    200, json={"errors": [{"message": f"Board {board_id} does not exist"}]}
                )
            items = board["items"][:limit]
            cursor = None
            if len(board["items"]) > limit:
                cursor = f"cursor-{board_id}-{limit}"
                self._cursors[cursor] = (board_id, limit)
            return httpx.Response(200, json={"data": {"boards": [{
                "id": board_id, "name": board["name"],
                "items_page": {"cursor": cursor, "items": items},
            }]}})

        return httpx.Response(200, json={"errors": [{"message": "unrecognised query"}]})


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def fake_api() -> FakeMondayAPI:
    return FakeMondayAPI()


@pytest.fixture
def monday_client(settings, fake_api):
    from monday.client import MondayClient

    return MondayClient(settings, transport=fake_api.transport(), max_retries=1)


@pytest.fixture
def business_data(settings, monday_client):
    """A full normalised snapshot of both fake boards, with time frozen at TODAY."""
    from agent.data_service import DataService

    service = DataService(settings, client=monday_client, today=TODAY)
    return service.get_data()


@pytest.fixture
def deals_dataset(business_data):
    return business_data.deals


@pytest.fixture
def work_orders_dataset(business_data):
    return business_data.work_orders


def make_mapping(columns: list[dict], specs, board_id="1", board_name="Test"):
    """Resolve a column mapping from raw column dicts (used by unit tests)."""
    return resolve_columns(
        board_id, board_name, [BoardColumn(**c) for c in columns], specs
    )


@pytest.fixture
def deal_mapping():
    return make_mapping(DEAL_COLUMNS, DEAL_FIELDS, "1111", "Deals")


@pytest.fixture
def wo_mapping():
    return make_mapping(WO_COLUMNS, WORK_ORDER_FIELDS, "2222", "Work Orders")
