"""Monday.com GraphQL client.

Responsibilities:

* authenticate with a personal/API token supplied through the environment,
* discover a board's real column IDs, titles and types (IDs are never assumed to
  match titles -- Monday generates ``text0``, ``date4``, ``numbers`` and so on),
* page through ``items_page`` with a cursor until every item is retrieved,
* flatten column values into a plain ``{column_title: value}`` dict,
* translate transport/API failures into typed, user-safe exceptions.
"""
from __future__ import annotations

import json
import random
import time
from collections.abc import Iterable
from typing import Any

import httpx
import pandas as pd

from config import Settings, get_settings
from monday.schemas import BoardColumn, BoardItem, BoardSnapshot
from utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "MondayError",
    "MondayAuthError",
    "MondayBoardNotFoundError",
    "MondayRateLimitError",
    "MondayUnavailableError",
    "MondayClient",
]


class MondayError(Exception):
    """Base class for every Monday.com failure surfaced to the application.

    ``user_message`` is safe to display; the exception string carries developer
    detail for the logs.
    """

    user_message = (
        "Unable to retrieve the latest Monday.com data. "
        "Please verify the board configuration and API connection."
    )

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        if user_message:
            self.user_message = user_message


class MondayAuthError(MondayError):
    user_message = (
        "Monday.com rejected the API token. Check that MONDAY_API_TOKEN is set to a "
        "valid, non-expired token with read access to both boards."
    )


class MondayBoardNotFoundError(MondayError):
    user_message = (
        "The configured Monday.com board could not be found. Check "
        "MONDAY_DEALS_BOARD_ID / MONDAY_WORK_ORDERS_BOARD_ID and that the token's "
        "account can see those boards."
    )


class MondayRateLimitError(MondayError):
    user_message = (
        "Monday.com is rate-limiting requests right now. Please wait a few seconds "
        "and use 'Refresh data' to try again."
    )


class MondayUnavailableError(MondayError):
    user_message = (
        "Monday.com could not be reached. Please check network connectivity and try again."
    )


# GraphQL ------------------------------------------------------------------

_BOARD_META_QUERY = """
query BoardMeta($boardId: [ID!]) {
  boards(ids: $boardId) {
    id
    name
    columns { id title type settings_str }
  }
}
"""

_ITEMS_FIRST_PAGE = """
query BoardItems($boardId: [ID!], $limit: Int!) {
  boards(ids: $boardId) {
    id
    name
    items_page(limit: $limit) {
      cursor
      items {
        id
        name
        group { title }
        column_values { id text type value ... on MirrorValue { display_value } }
      }
    }
  }
}
"""

_ITEMS_NEXT_PAGE = """
query NextItems($cursor: String!, $limit: Int!) {
  next_items_page(cursor: $cursor, limit: $limit) {
    cursor
    items {
      id
      name
      group { title }
      column_values { id text type value ... on MirrorValue { display_value } }
    }
  }
}
"""


class MondayClient:
    """Thin, dependency-light wrapper around the Monday.com GraphQL v2 API."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 3,
    ) -> None:
        self.settings = settings or get_settings()
        self.max_retries = max_retries
        self._transport = transport
        if not self.settings.monday_api_token:
            raise MondayAuthError(
                "MONDAY_API_TOKEN is not configured",
                user_message=(
                    "Monday.com is not configured yet. Set MONDAY_API_TOKEN, "
                    "MONDAY_DEALS_BOARD_ID and MONDAY_WORK_ORDERS_BOARD_ID."
                ),
            )

    # -- transport ---------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": str(self.settings.monday_api_token),
            "Content-Type": "application/json",
            "API-Version": self.settings.monday_api_version,
        }

    def execute(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run one GraphQL request with retry/backoff, returning the ``data`` block."""
        payload = {"query": query, "variables": variables or {}}
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                with httpx.Client(
                    timeout=self.settings.request_timeout_seconds,
                    transport=self._transport,
                ) as client:
                    response = client.post(
                        self.settings.monday_api_url, headers=self._headers(), json=payload
                    )
            except httpx.TimeoutException as exc:
                last_error = MondayUnavailableError(f"Monday.com request timed out: {exc}")
            except httpx.HTTPError as exc:
                last_error = MondayUnavailableError(f"Monday.com transport error: {exc}")
            else:
                try:
                    return self._handle_response(response)
                except MondayRateLimitError as exc:
                    last_error = exc
                except MondayUnavailableError as exc:
                    last_error = exc

            sleep_for = min(2 ** attempt + random.uniform(0, 0.4), 8)
            logger.warning(
                "Monday.com call failed (attempt %s/%s): %s -- retrying in %.1fs",
                attempt + 1,
                self.max_retries,
                last_error,
                sleep_for,
            )
            if attempt < self.max_retries - 1:
                time.sleep(sleep_for)

        raise last_error or MondayUnavailableError("Monday.com request failed")

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        if response.status_code in (401, 403):
            raise MondayAuthError(f"Monday.com auth failed with HTTP {response.status_code}")
        if response.status_code == 429:
            raise MondayRateLimitError("Monday.com returned HTTP 429 (rate limited)")
        if response.status_code >= 500:
            raise MondayUnavailableError(f"Monday.com returned HTTP {response.status_code}")
        if response.status_code >= 400:
            raise MondayError(
                f"Monday.com returned HTTP {response.status_code}: {response.text[:300]}"
            )

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise MondayError(f"Monday.com returned a non-JSON response: {exc}") from exc

        if body.get("errors"):
            raise self._classify_graphql_errors(body["errors"])
        if "error_message" in body and body.get("status_code") not in (None, 200):
            raise self._classify_graphql_errors([{"message": body["error_message"]}])
        data = body.get("data")
        if data is None:
            raise MondayError(f"Monday.com response contained no data block: {str(body)[:300]}")
        return data

    @staticmethod
    def _classify_graphql_errors(errors: Iterable[dict[str, Any]]) -> MondayError:
        messages = [str(e.get("message", e)) for e in errors]
        joined = "; ".join(messages)
        lowered = joined.lower()
        if "unauthor" in lowered or "authentication" in lowered or "not authenticated" in lowered:
            return MondayAuthError(f"Monday.com auth error: {joined}")
        if "complexity" in lowered or "rate limit" in lowered or "minute limit" in lowered:
            return MondayRateLimitError(f"Monday.com rate/complexity limit: {joined}")
        if "does not exist" in lowered or "not found" in lowered or "invalid board" in lowered:
            return MondayBoardNotFoundError(f"Monday.com board error: {joined}")
        return MondayError(f"Monday.com GraphQL error: {joined}")

    # -- board access ------------------------------------------------------
    def get_board_columns(self, board_id: str) -> tuple[str, list[BoardColumn]]:
        """Discover a board's real column IDs, titles and types."""
        data = self.execute(_BOARD_META_QUERY, {"boardId": [str(board_id)]})
        boards = data.get("boards") or []
        if not boards:
            raise MondayBoardNotFoundError(f"Board {board_id} returned no metadata")
        board = boards[0]
        columns = [
            BoardColumn(
                id=c.get("id", ""),
                title=c.get("title", ""),
                type=c.get("type", "unknown"),
                settings_str=c.get("settings_str"),
            )
            for c in (board.get("columns") or [])
        ]
        return board.get("name") or f"Board {board_id}", columns

    def get_board_items(self, board_id: str, *, page_size: int | None = None) -> BoardSnapshot:
        """Fetch every item on a board, following the ``items_page`` cursor.

        This is the generic retrieval entry point; ``get_deals`` and
        ``get_work_orders`` are thin wrappers over it.
        """
        board_id = str(board_id)
        limit = page_size or self.settings.page_size
        board_name, columns = self.get_board_columns(board_id)
        column_titles = {c.id: c.title for c in columns}
        column_types = {c.id: c.type for c in columns}

        data = self.execute(_ITEMS_FIRST_PAGE, {"boardId": [board_id], "limit": limit})
        boards = data.get("boards") or []
        if not boards:
            raise MondayBoardNotFoundError(f"Board {board_id} returned no items_page")

        page = boards[0].get("items_page") or {}
        raw_items = list(page.get("items") or [])
        cursor = page.get("cursor")

        pages = 1
        while cursor:
            if pages > 200:  # hard stop; ~50k items at the default page size
                logger.warning("Stopping pagination for board %s after %s pages", board_id, pages)
                break
            nxt = self.execute(_ITEMS_NEXT_PAGE, {"cursor": cursor, "limit": limit})
            block = nxt.get("next_items_page") or {}
            raw_items.extend(block.get("items") or [])
            cursor = block.get("cursor")
            pages += 1

        items = [self._flatten_item(i, column_titles, column_types) for i in raw_items]
        logger.info(
            "Fetched %s items from board %s (%s) across %s page(s)",
            len(items), board_id, board_name, pages,
        )
        return BoardSnapshot(
            board_id=board_id, board_name=board_name, columns=columns, items=items
        )

    @staticmethod
    def _flatten_item(
        raw: dict[str, Any],
        column_titles: dict[str, str],
        column_types: dict[str, str],
    ) -> BoardItem:
        """Flatten ``column_values`` into ``{column_title: python value}``.

        Monday returns both a display ``text`` and a structured ``value`` JSON.
        We prefer the structured value for dates and numbers (it is unambiguous)
        and fall back to the display text for everything else.
        """
        values: dict[str, object] = {}
        for cv in raw.get("column_values") or []:
            column_id = cv.get("id", "")
            title = column_titles.get(column_id, column_id)
            column_type = cv.get("type") or column_types.get(column_id, "")
            values[title] = _column_value(cv, column_type)
        return BoardItem(
            id=str(raw.get("id", "")),
            name=raw.get("name") or "",
            group=(raw.get("group") or {}).get("title"),
            values=values,
        )


def _column_value(column_value: dict[str, Any], column_type: str) -> object:
    """Extract the most reliable representation of a single Monday column value."""
    text = column_value.get("text")
    display = column_value.get("display_value")  # mirror / lookup columns
    raw_json = column_value.get("value")

    parsed: Any = None
    if raw_json:
        try:
            parsed = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            parsed = None

    if column_type == "date" and isinstance(parsed, dict):
        date_part = parsed.get("date")
        if date_part:
            time_part = parsed.get("time")
            return f"{date_part} {time_part}" if time_part else date_part
    if column_type in {"numbers", "numeric"} and parsed not in (None, ""):
        return parsed
    if column_type == "status" and isinstance(parsed, dict) and text:
        return text
    if isinstance(parsed, dict) and "label" in parsed and not text:
        return parsed["label"]

    if display not in (None, ""):
        return display
    if text not in (None, ""):
        return text
    if isinstance(parsed, (str, int, float)):
        return parsed
    return None


def snapshot_to_dataframe(snapshot: BoardSnapshot) -> pd.DataFrame:
    """Convert a :class:`BoardSnapshot` into a raw (un-normalised) DataFrame.

    Item ID / name / group are exposed as reserved ``__``-prefixed columns so they
    can never collide with a user-defined board column title.
    """
    if not snapshot.items:
        return pd.DataFrame(columns=["__item_id", "__item_name", "__group", *snapshot.column_titles()])
    rows = []
    for item in snapshot.items:
        row: dict[str, object] = {
            "__item_id": item.id,
            "__item_name": item.name,
            "__group": item.group,
        }
        row.update(item.values)
        rows.append(row)
    frame = pd.DataFrame(rows)
    for title in snapshot.column_titles():
        if title not in frame.columns:
            frame[title] = None
    return frame
