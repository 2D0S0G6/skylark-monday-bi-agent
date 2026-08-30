"""Typed structures describing what we read back from Monday.com."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, Field


class BoardColumn(BaseModel):
    """A column definition as returned by the Monday.com API."""

    id: str
    title: str
    type: str = "unknown"
    settings_str: str | None = None


class BoardItem(BaseModel):
    """A single Monday.com item flattened to ``{column_title: value}``."""

    id: str
    name: str
    group: str | None = None
    values: dict[str, object] = Field(default_factory=dict)


@dataclass
class BoardSnapshot:
    """Everything fetched for one board in a single refresh."""

    board_id: str
    board_name: str
    columns: list[BoardColumn]
    items: list[BoardItem]
    fetched_at: datetime = field(default_factory=datetime.now)

    @property
    def item_count(self) -> int:
        return len(self.items)

    def column_titles(self) -> list[str]:
        return [c.title for c in self.columns]
