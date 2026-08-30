"""Fetch + normalise Monday.com boards, with short-lived caching.

Caching rules (see DECISION_LOG.md):

* the cache is a *latency* optimisation, never the source of truth,
* a short TTL (default 5 minutes) expires it automatically,
* the fetch timestamp travels with the data and is shown in the UI,
* an explicit refresh always bypasses it,
* if a refresh fails, the caller is told the fetch failed -- stale data is only
  reused when explicitly permitted, and is labelled as stale.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from analytics.normalization import (
    NormalizedDataset,
    normalize_deals,
    normalize_work_orders,
)
from config import Settings, get_settings
from monday.client import MondayClient, MondayError
from monday.column_map import DEAL_FIELDS, WORK_ORDER_FIELDS, ColumnMapping, resolve_columns
from monday.client import snapshot_to_dataframe
from utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["BusinessData", "DataService"]


@dataclass
class BusinessData:
    """One consistent snapshot of both boards, normalised and ready for analytics."""

    deals: NormalizedDataset
    work_orders: NormalizedDataset
    fetched_at: datetime
    deals_mapping: ColumnMapping | None = None
    work_orders_mapping: ColumnMapping | None = None
    warnings: list[str] = field(default_factory=list)
    is_stale: bool = False

    @property
    def age_seconds(self) -> float:
        return (datetime.now() - self.fetched_at).total_seconds()

    def source_summary(self) -> dict:
        """Everything the 'Data sources' panel needs, with no secrets."""
        return {
            "fetched_at": self.fetched_at.strftime("%Y-%m-%d %H:%M:%S"),
            "age_seconds": round(self.age_seconds),
            "is_stale": self.is_stale,
            "deals": {
                "board_id": self.deals_mapping.board_id if self.deals_mapping else None,
                "board_name": self.deals_mapping.board_name if self.deals_mapping else None,
                "rows": len(self.deals.frame),
                "mapped_fields": self.deals_mapping.describe() if self.deals_mapping else [],
                "unmapped_fields": self.deals_mapping.unmapped_fields if self.deals_mapping else [],
                "unused_columns": self.deals_mapping.unused_columns if self.deals_mapping else [],
            },
            "work_orders": {
                "board_id": self.work_orders_mapping.board_id if self.work_orders_mapping else None,
                "board_name": self.work_orders_mapping.board_name if self.work_orders_mapping else None,
                "rows": len(self.work_orders.frame),
                "mapped_fields": self.work_orders_mapping.describe() if self.work_orders_mapping else [],
                "unmapped_fields": (
                    self.work_orders_mapping.unmapped_fields if self.work_orders_mapping else []
                ),
                "unused_columns": (
                    self.work_orders_mapping.unused_columns if self.work_orders_mapping else []
                ),
            },
            "warnings": self.warnings,
        }


class DataService:
    """Owns the Monday.com round-trip and the in-process TTL cache."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: MondayClient | None = None,
        column_overrides: dict[str, dict[str, str]] | None = None,
        today: pd.Timestamp | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._overrides = column_overrides or {}
        #: Injectable "now", so tests can freeze time. ``None`` means the real today.
        self._today = today
        self._cache: BusinessData | None = None
        self._cached_at: float = 0.0
        self._lock = threading.Lock()

    @property
    def client(self) -> MondayClient:
        if self._client is None:
            self._client = MondayClient(self.settings)
        return self._client

    def cache_age_seconds(self) -> float | None:
        return None if self._cache is None else time.monotonic() - self._cached_at

    def invalidate(self) -> None:
        with self._lock:
            self._cache = None
            self._cached_at = 0.0

    def get_data(self, *, force_refresh: bool = False) -> BusinessData:
        """Return a fresh snapshot, using the TTL cache when it is still valid."""
        ttl = self.settings.cache_ttl_seconds
        with self._lock:
            age = self.cache_age_seconds()
            if not force_refresh and self._cache is not None and age is not None and age < ttl:
                logger.debug("Serving Monday.com data from cache (age %.0fs, ttl %ss)", age, ttl)
                return self._cache

        data = self._fetch()
        with self._lock:
            self._cache = data
            self._cached_at = time.monotonic()
        return data

    # -- internals ---------------------------------------------------------
    def _fetch(self) -> BusinessData:
        settings = self.settings
        warnings: list[str] = []
        today = self._today or pd.Timestamp.today().normalize()

        deals_snapshot = self.client.get_board_items(str(settings.monday_deals_board_id))
        wo_snapshot = self.client.get_board_items(str(settings.monday_work_orders_board_id))

        deals_mapping = resolve_columns(
            deals_snapshot.board_id, deals_snapshot.board_name, deals_snapshot.columns,
            DEAL_FIELDS, overrides=self._overrides.get("deals"),
        )
        wo_mapping = resolve_columns(
            wo_snapshot.board_id, wo_snapshot.board_name, wo_snapshot.columns,
            WORK_ORDER_FIELDS, overrides=self._overrides.get("work_orders"),
        )

        for label, mapping, critical in (
            ("Deals", deals_mapping, ("value", "status", "sector")),
            ("Work Orders", wo_mapping, ("execution_status", "sector", "end_date")),
        ):
            missing = [f for f in critical if not mapping.has(f)]
            if missing:
                warnings.append(
                    f"{label} board: no column matched {', '.join(missing)}. "
                    f"Related metrics will be reported as unavailable. "
                    f"Columns seen: {', '.join(mapping.column_types)[:200]}"
                )

        deals = normalize_deals(snapshot_to_dataframe(deals_snapshot), deals_mapping, today=today)
        work_orders = normalize_work_orders(
            snapshot_to_dataframe(wo_snapshot), wo_mapping,
            today=today, delay_grace_days=settings.delay_grace_days,
        )

        if deals.empty:
            warnings.append("The Deals board returned no usable rows.")
        if work_orders.empty:
            warnings.append("The Work Orders board returned no usable rows.")

        return BusinessData(
            deals=deals,
            work_orders=work_orders,
            fetched_at=datetime.now(),
            deals_mapping=deals_mapping,
            work_orders_mapping=wo_mapping,
            warnings=warnings,
        )

    def get_data_or_stale(self, *, force_refresh: bool = False) -> tuple[BusinessData | None, MondayError | None]:
        """Fetch, and on failure fall back to a clearly-labelled stale snapshot."""
        try:
            return self.get_data(force_refresh=force_refresh), None
        except MondayError as exc:
            logger.error("Monday.com fetch failed: %s", exc)
            if self._cache is not None:
                stale = BusinessData(
                    deals=self._cache.deals,
                    work_orders=self._cache.work_orders,
                    fetched_at=self._cache.fetched_at,
                    deals_mapping=self._cache.deals_mapping,
                    work_orders_mapping=self._cache.work_orders_mapping,
                    warnings=[*self._cache.warnings, f"Live refresh failed: {exc.user_message}"],
                    is_stale=True,
                )
                return stale, exc
            return None, exc
