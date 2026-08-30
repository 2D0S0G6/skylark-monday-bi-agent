"""Runtime configuration, sourced from environment variables / Streamlit secrets.

No credential is ever hard-coded. Values are read from (in order):

1. process environment,
2. a local ``.env`` file (developer convenience, git-ignored),
3. ``st.secrets`` when running on Streamlit Community Cloud.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv(override=False)

#: Matches the numeric board ID inside a Monday board URL.
_BOARD_ID_IN_URL = re.compile(r"/boards/(\d+)")

#: Groq deprecates models periodically. If this one 404s, run
#: `python -m tools.list_models` to see what the account can currently use.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def _from_streamlit_secrets(key: str) -> str | None:
    """Read a key from ``st.secrets`` without requiring Streamlit at import time."""
    try:
        import streamlit as st  # noqa: PLC0415 - optional dependency at this layer
    except Exception:  # pragma: no cover - streamlit always present in the app
        return None
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:  # noqa: BLE001 - no secrets.toml configured
        return None
    return None


def env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    if value is None or value == "":
        value = _from_streamlit_secrets(key)
    if value is None or value == "":
        return default
    return value.strip()


def env_int(key: str, default: int) -> int:
    raw = env(key)
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return default


def board_id(key: str) -> str | None:
    """Read a board ID, accepting either the bare number or a full board URL.

    Pasting the whole URL (``https://acme.monday.com/boards/1234567890``) is the
    most common setup slip, so we extract the ID rather than failing later with an
    opaque "board not found" from the API.
    """
    raw = env(key)
    if not raw:
        return None
    raw = raw.strip().strip('"').strip("'")
    match = _BOARD_ID_IN_URL.search(raw)
    if match:
        return match.group(1)
    digits = re.fullmatch(r"\d+", raw)
    if digits:
        return raw
    # Last resort: a trailing number, e.g. "Deals 1234567890".
    trailing = re.search(r"(\d{6,})\s*$", raw)
    return trailing.group(1) if trailing else raw


def env_bool(key: str, default: bool = False) -> bool:
    raw = env(key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    monday_api_token: str | None
    monday_deals_board_id: str | None
    monday_work_orders_board_id: str | None
    monday_api_url: str
    monday_api_version: str
    groq_api_key: str | None
    groq_model: str
    cache_ttl_seconds: int
    fiscal_year_start_month: int
    page_size: int
    request_timeout_seconds: int
    delay_grace_days: int
    missing_reasons: tuple[str, ...] = field(default=("missing", "invalid"))

    # -- readiness -------------------------------------------------------
    @property
    def monday_configured(self) -> bool:
        return bool(
            self.monday_api_token
            and self.monday_deals_board_id
            and self.monday_work_orders_board_id
        )

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key)

    def missing_settings(self) -> list[str]:
        missing = []
        if not self.monday_api_token:
            missing.append("MONDAY_API_TOKEN")
        if not self.monday_deals_board_id:
            missing.append("MONDAY_DEALS_BOARD_ID")
        if not self.monday_work_orders_board_id:
            missing.append("MONDAY_WORK_ORDERS_BOARD_ID")
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        return missing

    def safe_summary(self) -> dict[str, str]:
        """Configuration snapshot for the UI. Secrets are shown only as present/absent."""
        return {
            "Monday API token": "configured" if self.monday_api_token else "missing",
            "Deals board ID": self.monday_deals_board_id or "missing",
            "Work Orders board ID": self.monday_work_orders_board_id or "missing",
            "Groq API key": "configured" if self.groq_api_key else "missing",
            "Groq model": self.groq_model,
            "Cache TTL": f"{self.cache_ttl_seconds}s",
            "Fiscal year starts": f"month {self.fiscal_year_start_month}",
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        monday_api_token=env("MONDAY_API_TOKEN"),
        monday_deals_board_id=board_id("MONDAY_DEALS_BOARD_ID"),
        monday_work_orders_board_id=board_id("MONDAY_WORK_ORDERS_BOARD_ID"),
        monday_api_url=env("MONDAY_API_URL", "https://api.monday.com/v2"),
        monday_api_version=env("MONDAY_API_VERSION", "2024-10"),
        groq_api_key=env("GROQ_API_KEY"),
        groq_model=env("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        cache_ttl_seconds=env_int("CACHE_TTL_SECONDS", 300),
        fiscal_year_start_month=env_int("FISCAL_YEAR_START_MONTH", 4),
        page_size=env_int("MONDAY_PAGE_SIZE", 250),
        request_timeout_seconds=env_int("REQUEST_TIMEOUT_SECONDS", 45),
        delay_grace_days=env_int("DELAY_GRACE_DAYS", 0),
    )


def reset_settings_cache() -> None:
    """Used by tests and the UI's 'reload configuration' path."""
    get_settings.cache_clear()
