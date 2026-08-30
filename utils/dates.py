"""Date parsing and fiscal/calendar period helpers.

Monday.com date columns normally return ISO strings, but text columns imported
from a spreadsheet can contain anything. Every parse is defensive: an
unparseable cell yields ``None`` plus a reason, never an exception.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from utils.numbers import is_missing

__all__ = [
    "ParsedDate",
    "DateRange",
    "parse_date",
    "parse_date_series",
    "current_quarter",
    "quarter_range",
    "resolve_date_range",
    "describe_range",
]

# Tried in order before falling back to pandas' own inference.
_EXPLICIT_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%d-%b-%Y", "%d %b %Y", "%d-%b-%y", "%d %b %y",
    "%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y",
    "%m/%d/%Y", "%m-%d-%Y",
    "%d/%m/%y", "%m/%d/%y",
    "%Y%m%d",
)

_AMBIGUOUS_SLASH = re.compile(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})")


@dataclass(frozen=True)
class ParsedDate:
    value: pd.Timestamp | None
    reason: str | None = None
    raw: object = None

    @property
    def ok(self) -> bool:
        return self.value is not None


@dataclass(frozen=True)
class DateRange:
    start: pd.Timestamp
    end: pd.Timestamp
    label: str

    def contains(self, ts: pd.Timestamp | None) -> bool:
        if ts is None or pd.isna(ts):
            return False
        return self.start <= ts <= self.end

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "start": self.start.date().isoformat(),
            "end": self.end.date().isoformat(),
        }


def parse_date(value: object, *, dayfirst: bool = True) -> ParsedDate:
    """Parse a single messy date cell into a pandas ``Timestamp``.

    ``dayfirst`` controls how ``12/08/2026`` is read. The data in this project is
    Indian-sourced, so day-first is the default; when the first component is
    greater than 12 the ordering is unambiguous and the hint is ignored.
    """
    if is_missing(value):
        return ParsedDate(None, "missing", value)

    if isinstance(value, pd.Timestamp):
        return ParsedDate(None, "invalid", value) if pd.isna(value) else ParsedDate(value, None, value)
    if isinstance(value, datetime):
        return ParsedDate(pd.Timestamp(value), None, value)
    if isinstance(value, date):
        return ParsedDate(pd.Timestamp(value), None, value)
    if isinstance(value, (int, float)):
        # Excel serial dates land here when a sheet is exported without formats.
        try:
            if 20000 < float(value) < 60000:
                return ParsedDate(
                    pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(value)), None, value
                )
        except (ValueError, OverflowError):
            pass
        return ParsedDate(None, "invalid", value)

    text = str(value).strip()
    if not text:
        return ParsedDate(None, "missing", value)

    ambiguous = _AMBIGUOUS_SLASH.match(text)
    effective_dayfirst = dayfirst
    if ambiguous:
        first, second = int(ambiguous.group(1)), int(ambiguous.group(2))
        if first > 12 >= second:
            effective_dayfirst = True
        elif second > 12 >= first:
            effective_dayfirst = False

    formats = _EXPLICIT_FORMATS
    if not effective_dayfirst:
        formats = tuple(f for f in formats if not f.startswith("%d")) + formats

    for fmt in formats:
        try:
            return ParsedDate(pd.Timestamp(datetime.strptime(text, fmt)), None, value)
        except ValueError:
            continue

    try:
        parsed = pd.to_datetime(text, dayfirst=effective_dayfirst, errors="raise")
    except Exception:  # noqa: BLE001 - pandas raises a wide family of errors
        return ParsedDate(None, "invalid", value)
    if pd.isna(parsed):
        return ParsedDate(None, "invalid", value)
    if isinstance(parsed, pd.Series):  # pragma: no cover - defensive
        return ParsedDate(None, "invalid", value)
    return ParsedDate(pd.Timestamp(parsed), None, value)


def parse_date_series(series: pd.Series, *, dayfirst: bool = True) -> tuple[pd.Series, pd.Series]:
    """Vectorised wrapper around :func:`parse_date`.

    Returns ``(timestamps, reasons)`` where ``reasons`` holds ``None`` for parsed
    rows and ``"missing"``/``"invalid"`` otherwise.
    """
    parsed = [parse_date(v, dayfirst=dayfirst) for v in series]
    values = pd.Series(
        [p.value for p in parsed], index=series.index, dtype="datetime64[ns]"
    )
    reasons = pd.Series([p.reason for p in parsed], index=series.index, dtype="object")
    return values, reasons


def _fiscal_quarter(ts: pd.Timestamp, fy_start_month: int) -> tuple[int, int]:
    """Return ``(quarter_number, fiscal_year_start_year)`` for ``ts``."""
    offset = (ts.month - fy_start_month) % 12
    quarter = offset // 3 + 1
    fy_start_year = ts.year if ts.month >= fy_start_month else ts.year - 1
    if fy_start_month == 1:
        fy_start_year = ts.year
    return quarter, fy_start_year


def quarter_range(
    ts: pd.Timestamp, *, fy_start_month: int = 4, offset_quarters: int = 0
) -> DateRange:
    """Build the fiscal quarter containing ``ts``, shifted by ``offset_quarters``."""
    quarter, fy_start_year = _fiscal_quarter(ts, fy_start_month)
    absolute = (fy_start_year * 4) + (quarter - 1) + offset_quarters
    fy_start_year, quarter_index = divmod(absolute, 4)
    start_month_abs = (fy_start_month - 1) + quarter_index * 3
    start_year = fy_start_year + start_month_abs // 12
    start_month = start_month_abs % 12 + 1
    start = pd.Timestamp(year=start_year, month=start_month, day=1)
    end = start + pd.offsets.QuarterEnd(startingMonth=(start_month + 2 - 1) % 12 + 1)
    end = (start + pd.DateOffset(months=3)) - pd.Timedelta(days=1)
    end = pd.Timestamp(end.date()) + pd.Timedelta(hours=23, minutes=59, seconds=59)

    if fy_start_month == 1:
        label = f"Q{quarter_index + 1} {start_year}"
    else:
        fy_end_short = str((fy_start_year + 1) % 100).zfill(2)
        label = f"Q{quarter_index + 1} FY{fy_start_year}-{fy_end_short}"
    label += f" ({start.strftime('%b %Y')}–{(start + pd.DateOffset(months=2)).strftime('%b %Y')})"
    return DateRange(start, end, label)


def current_quarter(*, today: pd.Timestamp | None = None, fy_start_month: int = 4) -> DateRange:
    return quarter_range(today or pd.Timestamp.today().normalize(), fy_start_month=fy_start_month)


def resolve_date_range(
    token: str | None,
    *,
    today: pd.Timestamp | None = None,
    fy_start_month: int = 4,
) -> DateRange | None:
    """Turn a planner token such as ``"current_quarter"`` into a concrete range.

    Returns ``None`` for ``"all_time"`` / unknown tokens, which callers treat as
    "no date filter".
    """
    if not token:
        return None
    now = today or pd.Timestamp.today().normalize()
    key = str(token).strip().lower().replace("-", "_").replace(" ", "_")

    if key in {"all", "all_time", "any", "none", "unspecified"}:
        return None
    if key in {"current_quarter", "this_quarter", "quarter"}:
        return quarter_range(now, fy_start_month=fy_start_month)
    if key in {"next_quarter", "upcoming_quarter"}:
        return quarter_range(now, fy_start_month=fy_start_month, offset_quarters=1)
    if key in {"last_quarter", "previous_quarter", "prior_quarter"}:
        return quarter_range(now, fy_start_month=fy_start_month, offset_quarters=-1)
    if key in {"current_month", "this_month", "month"}:
        start = now.replace(day=1)
        end = start + pd.offsets.MonthEnd(1)
        return DateRange(start, _end_of_day(end), start.strftime("%B %Y"))
    if key in {"last_month", "previous_month"}:
        start = (now.replace(day=1) - pd.DateOffset(months=1))
        end = start + pd.offsets.MonthEnd(1)
        return DateRange(start, _end_of_day(end), start.strftime("%B %Y"))
    if key in {"current_year", "this_year", "year", "calendar_year"}:
        start = pd.Timestamp(year=now.year, month=1, day=1)
        return DateRange(start, _end_of_day(pd.Timestamp(year=now.year, month=12, day=31)), str(now.year))
    if key in {"current_fy", "this_fy", "fiscal_year", "current_fiscal_year", "fy"}:
        fy_year = now.year if now.month >= fy_start_month else now.year - 1
        start = pd.Timestamp(year=fy_year, month=fy_start_month, day=1)
        end = start + pd.DateOffset(months=12) - pd.Timedelta(days=1)
        label = f"FY{fy_year}-{str((fy_year + 1) % 100).zfill(2)}"
        return DateRange(start, _end_of_day(end), label)
    if key in {"last_fy", "previous_fy", "last_fiscal_year"}:
        fy_year = (now.year if now.month >= fy_start_month else now.year - 1) - 1
        start = pd.Timestamp(year=fy_year, month=fy_start_month, day=1)
        end = start + pd.DateOffset(months=12) - pd.Timedelta(days=1)
        label = f"FY{fy_year}-{str((fy_year + 1) % 100).zfill(2)}"
        return DateRange(start, _end_of_day(end), label)
    if key in {"next_30_days", "next_month_rolling", "next_30d"}:
        return DateRange(now, _end_of_day(now + pd.Timedelta(days=30)), "next 30 days")
    if key in {"next_90_days", "next_quarter_rolling", "next_90d"}:
        return DateRange(now, _end_of_day(now + pd.Timedelta(days=90)), "next 90 days")
    if key in {"last_30_days", "past_30_days"}:
        return DateRange(now - pd.Timedelta(days=30), _end_of_day(now), "last 30 days")
    if key in {"last_90_days", "past_90_days"}:
        return DateRange(now - pd.Timedelta(days=90), _end_of_day(now), "last 90 days")
    if key in {"ytd", "year_to_date"}:
        start = pd.Timestamp(year=now.year, month=1, day=1)
        return DateRange(start, _end_of_day(now), f"{now.year} year-to-date")
    if key in {"overdue", "past", "past_due"}:
        return DateRange(pd.Timestamp("1970-01-01"), _end_of_day(now - pd.Timedelta(days=1)), "on or before today")

    # Explicit "Q3 FY2025-26" / "Q1 2026" style tokens.
    match = re.match(r"^q([1-4])[_ ]?(?:fy)?(\d{4})", key)
    if match:
        quarter_index = int(match.group(1)) - 1
        year = int(match.group(2))
        start_month_abs = (fy_start_month - 1) + quarter_index * 3
        start = pd.Timestamp(
            year=year + start_month_abs // 12, month=start_month_abs % 12 + 1, day=1
        )
        end = start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        prefix = "FY" if fy_start_month != 1 else ""
        return DateRange(start, _end_of_day(end), f"Q{quarter_index + 1} {prefix}{year}")

    match = re.match(r"^(\d{4})$", key)
    if match:
        year = int(match.group(1))
        return DateRange(
            pd.Timestamp(year=year, month=1, day=1),
            _end_of_day(pd.Timestamp(year=year, month=12, day=31)),
            str(year),
        )
    return None


def _end_of_day(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts.date()) + pd.Timedelta(hours=23, minutes=59, seconds=59)


def describe_range(date_range: DateRange | None) -> str:
    return date_range.label if date_range else "all time"
