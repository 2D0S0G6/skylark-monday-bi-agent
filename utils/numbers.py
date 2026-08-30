"""Safe numeric / currency parsing for messy spreadsheet-sourced values.

Design rules (see DECISION_LOG.md):

* A value that cannot be parsed **confidently** is returned as ``None`` with a
  reason. It is never silently coerced to ``0`` -- a missing deal value is not a
  zero-rupee deal.
* Indian magnitude suffixes are expanded using the standard definitions:
  ``1 Lakh = 1e5``, ``1 Crore = 1e7``. ``K/M/B`` use the western definitions.
* Currency is *detected*, never *converted*. A ``$`` amount is parsed to its
  numeric magnitude and tagged ``USD``; callers that total INR exclude it and
  report it as a data-quality issue. We do not invent FX rates.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

__all__ = [
    "MISSING_TOKENS",
    "ParsedNumber",
    "is_missing",
    "parse_amount",
    "parse_quantity",
    "parse_percentage",
    "format_inr",
    "safe_ratio",
]

#: Case-insensitive strings that mean "no value was recorded".
MISSING_TOKENS = {
    "", "-", "--", "---", "n/a", "na", "n.a.", "n.a", "none", "null", "nil",
    "nan", "unknown", "not available", "not applicable", "tbd", "tba",
    "to be decided", "?", "#n/a", "#value!", "#ref!", "missing", "blank",
}

_BASE_CURRENCY = "INR"

_CURRENCY_SYMBOLS = {
    "₹": "INR", "rs.": "INR", "rs": "INR", "inr": "INR", "₨": "INR",
    "$": "USD", "usd": "USD", "us$": "USD",
    "€": "EUR", "eur": "EUR",
    "£": "GBP", "gbp": "GBP",
}

# Ordered longest-first so "crore" matches before "cr".
_MULTIPLIERS: list[tuple[str, float]] = [
    ("crores", 1e7), ("crore", 1e7), ("cr.", 1e7), ("cr", 1e7),
    ("lakhs", 1e5), ("lakh", 1e5), ("lacs", 1e5), ("lac", 1e5), ("l", 1e5),
    ("thousand", 1e3), ("k", 1e3),
    ("millions", 1e6), ("million", 1e6), ("mn", 1e6), ("m", 1e6),
    ("billions", 1e9), ("billion", 1e9), ("bn", 1e9), ("b", 1e9),
]

_NUMBER_RE = re.compile(r"[-+]?\d[\d,\s]*(?:\.\d+)?")


@dataclass(frozen=True)
class ParsedNumber:
    """Outcome of parsing a single messy numeric cell."""

    value: float | None
    currency: str | None = None
    #: ``None`` when parsing succeeded, otherwise a short machine-readable reason.
    reason: str | None = None
    raw: object = None

    @property
    def ok(self) -> bool:
        return self.value is not None

    @property
    def is_base_currency(self) -> bool:
        """True when the amount can be added to an INR total."""
        return self.ok and (self.currency in (None, _BASE_CURRENCY))


def is_missing(value: object) -> bool:
    """True for ``None``, ``NaN`` and every recognised 'no value' placeholder."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, (int, float)):
        return False
    text = str(value).strip().lower()
    if text in MISSING_TOKENS:
        return True
    # Purely punctuation cells such as "--" or "()".
    return not any(ch.isalnum() for ch in text)


def _detect_currency(text: str) -> tuple[str | None, str]:
    """Strip a leading/trailing currency marker, returning (currency, remainder)."""
    lowered = text.lower()
    for symbol, code in sorted(_CURRENCY_SYMBOLS.items(), key=lambda kv: -len(kv[0])):
        if symbol in lowered:
            # Only treat bare letters as currency when they stand apart from digits
            # ("l" in "25 L" is a multiplier, not a currency).
            if symbol.isalpha() and not re.search(rf"\b{re.escape(symbol)}\b", lowered):
                continue
            idx = lowered.index(symbol)
            remainder = text[:idx] + text[idx + len(symbol):]
            return code, remainder
    return None, text


def _apply_multiplier(remainder: str, number: float) -> tuple[float, str | None]:
    """Apply an Indian/western magnitude suffix found in ``remainder``."""
    tail = remainder.lower().strip(" .")
    if not tail:
        return number, None
    for token, factor in _MULTIPLIERS:
        if re.fullmatch(rf"{re.escape(token)}s?", tail):
            return number * factor, None
    return number, "unrecognised_unit"


def parse_amount(value: object, *, assume_currency: str | None = None) -> ParsedNumber:
    """Parse a monetary cell into a float magnitude plus a detected currency.

    Examples
    --------
    ``"₹2.5 Cr"`` -> ``25_000_000`` INR, ``"25 L"`` -> ``2_500_000`` INR,
    ``"2,500,000"`` -> ``2_500_000``, ``"$100000"`` -> ``100_000`` USD,
    ``"N/A"`` -> ``None`` with reason ``missing``.
    """
    if is_missing(value):
        return ParsedNumber(None, None, "missing", value)

    if isinstance(value, bool):
        return ParsedNumber(None, None, "unparseable", value)
    if isinstance(value, (int, float)):
        return ParsedNumber(float(value), assume_currency or _BASE_CURRENCY, None, value)

    text = str(value).strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]

    currency, remainder = _detect_currency(text)
    match = _NUMBER_RE.search(remainder)
    if not match:
        return ParsedNumber(None, currency, "unparseable", value)

    raw_number = match.group(0).replace(",", "").replace(" ", "")
    try:
        number = float(raw_number)
    except ValueError:
        return ParsedNumber(None, currency, "unparseable", value)

    suffix = remainder[match.end():]
    prefix = remainder[: match.start()]
    if prefix.strip(" .-+"):
        # Text before the number that is not a currency marker, e.g. "approx 5".
        # Tolerated, but only when it is alphabetic noise.
        if not prefix.strip(" .-+").replace(".", "").isalpha():
            return ParsedNumber(None, currency, "unparseable", value)

    number, unit_reason = _apply_multiplier(suffix, number)
    if unit_reason:
        return ParsedNumber(None, currency, unit_reason, value)

    if negative:
        number = -number
    if not math.isfinite(number):
        return ParsedNumber(None, currency, "unparseable", value)

    return ParsedNumber(number, currency or assume_currency or _BASE_CURRENCY, None, value)


#: Units that legitimately trail a *quantity* (as opposed to a currency amount).
_QUANTITY_UNITS = {
    "ha", "hectare", "hectares", "acre", "acres", "km", "kms", "sqkm", "sq km",
    "m", "mtr", "nos", "no", "units", "unit", "visits", "visit", "kw", "mw",
    "gw", "towers", "tower", "site", "sites", "months", "month", "days", "day",
}


def parse_quantity(value: object) -> ParsedNumber:
    """Parse a quantity cell such as ``"5360 HA"`` or ``"59.33"``.

    Unlike :func:`parse_amount`, a trailing physical unit is expected and is
    reported through :attr:`ParsedNumber.currency` (reused as a unit slot).
    """
    if is_missing(value):
        return ParsedNumber(None, None, "missing", value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return ParsedNumber(float(value), None, None, value)

    text = str(value).strip()
    match = _NUMBER_RE.search(text)
    if not match:
        return ParsedNumber(None, None, "unparseable", value)
    try:
        number = float(match.group(0).replace(",", "").replace(" ", ""))
    except ValueError:
        return ParsedNumber(None, None, "unparseable", value)

    unit = text[match.end():].strip().lower() or None
    if unit and unit not in _QUANTITY_UNITS:
        # Might still be a magnitude suffix ("2.5 Cr" used as a quantity).
        scaled, reason = _apply_multiplier(unit, number)
        if reason is None:
            return ParsedNumber(scaled, None, None, value)
        return ParsedNumber(number, unit, None, value)
    return ParsedNumber(number, unit, None, value)


def parse_percentage(value: object) -> ParsedNumber:
    """Parse ``"75%"`` / ``0.75`` / ``75`` into a 0..1 fraction."""
    if is_missing(value):
        return ParsedNumber(None, None, "missing", value)
    text = str(value).strip()
    has_symbol = "%" in text
    parsed = parse_amount(text.replace("%", ""))
    if not parsed.ok:
        return ParsedNumber(None, None, parsed.reason or "unparseable", value)
    number = float(parsed.value)
    if has_symbol or number > 1:
        number = number / 100.0
    if not 0 <= number <= 1:
        return ParsedNumber(None, None, "out_of_range", value)
    return ParsedNumber(number, None, None, value)


def format_inr(amount: float | None, *, precision: int = 2) -> str:
    """Format an INR magnitude the way an Indian founder reads it (Cr / L)."""
    if amount is None or (isinstance(amount, float) and math.isnan(amount)):
        return "n/a"
    sign = "-" if amount < 0 else ""
    magnitude = abs(float(amount))
    if magnitude >= 1e7:
        return f"{sign}₹{magnitude / 1e7:,.{precision}f} Cr"
    if magnitude >= 1e5:
        return f"{sign}₹{magnitude / 1e5:,.{precision}f} L"
    if magnitude >= 1e3:
        return f"{sign}₹{magnitude:,.0f}"
    return f"{sign}₹{magnitude:,.0f}"


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Division that returns ``None`` instead of raising or producing ``inf``."""
    if numerator is None or denominator in (None, 0):
        return None
    try:
        result = float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) else None
