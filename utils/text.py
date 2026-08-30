"""Text canonicalisation for sectors, statuses, stages and other free-text labels.

The original spreadsheet value is never destroyed: normalisation always produces
a *new* canonical field alongside the raw one (see
``analytics.normalization``). Matching is layered:

1. exact match on a slugified key,
2. alias table lookup (hand-built from the observed seed data),
3. substring/keyword rules,
4. fuzzy match against known canonical labels (``difflib``),
5. title-cased passthrough of the original -- an unknown sector stays visible
   rather than being silently bucketed into "Others".
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from utils.numbers import is_missing

__all__ = [
    "slugify",
    "title_case",
    "CanonicalValue",
    "canonical_sector",
    "canonical_deal_status",
    "canonical_deal_stage",
    "canonical_execution_status",
    "canonical_probability",
    "DEAL_STATUSES",
    "EXECUTION_STATUSES",
    "STAGE_GROUPS",
    "UNKNOWN",
]

UNKNOWN = "Unknown"

_TRAILING_NOISE = re.compile(
    r"\b(sector|sectors|service|services|industry|vertical|segment|domain)\b", re.IGNORECASE
)


def slugify(value: object) -> str:
    """Lowercase alphanumeric key used for dictionary lookups."""
    if is_missing(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def title_case(value: str) -> str:
    """Title-case a label while keeping common acronyms upper-case."""
    acronyms = {"dsp", "poc", "ar", "po", "loi", "loa", "gst", "wo", "rgb", "bd", "kam"}
    parts = []
    for word in re.split(r"(\s+|/|-)", str(value).strip()):
        if word.lower() in acronyms:
            parts.append(word.upper())
        elif word.strip() and word not in {"/", "-"}:
            parts.append(word[0].upper() + word[1:].lower() if len(word) > 1 else word.upper())
        else:
            parts.append(word)
    return "".join(parts)


@dataclass(frozen=True)
class CanonicalValue:
    """A normalised label plus provenance about how it was derived."""

    value: str
    raw: object = None
    #: ``exact`` | ``alias`` | ``keyword`` | ``fuzzy`` | ``passthrough`` | ``missing``
    method: str = "exact"

    @property
    def is_known(self) -> bool:
        return self.value != UNKNOWN

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.value


def _fuzzy(key: str, candidates: dict[str, str], cutoff: float = 0.86) -> str | None:
    matches = difflib.get_close_matches(key, list(candidates), n=1, cutoff=cutoff)
    return candidates[matches[0]] if matches else None


# --------------------------------------------------------------------------
# Sector
# --------------------------------------------------------------------------

#: Canonical sector labels observed across both boards, plus common synonyms.
SECTOR_ALIASES: dict[str, str] = {
    "mining": "Mining", "mines": "Mining", "mineral": "Mining", "coal": "Mining",
    "renewables": "Renewables", "renewable": "Renewables", "solar": "Renewables",
    "wind": "Renewables", "green_energy": "Renewables", "clean_energy": "Renewables",
    "energy": "Energy", "power": "Energy", "oil_and_gas": "Energy", "oil_gas": "Energy",
    "utilities": "Energy",
    "powerline": "Powerline", "power_line": "Powerline", "transmission": "Powerline",
    "t_and_d": "Powerline", "powerlines": "Powerline",
    "railways": "Railways", "railway": "Railways", "rail": "Railways",
    "construction": "Construction", "infra": "Infrastructure",
    "infrastructure": "Infrastructure", "roads": "Infrastructure", "highways": "Infrastructure",
    "manufacturing": "Manufacturing", "factory": "Manufacturing",
    "aviation": "Aviation", "airport": "Aviation", "airports": "Aviation",
    "security_and_surveillance": "Security and Surveillance",
    "security_surveillance": "Security and Surveillance",
    "security": "Security and Surveillance", "surveillance": "Security and Surveillance",
    "dsp": "DSP", "drone_service_provider": "DSP",
    "tender": "Tender", "tenders": "Tender", "govt_tender": "Tender",
    "agriculture": "Agriculture", "agri": "Agriculture",
    "telecom": "Telecom",
    "others": "Others", "other": "Others", "misc": "Others", "miscellaneous": "Others",
}

_SECTOR_CANONICALS = {slugify(v): v for v in set(SECTOR_ALIASES.values())}


def canonical_sector(value: object) -> CanonicalValue:
    """Map ``"energy sector"`` / ``"ENERGY"`` / ``"Energy"`` to one canonical label.

    Note: ``Renewables`` and ``Energy`` are kept distinct because the seed boards
    use ``Renewables`` as a first-class sector; conflating them would misstate
    sector totals. ``sector_matches`` in ``analytics.normalization`` handles the
    "energy" umbrella query separately.
    """
    if is_missing(value):
        return CanonicalValue(UNKNOWN, value, "missing")
    raw = str(value).strip()
    key = slugify(raw)
    if key in SECTOR_ALIASES:
        return CanonicalValue(SECTOR_ALIASES[key], value, "alias")
    stripped = slugify(_TRAILING_NOISE.sub("", raw))
    if stripped and stripped in SECTOR_ALIASES:
        return CanonicalValue(SECTOR_ALIASES[stripped], value, "alias")
    for token, canonical in SECTOR_ALIASES.items():
        if token and re.search(rf"(^|_){re.escape(token)}(_|$)", stripped or key):
            return CanonicalValue(canonical, value, "keyword")
    guess = _fuzzy(stripped or key, SECTOR_ALIASES)
    if guess:
        return CanonicalValue(guess, value, "fuzzy")
    # Unknown sectors are preserved (title-cased) rather than dumped into Others.
    return CanonicalValue(title_case(raw), value, "passthrough")


#: Sector labels an "energy" style umbrella question should also cover.
SECTOR_UMBRELLAS: dict[str, tuple[str, ...]] = {
    "Energy": ("Energy", "Renewables", "Powerline"),
    "Infrastructure": ("Infrastructure", "Construction", "Railways"),
}


# --------------------------------------------------------------------------
# Deal status
# --------------------------------------------------------------------------

DEAL_STATUSES = ("Open", "Won", "Lost", "On Hold", UNKNOWN)

DEAL_STATUS_ALIASES: dict[str, str] = {
    "open": "Open", "active": "Open", "in_progress": "Open", "in_pipeline": "Open",
    "live": "Open", "ongoing": "Open", "pending": "Open", "new": "Open",
    "won": "Won", "closed_won": "Won", "close_won": "Won", "closedwon": "Won",
    "win": "Won", "project_won": "Won", "converted": "Won", "success": "Won",
    "lost": "Lost", "closed_lost": "Lost", "close_lost": "Lost", "dead": "Lost",
    "closedlost": "Lost", "project_lost": "Lost", "cancelled": "Lost",
    "canceled": "Lost", "dropped": "Lost", "not_relevant": "Lost",
    "disqualified": "Lost", "churned": "Lost", "rejected": "Lost",
    "on_hold": "On Hold", "hold": "On Hold", "onhold": "On Hold",
    "paused": "On Hold", "stalled": "On Hold", "deferred": "On Hold",
}


def canonical_deal_status(value: object) -> CanonicalValue:
    if is_missing(value):
        return CanonicalValue(UNKNOWN, value, "missing")
    key = slugify(value)
    if key in DEAL_STATUS_ALIASES:
        return CanonicalValue(DEAL_STATUS_ALIASES[key], value, "alias")
    for token, canonical in DEAL_STATUS_ALIASES.items():
        if re.search(rf"(^|_){re.escape(token)}(_|$)", key):
            return CanonicalValue(canonical, value, "keyword")
    guess = _fuzzy(key, DEAL_STATUS_ALIASES)
    if guess:
        return CanonicalValue(guess, value, "fuzzy")
    return CanonicalValue(UNKNOWN, value, "passthrough")


# --------------------------------------------------------------------------
# Deal stage
# --------------------------------------------------------------------------

#: Funnel groups used by the analytics layer. ``order`` drives "late stage".
STAGE_GROUPS: dict[str, dict] = {
    "early": {"order": 1, "label": "Early stage"},
    "mid": {"order": 2, "label": "Mid stage"},
    "late": {"order": 3, "label": "Late stage"},
    "won": {"order": 4, "label": "Won / in execution"},
    "lost": {"order": 5, "label": "Lost"},
    "on_hold": {"order": 6, "label": "On hold"},
    "unknown": {"order": 7, "label": "Unknown stage"},
}

# The seed board prefixes stages with an ordering letter ("E. Proposal ...").
_STAGE_LETTER_GROUPS = {
    "a": "early", "b": "early", "c": "mid", "d": "mid",
    "e": "late", "f": "late",
    "g": "won", "h": "won", "i": "won", "j": "won", "k": "won",
    "l": "lost", "m": "on_hold", "n": "lost", "o": "lost",
}

_STAGE_KEYWORD_GROUPS = [
    ("lead", "early"), ("prospect", "early"), ("qualified", "early"),
    ("discovery", "early"), ("demo", "mid"), ("feasibility", "mid"),
    ("poc", "won"), ("proof_of_concept", "mid"),
    ("proposal", "late"), ("commercial", "late"), ("quote", "late"),
    ("negotiat", "late"),
    ("won", "won"), ("work_order", "won"), ("invoice", "won"),
    ("accrued", "won"), ("completed", "won"), ("delivered", "won"),
    ("lost", "lost"), ("not_relevant", "lost"), ("dead", "lost"),
    ("hold", "on_hold"),
]


@dataclass(frozen=True)
class CanonicalStage:
    label: str
    group: str
    order: float
    raw: object = None

    @property
    def is_known(self) -> bool:
        return self.group != "unknown"


def canonical_deal_stage(value: object) -> CanonicalStage:
    """Normalise a funnel stage, retaining its label and deriving a funnel group."""
    if is_missing(value):
        return CanonicalStage(UNKNOWN, "unknown", 99.0, value)
    raw = str(value).strip()
    letter_match = re.match(r"^\s*([A-Za-z])[\.\)]\s*(.+)$", raw)
    if letter_match:
        letter = letter_match.group(1).lower()
        label = title_case(letter_match.group(2).strip())
        group = _STAGE_LETTER_GROUPS.get(letter, "unknown")
        order = float(ord(letter) - ord("a") + 1)
        if group == "unknown":
            group = _stage_group_from_keywords(slugify(label))
        return CanonicalStage(label, group, order, value)

    key = slugify(raw)
    group = _stage_group_from_keywords(key)
    return CanonicalStage(title_case(raw), group, float(STAGE_GROUPS[group]["order"]) + 10, value)


def _stage_group_from_keywords(key: str) -> str:
    for token, group in _STAGE_KEYWORD_GROUPS:
        if token in key:
            return group
    return "unknown"


# --------------------------------------------------------------------------
# Work-order execution status
# --------------------------------------------------------------------------

EXECUTION_STATUSES = ("Not Started", "In Progress", "Completed", "Blocked", UNKNOWN)

EXECUTION_STATUS_ALIASES: dict[str, str] = {
    "not_started": "Not Started", "yet_to_start": "Not Started", "new": "Not Started",
    "planned": "Not Started", "scheduled": "Not Started",
    "ongoing": "In Progress", "in_progress": "In Progress", "wip": "In Progress",
    "executing": "In Progress", "started": "In Progress", "active": "In Progress",
    "partial_completed": "In Progress", "partially_completed": "In Progress",
    "executed_until_current_month": "In Progress", "running": "In Progress",
    "completed": "Completed", "complete": "Completed", "done": "Completed",
    "closed": "Completed", "delivered": "Completed", "finished": "Completed",
    "pause_struck": "Blocked", "paused": "Blocked", "on_hold": "Blocked",
    "hold": "Blocked", "stuck": "Blocked", "blocked": "Blocked",
    "details_pending_from_client": "Blocked", "pending_from_client": "Blocked",
    "cancelled": "Blocked", "dropped": "Blocked",
}

#: Statuses that still consume operational capacity.
ACTIVE_EXECUTION_STATUSES = ("Not Started", "In Progress")


def canonical_execution_status(value: object) -> CanonicalValue:
    if is_missing(value):
        return CanonicalValue(UNKNOWN, value, "missing")
    key = slugify(value)
    if key in EXECUTION_STATUS_ALIASES:
        return CanonicalValue(EXECUTION_STATUS_ALIASES[key], value, "alias")
    for token, canonical in EXECUTION_STATUS_ALIASES.items():
        if re.search(rf"(^|_){re.escape(token)}(_|$)", key):
            return CanonicalValue(canonical, value, "keyword")
    guess = _fuzzy(key, EXECUTION_STATUS_ALIASES)
    if guess:
        return CanonicalValue(guess, value, "fuzzy")
    return CanonicalValue(UNKNOWN, value, "passthrough")


# --------------------------------------------------------------------------
# Closure probability
# --------------------------------------------------------------------------

#: Weights applied to categorical closure probability when computing weighted
#: pipeline. Documented as an explicit assumption in DECISION_LOG.md.
PROBABILITY_BANDS: dict[str, float] = {"High": 0.75, "Medium": 0.45, "Low": 0.20}

_PROBABILITY_ALIASES = {
    "high": "High", "h": "High", "hot": "High", "very_high": "High",
    "medium": "Medium", "med": "Medium", "m": "Medium", "moderate": "Medium",
    "warm": "Medium", "mid": "Medium",
    "low": "Low", "l": "Low", "cold": "Low", "very_low": "Low",
}


def canonical_probability(value: object) -> CanonicalValue:
    """Normalise ``High``/``Medium``/``Low``; numeric percentages map to a band."""
    if is_missing(value):
        return CanonicalValue(UNKNOWN, value, "missing")
    key = slugify(value)
    if key in _PROBABILITY_ALIASES:
        return CanonicalValue(_PROBABILITY_ALIASES[key], value, "alias")

    from utils.numbers import parse_percentage  # local import avoids a cycle

    pct = parse_percentage(value)
    if pct.ok:
        band = "High" if pct.value >= 0.6 else "Medium" if pct.value >= 0.3 else "Low"
        return CanonicalValue(band, value, "numeric")
    guess = _fuzzy(key, _PROBABILITY_ALIASES)
    if guess:
        return CanonicalValue(guess, value, "fuzzy")
    return CanonicalValue(UNKNOWN, value, "passthrough")
