"""Natural language -> :class:`QueryPlan`.

Primary path: Groq in JSON mode, validated with Pydantic.
Fallback path: a deterministic keyword classifier that keeps the whole app
usable when Groq is unavailable, misconfigured or returns unusable JSON.
"""
from __future__ import annotations

import re

import pandas as pd

from agent.llm import GroqLLM, LLMError, extract_json
from agent.prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_TEMPLATE
from agent.schemas import ALL_INTENTS, QueryPlan
from utils.logging import get_logger
from utils.text import SECTOR_ALIASES, canonical_sector, slugify

logger = get_logger(__name__)

__all__ = ["QueryPlanner", "heuristic_plan"]


# --- deterministic fallback -------------------------------------------------

_INTENT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("leadership_update", ("leadership update", "executive summary", "exec summary",
                           "board update", "leadership brief", "management update",
                           "board deck", "briefing")),
    ("cross_board_analysis", ("compare pipeline", "pipeline vs", "pipeline versus",
                              "sales vs", "vs operational", "vs operations",
                              "capacity", "workload compared", "compare sales",
                              "both boards", "sales and delivery", "sales and operations")),
    ("data_quality", ("data quality", "data issues", "missing data", "how clean",
                      "incomplete records", "data gaps")),
    ("work_order_analysis", ("work order", "work orders", "wo ", "project delayed",
                             "delayed project", "delays", "delayed", "execution",
                             "delivery", "operations", "operational", "projects")),
    ("sales_rep_analysis", ("sales rep", "owner", "by rep", "which rep", "account manager",
                            "salesperson", "bd ")),
    ("revenue_analysis", ("revenue", "closed won", "won value", "billing", "billed",
                          "collections", "collected", "receivable", "invoice")),
    ("sector_analysis", ("sector", "vertical", "industry", "segment")),
    ("deal_analysis", ("biggest deal", "largest deal", "top deal", "at risk", "risky",
                       "which deals", "deal ", "opportunit")),
    ("pipeline_analysis", ("pipeline", "funnel", "forecast", "expected close",
                           "weighted", "open deals")),
    ("operational_health", ("operational health", "how is delivery", "delivery health")),
]

#: Greetings and "what can you do" openers. Matched before anything else so a
#: "hi" never costs an LLM round-trip or a board fetch.
_GREETINGS = (
    "hi", "hii", "hiii", "hey", "heya", "hello", "helo", "yo", "sup", "hola",
    "namaste", "good morning", "good afternoon", "good evening", "gm", "greetings",
    "thanks", "thank you", "thx", "ty", "cheers", "ok thanks", "okay thanks",
    "bye", "goodbye", "see you",
)

_CAPABILITY_QUESTIONS = (
    "what can you do", "what can you help", "who are you", "what are you",
    "how do you work", "what do you do", "help", "how can you help",
    "what questions", "what should i ask", "how does this work",
    "what data do you have", "capabilities",
)


def _is_greeting(text: str) -> bool:
    """True for a bare greeting or a capability question.

    Deliberately conservative: the greeting must be essentially the whole message,
    so "hi, how is the mining pipeline?" is treated as the real question it is.
    """
    cleaned = text.strip().strip("!?.,").lower()
    if not cleaned:
        return False
    if cleaned in _GREETINGS:
        return True
    if any(cleaned.startswith(g) and len(cleaned.split()) <= 3 for g in _GREETINGS):
        return True
    return any(phrase in cleaned for phrase in _CAPABILITY_QUESTIONS)


#: Vocabulary that marks a question as being about this business data. Used only
#: by the fallback planner, to avoid answering an unrelated question with a
#: business summary when the LLM is unavailable to classify it.
_DOMAIN_VOCABULARY = frozenset("""
pipeline funnel deal deals opportunity opportunities revenue sales sold sell
sector sectors vertical verticals industry segment client clients customer
customers account accounts stage stages won win wins winning lost lose losing
close closing closed forecast quarter quarterly month monthly year fy fiscal
value values amount amounts crore cr lakh lakhs rupee rupees inr money
work workorder workorders order orders project projects delivery deliver
delivered delayed delay delays late overdue execution executing operational
operations ops workload capacity billing billed invoice invoiced collection
collections receivable owner owners rep reps kam bd risk risky risks
performance performing health summary update briefing leadership executive
business company data quality missing status probability weighted stalled
active open ongoing completed blocked
""".split())


def _mentions_the_business(text: str, history: list[dict] | None) -> bool:
    """True when a question plausibly concerns the Deals / Work Orders data.

    Used only in the fallback planner. Deliberately generous: a false positive
    costs a slightly odd answer, whereas a false negative refuses a legitimate
    question.
    """
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    if tokens & _DOMAIN_VOCABULARY:
        return True
    # Sector names and owner codes count as domain vocabulary.
    if _detect_sector(text) or _detect_owner(text):
        return True
    # A short follow-up inherits the previous turn's subject.
    return bool(history and len(text.split()) <= 6)


_VAGUE_QUESTIONS = (
    "how are we doing", "how's it going", "hows it going", "how are things",
    "give me an update", "what's up", "whats up", "how is business",
    "how's business", "hows business", "status", "overview",
)

_DATE_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("current_quarter", ("this quarter", "current quarter", "the quarter", "qtr")),
    ("next_quarter", ("next quarter", "coming quarter", "upcoming quarter")),
    ("last_quarter", ("last quarter", "previous quarter", "prior quarter")),
    ("current_month", ("this month", "current month")),
    ("last_month", ("last month", "previous month")),
    ("current_fy", ("this fiscal year", "this financial year", "current fy", "this fy",
                    "fiscal year")),
    ("last_fy", ("last fiscal year", "last financial year", "last fy")),
    ("ytd", ("year to date", "ytd")),
    ("next_30_days", ("next 30 days", "next month", "coming 30 days")),
    ("next_90_days", ("next 90 days", "next three months", "next 3 months")),
    ("last_90_days", ("last 90 days", "past 90 days", "last three months")),
    ("last_30_days", ("last 30 days", "past 30 days")),
    ("current_year", ("this year", "calendar year")),
    ("overdue", ("overdue", "past due", "behind schedule")),
]


def _detect_sector(question: str) -> str | None:
    """Find a sector mentioned in the question, tolerating plurals and casing."""
    key = f" {slugify(question)} "
    best: tuple[int, str] | None = None
    for alias in sorted(SECTOR_ALIASES, key=len, reverse=True):
        if not alias or len(alias) < 3:
            continue
        if re.search(rf"[_ ]{re.escape(alias)}s?[_ ]", key):
            if best is None or len(alias) > best[0]:
                best = (len(alias), alias)
    return canonical_sector(best[1]).value if best else None


def _detect_owner(question: str) -> str | None:
    match = re.search(r"\bowner[_ ]?0*(\d+)\b", question, re.IGNORECASE)
    if match:
        return f"OWNER_{int(match.group(1)):03d}"
    return None


# --- replying to a clarification -------------------------------------------
#
# When the agent offers numbered options, the natural reply is "1" -- which on
# its own means nothing to a planner. Resolving the selection here, before any
# model call, is both deterministic and the only way the keyword planner can
# honour a choice at all.

_ORDINAL_WORDS: dict[str, int] = {
    "first": 1, "1st": 1, "one": 1,
    "second": 2, "2nd": 2, "two": 2,
    "third": 3, "3rd": 3, "three": 3,
    "fourth": 4, "4th": 4, "four": 4,
    "fifth": 5, "5th": 5, "five": 5,
    "last": -1,
}

_SELECTION_RE = re.compile(r"^[#(\[]?\s*(\d{1,2})\s*[.)\]]?$")
_SELECTION_WORDED_RE = re.compile(r"^(?:option|number|no\.?|choice|answer)\s*(\d{1,2})$")


def offered_options(history: list[dict] | None) -> list[str]:
    """The clarification options offered on the immediately preceding turn.

    Only the last turn counts: once the user has moved on, a stale clarification
    must not hijack an unrelated question.
    """
    if not history:
        return []
    plan = history[-1].get("plan") or {}
    if not plan.get("needs_clarification"):
        return []
    return [str(o).strip() for o in (plan.get("clarification_options") or []) if str(o).strip()]


def resolve_clarification_reply(question: str, history: list[dict] | None) -> str | None:
    """Map a reply such as ``"1"``, ``"the second one"`` or ``"revenue"`` onto the
    option it selects. Returns ``None`` when the reply is not a selection.
    """
    options = offered_options(history)
    if not options:
        return None
    text = (question or "").strip().lower().strip(".)!?,")
    if not text:
        return None

    match = _SELECTION_RE.match(text) or _SELECTION_WORDED_RE.match(text)
    if match:
        index = int(match.group(1))
        return options[index - 1] if 1 <= index <= len(options) else None

    tokens = re.findall(r"[a-z0-9]+", text)
    if len(tokens) <= 4:
        for token in tokens:
            index = _ORDINAL_WORDS.get(token)
            if index == -1:
                return options[-1]
            if index and index <= len(options):
                return options[index - 1]

    # The option's own wording, or an unambiguous part of it ("revenue").
    for option in options:
        if text == option.lower():
            return option
    if len(text) >= 4:
        partial = [o for o in options if text in o.lower()]
        if len(partial) == 1:
            return partial[0]
    return None


#: The four angles offered when a question could mean several things.
STANDARD_CLARIFICATION_OPTIONS = [
    "Sales / pipeline",
    "Revenue (won and billed)",
    "Operations / work orders",
    "Overall business health",
]


def _is_bare_selection(text: str) -> bool:
    """True for a lone ``"2"`` / ``"third"`` -- a choice with nothing to choose from."""
    cleaned = (text or "").strip().lower().strip(".)!?,")
    if not cleaned:
        return False
    if _SELECTION_RE.match(cleaned) or _SELECTION_WORDED_RE.match(cleaned):
        return True
    return cleaned in _ORDINAL_WORDS


def _ask_what_they_meant(reason: str, source: str) -> QueryPlan:
    return QueryPlan(
        intent="general_business_summary",
        boards=["deals", "work_orders"],
        needs_clarification=True,
        clarification_question=(
            "I'm not sure what that refers to — which of these would you like?"
        ),
        clarification_options=list(STANDARD_CLARIFICATION_OPTIONS),
        reasoning=reason,
        source=source,
    )


def heuristic_plan(question: str, history: list[dict] | None = None) -> QueryPlan:
    """Keyword-based planner used when Groq is unavailable or returns bad JSON."""
    text = (question or "").strip()
    # "1" in reply to a numbered clarification means the first option.
    selected = resolve_clarification_reply(text, history)
    if selected:
        text = selected
    elif _is_bare_selection(text):
        # A number with no pending question: answering the previous topic again
        # would look like the agent ignored them.
        return _ask_what_they_meant("A bare selection with no options pending.", "fallback")
    lowered = text.lower()

    if _is_greeting(text):
        return QueryPlan(
            intent="greeting", boards=["deals"],
            reasoning="Greeting or capability question; no data lookup required.",
            source="fallback",
        )

    intent = "general_business_summary"
    for candidate, keywords in _INTENT_KEYWORDS:
        if any(k in lowered for k in keywords):
            intent = candidate
            break

    date_range = "all_time"
    for token, keywords in _DATE_PATTERNS:
        if any(k in lowered for k in keywords):
            date_range = token
            break

    sector = _detect_sector(lowered)
    owner = _detect_owner(lowered)

    status_filter = None
    if intent in {"pipeline_analysis", "sector_analysis", "deal_analysis"}:
        status_filter = "open"
    if "won" in lowered or "closed won" in lowered:
        status_filter = "won"
    if "lost" in lowered or "dead deal" in lowered:
        status_filter = "lost"
    if intent in {"work_order_analysis", "operational_health"} and any(
        k in lowered for k in ("active", "ongoing", "in progress")
    ):
        status_filter = "active"

    group_by = "sector"
    if "by stage" in lowered or "stage" in lowered:
        group_by = "stage_group"
    if "by owner" in lowered or "by rep" in lowered or intent == "sales_rep_analysis":
        group_by = "owner"
    if intent in {"work_order_analysis", "operational_health"} and "status" in lowered:
        group_by = "execution_status"

    # A follow-up such as "what about infrastructure?" inherits the previous intent.
    if history and len(lowered.split()) <= 6 and intent == "general_business_summary":
        for turn in reversed(history):
            previous = turn.get("plan")
            if previous:
                intent = previous.get("intent", intent)
                if not sector:
                    sector = previous.get("sector")
                if date_range == "all_time":
                    date_range = previous.get("date_range", "all_time")
                status_filter = status_filter or previous.get("status_filter")
                break

    # Nothing in the question relates to this data, and it is not a vague
    # "how are we doing?" opener -> decline rather than answer with a summary.
    if not _mentions_the_business(text, history) and not any(
        v in lowered for v in _VAGUE_QUESTIONS
    ):
        return QueryPlan(
            intent="out_of_scope", boards=["deals"],
            reasoning="No reference to the Deals or Work Orders data.",
            source="fallback",
        )

    needs_clarification = False
    clarification_question = None
    options: list[str] = []
    if (
        intent == "general_business_summary"
        and not sector
        and any(v in lowered for v in _VAGUE_QUESTIONS)
        and len(lowered.split()) <= 8
        # Never ask twice in a row: the previous turn already offered options, so
        # this turn commits to an answer whatever the reply looked like.
        and not offered_options(history)
    ):
        needs_clarification = True
        clarification_question = "I can analyse that from a few angles. Which would you like?"
        options = [
            "Sales / pipeline",
            "Revenue (won and billed)",
            "Operations / work orders",
            "Overall business health",
        ]

    boards = ["deals"]
    if intent in {"work_order_analysis", "operational_health"}:
        boards = ["work_orders"]
    elif intent in {"cross_board_analysis", "leadership_update", "data_quality",
                    "general_business_summary", "sector_analysis"}:
        boards = ["deals", "work_orders"]

    plan = QueryPlan(
        intent=intent,
        boards=boards,
        metric=None,
        sector=sector,
        owner=owner,
        date_range=date_range,
        status_filter=status_filter,
        group_by=group_by,
        requires_cross_board=intent in {"cross_board_analysis", "leadership_update"},
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
        clarification_options=options,
        reasoning="Classified by keyword rules (LLM planner unavailable).",
        source="fallback",
    )
    return plan.with_defaults()


# --- LLM planner ------------------------------------------------------------


class QueryPlanner:
    """Converts a founder question into a validated :class:`QueryPlan`."""

    def __init__(self, llm: GroqLLM | None = None, *, fy_start_month: int = 4) -> None:
        self.llm = llm if llm is not None else GroqLLM()
        self.fy_start_month = fy_start_month

    def plan(
        self,
        question: str,
        history: list[dict] | None = None,
        *,
        today: pd.Timestamp | None = None,
    ) -> QueryPlan:
        """Plan a question, falling back to heuristics on any LLM problem."""
        if not question or not question.strip():
            return heuristic_plan("", history)

        # A reply to a numbered clarification is resolved to the option it picks
        # before the model sees it, so "1" carries its meaning rather than none.
        selected = resolve_clarification_reply(question, history)
        answering_clarification = bool(offered_options(history))
        if selected:
            question = selected
        elif _is_bare_selection(question):
            return _ask_what_they_meant("A bare selection with no options pending.", "fallback")

        # A greeting needs no model call and no board fetch.
        if _is_greeting(question):
            return heuristic_plan(question, history)

        if not self.llm.available:
            logger.info("Groq unavailable; using the heuristic planner")
            return heuristic_plan(question, history)

        prompt = PLANNER_USER_TEMPLATE.format(
            history=_format_history(history),
            question=question.strip(),
            today=(today or pd.Timestamp.today()).date().isoformat(),
            fy_start_month=self.fy_start_month,
        )
        try:
            response = self.llm.complete(
                PLANNER_SYSTEM_PROMPT, prompt, json_mode=True, temperature=0.0, max_tokens=1200
            )
        except LLMError as exc:
            logger.warning("Planner LLM call failed (%s); falling back to heuristics", exc)
            return heuristic_plan(question, history)

        payload = extract_json(response.text)
        if not payload:
            logger.warning("Planner returned unparseable JSON; falling back to heuristics")
            return heuristic_plan(question, history)

        try:
            plan = QueryPlan(**payload)
        except Exception as exc:  # noqa: BLE001 - pydantic ValidationError family
            logger.warning("Planner JSON failed validation (%s); repairing with heuristics", exc)
            repaired = heuristic_plan(question, history)
            merged = repaired.model_dump()
            for key in ("intent", "sector", "owner", "date_range", "status_filter", "group_by"):
                value = payload.get(key)
                if not value:
                    continue
                # Salvage only the fields that are individually valid; an invalid
                # intent must not discard a correctly extracted sector.
                if key == "intent" and value not in ALL_INTENTS:
                    continue
                merged[key] = value
            try:
                plan = QueryPlan(**merged)
                plan = plan.model_copy(update={"source": "llm_repaired"})
            except Exception:  # noqa: BLE001 - give up and use the clean fallback
                return repaired

        plan = plan.model_copy(update={"source": plan.source or "llm"})
        if answering_clarification and plan.needs_clarification:
            # The model asked again. One clarification is a helpful question;
            # two in a row is a loop, so this turn answers with what it has.
            plan = plan.model_copy(update={
                "needs_clarification": False, "clarification_options": [],
            })
        return plan.with_defaults()


def _format_history(history: list[dict] | None, limit: int = 4) -> str:
    """Render recent turns compactly so follow-ups resolve correctly."""
    if not history:
        return "(no previous turns)"
    lines = []
    for turn in history[-limit:]:
        question = turn.get("question")
        if question:
            lines.append(f"User: {question}")
        plan = turn.get("plan")
        if plan:
            lines.append(
                f"Interpreted as: intent={plan.get('intent')}, sector={plan.get('sector')}, "
                f"period={plan.get('date_range')}"
            )
            # Without this, a reply of "1" is unresolvable and the model can only
            # ask again -- which is exactly how a clarification loop starts.
            options = plan.get("clarification_options") or []
            if plan.get("needs_clarification") and options:
                offered = "; ".join(f"{i + 1}. {o}" for i, o in enumerate(options))
                lines.append(f"Assistant asked the user to choose between: {offered}")
    return "\n".join(lines) or "(no previous turns)"
