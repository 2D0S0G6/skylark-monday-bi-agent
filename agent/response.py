"""Turn computed facts into a founder-facing answer.

Primary path: Groq narrates the pre-computed facts.
Fallback path: a deterministic markdown renderer, so a Groq outage still yields
a correct (if plainer) answer built from exactly the same numbers.
"""
from __future__ import annotations

import json

import pandas as pd

from agent.llm import GroqLLM, LLMError
from agent.prompts import (
    LEADERSHIP_SYSTEM_PROMPT,
    LEADERSHIP_USER_TEMPLATE,
    NARRATOR_SYSTEM_PROMPT,
    NARRATOR_USER_TEMPLATE,
)
from agent.schemas import QueryPlan
from utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["ResponseWriter", "render_facts_markdown"]

#: Facts are truncated before being sent to the model so a huge board cannot
#: blow the context window.
_MAX_FACTS_CHARS = 14000


def _plan_summary(plan: QueryPlan) -> str:
    bits = [f"intent={plan.intent}"]
    if plan.sector:
        bits.append(f"sector={plan.sector}")
    if plan.owner:
        bits.append(f"owner={plan.owner}")
    if plan.date_range and plan.date_range != "all_time":
        bits.append(f"period={plan.date_range}")
    if plan.status_filter:
        bits.append(f"status={plan.status_filter}")
    if plan.group_by and plan.group_by != "none":
        bits.append(f"grouped by {plan.group_by}")
    return ", ".join(bits)


def _serialise(facts: dict) -> str:
    text = json.dumps(facts, indent=2, default=str, ensure_ascii=False)
    if len(text) <= _MAX_FACTS_CHARS:
        return text
    # Drop the bulkiest list sections first rather than truncating mid-JSON.
    trimmed = json.loads(json.dumps(facts, default=str))
    for section in ("deals", "work_orders"):
        block = trimmed.get(section)
        if isinstance(block, dict):
            for key in ("top_opportunities", "delayed", "upcoming_completions"):
                if isinstance(block.get(key), dict) and isinstance(block[key].get("rows"), list):
                    block[key]["rows"] = block[key]["rows"][:3]
    text = json.dumps(trimmed, indent=2, default=str, ensure_ascii=False)
    return text[:_MAX_FACTS_CHARS]


class ResponseWriter:
    """Narrates computed facts. Never introduces a number of its own."""

    def __init__(self, llm: GroqLLM | None = None) -> None:
        self.llm = llm if llm is not None else GroqLLM()

    def write(self, question: str, plan: QueryPlan, facts: dict) -> tuple[str, str]:
        """Return ``(markdown_answer, source)`` where source is ``llm`` or ``deterministic``."""
        if not self.llm.available:
            return render_facts_markdown(plan, facts), "deterministic"

        prompt = NARRATOR_USER_TEMPLATE.format(
            question=question,
            plan_summary=_plan_summary(plan),
            facts=_serialise(facts),
        )
        try:
            response = self.llm.complete(
                NARRATOR_SYSTEM_PROMPT, prompt, temperature=0.25, max_tokens=2000
            )
        except LLMError as exc:
            logger.warning("Narration failed (%s); using the deterministic renderer", exc)
            return render_facts_markdown(plan, facts), "deterministic"

        text = (response.text or "").strip()
        if len(text) < 40:
            return render_facts_markdown(plan, facts), "deterministic"
        return text, "llm"

    def write_leadership_update(
        self, facts: dict, *, period: str, today: pd.Timestamp | None = None
    ) -> tuple[str, str]:
        if not self.llm.available:
            return render_leadership_markdown(facts, period=period), "deterministic"

        prompt = LEADERSHIP_USER_TEMPLATE.format(
            period=period,
            today=(today or pd.Timestamp.today()).date().isoformat(),
            facts=_serialise(facts),
        )
        try:
            response = self.llm.complete(
                LEADERSHIP_SYSTEM_PROMPT, prompt, temperature=0.25, max_tokens=2500
            )
        except LLMError as exc:
            logger.warning("Leadership narration failed (%s); using the renderer", exc)
            return render_leadership_markdown(facts, period=period), "deterministic"

        text = (response.text or "").strip()
        if len(text) < 80:
            return render_leadership_markdown(facts, period=period), "deterministic"
        return text, "llm"


# ---------------------------------------------------------------------------
# deterministic renderers (used when Groq is unavailable)
# ---------------------------------------------------------------------------

def _display(money: dict | None) -> str:
    if not money:
        return "not available"
    return money.get("display", "not available")


def _deal_lines(deals: dict) -> list[str]:
    lines: list[str] = []
    summary = deals.get("summary", {})
    scope = deals.get("scope", {})
    if summary.get("deal_count") == 0:
        return ["- No deals match this scope."]

    if summary.get("open_pipeline_value"):
        lines.append(
            f"- **{_display(summary['open_pipeline_value'])}** open pipeline across "
            f"**{summary.get('open_deal_count', 0)}** open deals."
        )
    else:
        lines.append(f"- **{summary.get('open_deal_count', 0)}** open deals in scope.")

    if summary.get("late_stage_open_value"):
        share = summary.get("late_stage_share_of_open_pct")
        share_text = f" ({share}% of open pipeline)" if share is not None else ""
        lines.append(
            f"- **{_display(summary['late_stage_open_value'])}** sits in late-stage deals"
            f"{share_text}."
        )
    if summary.get("weighted_open_pipeline_value"):
        lines.append(
            f"- Probability-weighted open pipeline: "
            f"**{_display(summary['weighted_open_pipeline_value'])}**."
        )
    if summary.get("won_value"):
        lines.append(
            f"- Closed-won value in scope: **{_display(summary['won_value'])}** "
            f"across {summary.get('won_deal_count', 0)} deals."
        )

    breakdown = deals.get("breakdown") or {}
    for row in (breakdown.get("rows") or [])[:4]:
        label = row.get(breakdown.get("dimension", "sector"), "Unknown")
        value = _display(row.get("value"))
        share = row.get("value_share_pct")
        share_text = f" — {share}% of the total" if share is not None else ""
        lines.append(f"- {label}: **{value}** across {row.get('deal_count', 0)} deals{share_text}.")

    if scope.get("period") and scope["period"] != "all time":
        lines.append(f"- Period applied: {scope['period']}.")
    return lines


def _wo_lines(work_orders: dict) -> list[str]:
    lines: list[str] = []
    summary = work_orders.get("summary", {})
    if summary.get("work_order_count") == 0:
        return ["- No work orders match this scope."]

    lines.append(
        f"- **{summary.get('active_work_orders', 0)}** active of "
        f"**{summary.get('work_order_count', 0)}** work orders in scope."
    )
    if summary.get("active_order_value"):
        lines.append(f"- Active order value: **{_display(summary['active_order_value'])}**.")
    delayed = work_orders.get("delayed", {})
    if delayed.get("delayed_count"):
        share = delayed.get("delayed_share_pct")
        share_text = f" ({share}% of those that can be assessed)" if share is not None else ""
        lines.append(f"- **{delayed['delayed_count']}** work orders are past their planned end date{share_text}.")
        if delayed.get("average_days_overdue") is not None:
            lines.append(f"- Average delay: **{delayed['average_days_overdue']} days**.")
    return lines


def _quality_lines(facts: dict, limit: int = 3) -> list[str]:
    issues: list[dict] = []
    for block in ("deals", "work_orders", "cross_board"):
        section = facts.get(block) or {}
        dq = section.get("data_quality") or {}
        issues.extend(dq.get("issues") or [])
    dq = facts.get("data_quality") or {}
    issues.extend(dq.get("issues") or [])

    ranked = sorted(
        issues,
        key=lambda i: ({"excluded": 0, "included_with_gap": 1, "info": 2}.get(i.get("severity"), 3),
                       -i.get("count", 0)),
    )
    seen, lines = set(), []
    for issue in ranked:
        code = issue.get("code")
        if code in seen:
            continue
        seen.add(code)
        lines.append(f"- {issue.get('message')}")
        if len(lines) >= limit:
            break
    return lines


def render_facts_markdown(plan: QueryPlan, facts: dict) -> str:
    """Deterministic fallback answer, built only from the computed facts."""
    parts: list[str] = [f"### {plan.intent.replace('_', ' ').title()}"]

    if facts.get("deals"):
        parts.append("**Pipeline**")
        parts.extend(_deal_lines(facts["deals"]))
    at_risk = (facts.get("deals") or {}).get("at_risk") or {}
    if at_risk.get("at_risk_count"):
        parts.append("")
        parts.append(
            f"**Deals carrying a risk signal: {at_risk['at_risk_count']} of "
            f"{at_risk.get('deals_considered', 0)}**"
        )
        for row in (at_risk.get("rows") or [])[:4]:
            parts.append(
                f"- {row['deal']} ({row.get('sector') or 'sector unknown'}), "
                f"{_display(row.get('value'))} — {'; '.join(row.get('risk_signals', []))}"
            )

    if facts.get("work_orders"):
        parts.append("")
        parts.append("**Operations**")
        parts.extend(_wo_lines(facts["work_orders"]))

    cross = facts.get("cross_board") or {}
    if cross.get("available"):
        signals = cross.get("signals", {})
        ahead = signals.get("pipeline_ahead_of_delivery") or []
        behind = signals.get("delivery_ahead_of_pipeline") or []
        parts.append("")
        parts.append("**Sales vs delivery balance (by sector, aggregate indication)**")
        if ahead:
            parts.append(
                "- Pipeline share runs ahead of delivery share in: "
                + ", ".join(s["sector"] for s in ahead[:4])
            )
        if behind:
            parts.append(
                "- Delivery share runs ahead of pipeline share in: "
                + ", ".join(s["sector"] for s in behind[:4])
            )
        if not ahead and not behind:
            parts.append("- No sector shows a material imbalance at the current threshold.")

    quality = _quality_lines(facts)
    if quality:
        parts.append("")
        parts.append("**Data quality**")
        parts.extend(quality)

    parts.append("")
    parts.append(
        "_Narrative generation is unavailable, so this answer is a direct rendering "
        "of the computed figures._"
    )
    return "\n".join(parts)


def render_leadership_markdown(facts: dict, *, period: str) -> str:
    """Deterministic leadership update used when Groq is unavailable."""
    parts = [
        "## Executive Summary",
        f"Position for {period}, computed directly from the live Monday.com boards.",
        "",
        "## Pipeline",
    ]
    parts.extend(_deal_lines(facts.get("deals", {})) or ["- Pipeline data is not available."])
    parts.append("")
    parts.append("## Operations")
    parts.extend(_wo_lines(facts.get("work_orders", {})) or ["- Operations data is not available."])

    cross = facts.get("cross_board") or {}
    signals = (cross.get("signals") or {}) if cross.get("available") else {}
    parts.append("")
    parts.append("## Key Risks")
    risks: list[str] = []
    delayed = (facts.get("work_orders") or {}).get("delayed", {})
    if delayed.get("delayed_count"):
        risks.append(f"- {delayed['delayed_count']} work orders are behind their planned end date.")
    at_risk = (facts.get("deals") or {}).get("at_risk", {})
    if at_risk.get("at_risk_count"):
        risks.append(
            f"- {at_risk['at_risk_count']} open deals carry a risk signal "
            f"({', '.join(at_risk.get('criteria', [])[:2])})."
        )
    for entry in (signals.get("delivery_ahead_of_pipeline") or [])[:2]:
        risks.append(
            f"- {entry['sector']}: delivery workload share exceeds pipeline share by "
            f"{abs(entry['gap_share_pts'])} points."
        )
    parts.extend(risks or ["- No data-supported risk was identified in this period."])

    parts.append("")
    parts.append("## Opportunities")
    opportunities: list[str] = []
    breakdown = (facts.get("deals") or {}).get("breakdown") or {}
    for position, row in enumerate((breakdown.get("rows") or [])[:2]):
        label = row.get(breakdown.get("dimension", "sector"), "Unknown")
        verb = "leads the pipeline at" if position == 0 else "follows at"
        opportunities.append(
            f"- {label} {verb} {_display(row.get('value'))} "
            f"({row.get('value_share_pct')}% of the total)."
        )
    for entry in (signals.get("pipeline_ahead_of_delivery") or [])[:2]:
        opportunities.append(
            f"- {entry['sector']}: pipeline share exceeds delivery share by "
            f"{entry['gap_share_pts']} points, indicating growth ahead of current workload."
        )
    parts.extend(opportunities or ["- No data-supported opportunity stands out this period."])

    parts.append("")
    parts.append("## Data Quality")
    parts.extend(_quality_lines(facts, limit=3) or ["- No material data-quality issues detected."])
    return "\n".join(parts)
