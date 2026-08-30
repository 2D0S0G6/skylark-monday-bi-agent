"""Skylark Drones — Monday.com Business Intelligence Agent (Streamlit UI).

Run with:  streamlit run app.py

The UI is a thin shell. All understanding happens in :mod:`agent.planner`, all
arithmetic in :mod:`analytics`, and all narration in :mod:`agent.response`.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from agent.data_service import DataService
from agent.llm import GroqLLM
from agent.orchestrator import AgentAnswer, BIAgent
from config import get_settings, reset_settings_cache
from monday.client import MondayError
from utils.logging import configure_logging

configure_logging()

st.set_page_config(
    page_title="Skylark BI Agent",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLE_QUESTIONS = [
    "How's our pipeline looking this quarter?",
    "What's the energy sector pipeline?",
    "Which sectors have the strongest pipeline?",
    "What are our biggest opportunities?",
    "Which deals are at risk?",
    "How many active work orders do we have?",
    "How many work orders are delayed?",
    "Compare pipeline vs operational workload.",
    "Where might we need delivery capacity next?",
    "Prepare a leadership update.",
]

STYLE = """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1180px; }
  .sk-metric {
    border: 1px solid rgba(128,128,128,.25); border-radius: 10px;
    padding: .7rem .9rem; background: rgba(128,128,128,.06);
  }
  .sk-metric .sk-label {
    font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; opacity: .7;
  }
  .sk-metric .sk-value { font-size: 1.35rem; font-weight: 650; line-height: 1.5; }
  .sk-metric .sk-sub { font-size: .75rem; opacity: .65; }
  .sk-pill {
    display:inline-block; font-size:.7rem; padding:.12rem .5rem; border-radius:999px;
    border:1px solid rgba(128,128,128,.35); margin-right:.35rem; opacity:.85;
  }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    st.session_state.setdefault("history", [])       # conversational memory
    st.session_state.setdefault("pending_question", None)
    st.session_state.setdefault("last_answer", None)


@st.cache_resource(show_spinner=False)
def get_agent(_cache_key: str) -> BIAgent:
    """One agent (and therefore one data cache) per configuration."""
    settings = get_settings()
    return BIAgent(
        settings=settings,
        data_service=DataService(settings),
        llm=GroqLLM(settings),
    )


def _cache_key() -> str:
    s = get_settings()
    return f"{s.monday_deals_board_id}:{s.monday_work_orders_board_id}:{s.groq_model}"


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------

def render_sidebar(agent: BIAgent | None) -> None:
    settings = get_settings()
    with st.sidebar:
        st.markdown("### 🛰️ Skylark BI Agent")
        st.caption("Live executive analytics over the Monday.com Deals and Work Orders boards.")

        st.markdown("#### Ask about")
        for i, question in enumerate(EXAMPLE_QUESTIONS):
            if st.button(question, key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()

        st.divider()
        st.markdown("#### Data")
        if agent is not None:
            data = getattr(agent.data_service, "_cache", None)
            if data is not None:
                st.caption(
                    f"Last fetched **{data.fetched_at.strftime('%H:%M:%S')}** "
                    f"({int(data.age_seconds)}s ago) · TTL {settings.cache_ttl_seconds}s"
                )
                st.caption(
                    f"{len(data.deals.frame)} deals · {len(data.work_orders.frame)} work orders"
                )
            else:
                st.caption("No data fetched yet.")

            if st.button("🔄 Refresh data from Monday.com", use_container_width=True):
                with st.spinner("Fetching the latest board data…"):
                    agent.data_service.invalidate()
                    data, error = agent.data_service.get_data_or_stale(force_refresh=True)
                if error:
                    st.error(error.user_message)
                elif data:
                    st.success(f"Refreshed at {data.fetched_at.strftime('%H:%M:%S')}.")
                st.rerun()

        if st.button("🧹 Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_answer = None
            st.rerun()

        st.divider()
        with st.expander("Configuration", expanded=False):
            for key, value in settings.safe_summary().items():
                st.write(f"**{key}:** {value}")
            st.caption("Secrets are read from the environment and are never displayed.")
            if st.button("Reload configuration", use_container_width=True):
                reset_settings_cache()
                st.cache_resource.clear()
                st.rerun()


# ---------------------------------------------------------------------------
# setup screen
# ---------------------------------------------------------------------------

def render_setup_screen(missing: list[str]) -> None:
    st.title("🛰️ Skylark Drones — Business Intelligence Agent")
    st.warning(
        "The agent is not configured yet. It reads its credentials from environment "
        "variables (or Streamlit secrets) and never stores them in the repository."
    )
    st.markdown("**Missing configuration:** " + ", ".join(f"`{m}`" for m in missing))
    st.markdown(
        """
Create a `.env` file next to `app.py` (copy `.env.example`) containing:

```bash
MONDAY_API_TOKEN=your_monday_api_token
MONDAY_DEALS_BOARD_ID=1234567890
MONDAY_WORK_ORDERS_BOARD_ID=0987654321
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

* **Monday API token** — Monday.com → your avatar → *Developers* → *My access tokens*.
* **Board IDs** — open a board; the ID is the number in the URL
  (`.../boards/1234567890`).
* **Groq key** — <https://console.groq.com> → *API Keys*.

On Streamlit Community Cloud, put the same keys in *App settings → Secrets*.

See `README.md` for the full board-import walkthrough.
"""
    )
    if st.button("I've configured it — reload"):
        reset_settings_cache()
        st.cache_resource.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# answer rendering
# ---------------------------------------------------------------------------

def _metric_card(label: str, value: str, sub: str = "") -> str:
    return (
        f"<div class='sk-metric'><div class='sk-label'>{label}</div>"
        f"<div class='sk-value'>{value}</div>"
        f"<div class='sk-sub'>{sub}</div></div>"
    )


def render_headline_metrics(facts: dict) -> None:
    """A compact KPI strip drawn only from figures that were actually computed."""
    cards: list[tuple[str, str, str]] = []

    deals = facts.get("deals") or {}
    summary = deals.get("summary") or {}
    if summary.get("open_pipeline_value"):
        cards.append((
            "Open pipeline",
            summary["open_pipeline_value"]["display"],
            f"{summary.get('open_deal_count', 0)} open deals",
        ))
    elif summary.get("open_deal_count") is not None:
        cards.append(("Open deals", str(summary["open_deal_count"]), "value unavailable"))

    if summary.get("late_stage_open_value"):
        share = summary.get("late_stage_share_of_open_pct")
        cards.append((
            "Late-stage",
            summary["late_stage_open_value"]["display"],
            f"{share}% of open pipeline" if share is not None else "",
        ))
    if summary.get("weighted_open_pipeline_value"):
        cards.append((
            "Weighted pipeline",
            summary["weighted_open_pipeline_value"]["display"],
            "probability-weighted",
        ))

    work_orders = facts.get("work_orders") or {}
    wo_summary = work_orders.get("summary") or {}
    if wo_summary.get("active_work_orders") is not None:
        cards.append((
            "Active work orders",
            str(wo_summary["active_work_orders"]),
            f"of {wo_summary.get('work_order_count', 0)} in scope",
        ))
    delayed = work_orders.get("delayed") or {}
    if delayed.get("delayed_count") is not None and wo_summary:
        share = delayed.get("delayed_share_pct")
        cards.append((
            "Delayed",
            str(delayed.get("delayed_count", 0)),
            f"{share}% of assessable" if share is not None else "",
        ))

    if not cards:
        return
    columns = st.columns(min(len(cards), 5))
    for column, (label, value, sub) in zip(columns, cards[:5]):
        column.markdown(_metric_card(label, value, sub), unsafe_allow_html=True)


def _quality_notices(facts: dict) -> list[dict]:
    issues: list[dict] = []
    for block in ("deals", "work_orders", "cross_board"):
        section = facts.get(block) or {}
        issues.extend(((section.get("data_quality") or {}).get("issues") or []))
    seen, unique = set(), []
    for issue in issues:
        if issue.get("code") in seen:
            continue
        seen.add(issue.get("code"))
        unique.append(issue)
    order = {"excluded": 0, "included_with_gap": 1, "info": 2}
    unique.sort(key=lambda i: (order.get(i.get("severity"), 3), -i.get("count", 0)))
    return unique


def render_details(answer: AgentAnswer) -> None:
    """Expandable transparency panels: how the question was read and what was computed."""
    facts = answer.facts or {}

    notices = _quality_notices(facts)
    if notices:
        excluded = [n for n in notices if n["severity"] == "excluded"]
        with st.expander(
            f"⚠️ Data quality — {len(notices)} issue(s) detected"
            + (f", {len(excluded)} affecting the figures above" if excluded else ""),
            expanded=False,
        ):
            for notice in notices[:12]:
                icon = {"excluded": "🔴", "included_with_gap": "🟡"}.get(notice["severity"], "⚪")
                share = f" · {notice['share_pct']}% of rows" if notice.get("share_pct") else ""
                st.markdown(f"{icon} {notice['message']}{share}")
            st.caption(
                "🔴 excluded from the calculation · 🟡 included but incomplete · ⚪ informational"
            )

    with st.expander("🔍 Analysis details", expanded=False):
        if answer.plan:
            plan = answer.plan
            pills = [
                f"intent: {plan.intent}",
                f"boards: {', '.join(plan.boards)}",
                f"period: {plan.date_range}",
            ]
            if plan.sector:
                pills.append(f"sector: {plan.sector}")
            if plan.status_filter:
                pills.append(f"status: {plan.status_filter}")
            pills.append(f"planner: {plan.source}")
            pills.append(f"narration: {answer.narration_source}")
            st.markdown(
                " ".join(f"<span class='sk-pill'>{p}</span>" for p in pills),
                unsafe_allow_html=True,
            )
            if plan.reasoning:
                st.caption(plan.reasoning)

        validation = facts.get("validation") or {}
        if validation.get("checks_run"):
            status = "✅ passed" if validation.get("passed") else "⚠️ warnings"
            st.markdown(
                f"**Validation:** {status} — {', '.join(validation['checks_run'])}"
            )
            for warning in validation.get("warnings", []):
                st.warning(warning)

        _render_tables(facts)

        st.markdown("**Computed facts (raw)**")
        st.json(facts, expanded=False)


def _render_tables(facts: dict) -> None:
    """Show the grouped tables behind the narrative, when they exist."""
    deals = facts.get("deals") or {}
    breakdown = deals.get("breakdown") or {}
    if breakdown.get("rows"):
        st.markdown(f"**Pipeline by {breakdown['dimension'].replace('_', ' ')}**")
        st.dataframe(_flatten_rows(breakdown["rows"]), use_container_width=True, hide_index=True)

    top = deals.get("top_opportunities") or {}
    if top.get("rows"):
        st.markdown("**Largest open opportunities**")
        st.dataframe(_flatten_rows(top["rows"]), use_container_width=True, hide_index=True)

    at_risk = deals.get("at_risk") or {}
    if at_risk.get("rows"):
        st.markdown("**Deals carrying a risk signal**")
        st.dataframe(_flatten_rows(at_risk["rows"]), use_container_width=True, hide_index=True)

    work_orders = facts.get("work_orders") or {}
    wo_breakdown = work_orders.get("breakdown") or {}
    if wo_breakdown.get("rows"):
        st.markdown(f"**Work orders by {wo_breakdown['dimension'].replace('_', ' ')}**")
        st.dataframe(_flatten_rows(wo_breakdown["rows"]), use_container_width=True, hide_index=True)

    delayed = work_orders.get("delayed") or {}
    if delayed.get("rows"):
        st.markdown("**Delayed work orders**")
        st.dataframe(_flatten_rows(delayed["rows"]), use_container_width=True, hide_index=True)

    cross = facts.get("cross_board") or {}
    comparison = cross.get("comparison") or {}
    if comparison.get("rows"):
        st.markdown("**Pipeline vs active workload, by sector**")
        st.dataframe(_flatten_rows(comparison["rows"]), use_container_width=True, hide_index=True)
        st.caption(comparison.get("join_policy", {}).get("interpretation", ""))


def _flatten_rows(rows: list[dict]) -> pd.DataFrame:
    """Flatten ``{"amount": ..., "display": ...}`` money blocks for display."""
    flattened = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if isinstance(value, dict) and "display" in value:
                item[key] = value["display"]
            elif isinstance(value, list):
                item[key] = ", ".join(str(v) for v in value)
            else:
                item[key] = value
        flattened.append(item)
    return pd.DataFrame(flattened)


def render_data_sources(agent: BIAgent) -> None:
    data = getattr(agent.data_service, "_cache", None)
    if data is None:
        return
    with st.expander("🗂️ Data sources", expanded=False):
        summary = data.source_summary()
        st.caption(
            f"Fetched from Monday.com at **{summary['fetched_at']}** "
            f"({summary['age_seconds']}s ago)."
            + (" ⚠️ This snapshot is stale." if summary["is_stale"] else "")
        )
        for key, label in (("deals", "Deals board"), ("work_orders", "Work Orders board")):
            block = summary[key]
            st.markdown(
                f"**{label}** — `{block['board_name']}` (ID `{block['board_id']}`), "
                f"{block['rows']} rows"
            )
            if block["mapped_fields"]:
                st.dataframe(
                    pd.DataFrame(block["mapped_fields"]),
                    use_container_width=True, hide_index=True,
                )
            if block["unmapped_fields"]:
                st.caption(
                    "Fields with no matching column (related metrics are reported as "
                    f"unavailable): {', '.join(block['unmapped_fields'])}"
                )
        for warning in summary["warnings"]:
            st.warning(warning)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_state()
    settings = get_settings()

    missing = [m for m in settings.missing_settings() if m != "GROQ_API_KEY"]
    if missing:
        render_sidebar(None)
        render_setup_screen(settings.missing_settings())
        return

    try:
        agent = get_agent(_cache_key())
    except MondayError as exc:
        render_sidebar(None)
        st.error(exc.user_message)
        return

    render_sidebar(agent)

    st.title("🛰️ Skylark Drones — Business Intelligence Agent")
    st.caption(
        "Ask about pipeline, revenue, sectors, deal risk, work-order execution, or how "
        "sales compares with delivery. Every figure is computed in Python from live "
        "Monday.com data; the language model only interprets your question and explains "
        "the results."
    )

    if not settings.groq_configured:
        st.info(
            "`GROQ_API_KEY` is not set. The agent still answers using keyword-based "
            "question understanding and computed figures, but without LLM narration."
        )

    # replay conversation
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])

    question = st.chat_input("Ask about pipeline, sectors, deals or operations…")
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if not question:
        if st.session_state.last_answer:
            answer = st.session_state.last_answer
            render_headline_metrics(answer.facts)
            render_details(answer)
            render_data_sources(agent)
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Reading the question, querying Monday.com and computing…"):
            answer = agent.ask(question, history=st.session_state.history)

        st.markdown(answer.answer)

        if answer.error:
            st.error("The analysis could not be completed. See 'Analysis details' for context.")
        for warning in answer.warnings[:3]:
            st.warning(warning)
        if answer.data_fetched_at:
            st.caption(
                f"Monday.com data fetched at {answer.data_fetched_at}"
                + (" · ⚠️ stale snapshot" if answer.is_stale else "")
            )

    if answer.facts:
        render_headline_metrics(answer.facts)
    render_details(answer)
    render_data_sources(agent)

    st.session_state.history.append({
        "question": question,
        "answer": answer.answer,
        "plan": answer.plan.model_dump() if answer.plan else None,
    })
    st.session_state.history = st.session_state.history[-12:]
    st.session_state.last_answer = answer


if __name__ == "__main__":
    main()
