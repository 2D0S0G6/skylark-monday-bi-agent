"""End-to-end orchestration.

    question
      -> Groq planner            (understanding only)
      -> Monday.com fetch        (live source of truth)
      -> normalisation           (pandas)
      -> deterministic analytics (pandas)
      -> validation              (sanity checks on the computed facts)
      -> Groq narration          (explanation only)
      -> answer

Every failure mode short-circuits to a user-safe message; nothing raises out of
:meth:`BIAgent.ask`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from agent.data_service import BusinessData, DataService
from agent.llm import GroqLLM
from agent.planner import QueryPlanner
from agent.prompts import CLARIFICATION_TEMPLATE
from agent.response import ResponseWriter
from agent.schemas import QueryPlan
from analytics.cross_board import analyze_cross_board
from analytics.deals import analyze_deals, deal_scope, deals_at_risk, pipeline_by_dimension
from analytics.work_orders import analyze_work_orders, work_order_scope
from config import Settings, get_settings
from monday.client import MondayError
from utils.dates import DateRange, resolve_date_range
from utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["AgentAnswer", "BIAgent"]


@dataclass
class AgentAnswer:
    """Everything the UI needs to render one turn."""

    answer: str
    plan: QueryPlan | None = None
    facts: dict = field(default_factory=dict)
    #: ``llm`` | ``deterministic`` | ``clarification`` | ``error``
    narration_source: str = "llm"
    data_fetched_at: str | None = None
    is_stale: bool = False
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    needs_clarification: bool = False


class BIAgent:
    """The conversational BI agent."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        data_service: DataService | None = None,
        llm: GroqLLM | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.data_service = data_service or DataService(self.settings)
        self.llm = llm if llm is not None else GroqLLM(self.settings)
        self.planner = QueryPlanner(self.llm, fy_start_month=self.settings.fiscal_year_start_month)
        self.writer = ResponseWriter(self.llm)

    # -- public API --------------------------------------------------------
    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
        *,
        force_refresh: bool = False,
        today: pd.Timestamp | None = None,
    ) -> AgentAnswer:
        """Answer one question. Never raises."""
        today = today or pd.Timestamp.today().normalize()

        try:
            plan = self.planner.plan(question, history, today=today)
        except Exception as exc:  # noqa: BLE001 - planning must never break the app
            logger.exception("Planner crashed: %s", exc)
            from agent.planner import heuristic_plan

            plan = heuristic_plan(question, history)

        if plan.intent == "out_of_scope":
            return AgentAnswer(
                answer=(
                    "I can only answer questions about the Skylark Deals and Work Orders "
                    "boards — pipeline, revenue, sectors, deal risk, work-order execution "
                    "and how sales compares with delivery. Try one of the example questions "
                    "in the sidebar."
                ),
                plan=plan,
                narration_source="clarification",
            )

        if plan.needs_clarification and plan.clarification_options:
            options = "\n".join(
                f"{i + 1}. {opt}" for i, opt in enumerate(plan.clarification_options)
            )
            return AgentAnswer(
                answer=(plan.clarification_question or "Which angle would you like?")
                + "\n\n"
                + CLARIFICATION_TEMPLATE.format(options=options).split("\n\n", 1)[1],
                plan=plan,
                narration_source="clarification",
                needs_clarification=True,
            )

        data, fetch_error = self.data_service.get_data_or_stale(force_refresh=force_refresh)
        if data is None:
            message = fetch_error.user_message if fetch_error else (
                "Unable to retrieve the latest Monday.com data. Please verify the board "
                "configuration and API connection."
            )
            return AgentAnswer(
                answer=message, plan=plan, narration_source="error", error=message
            )

        try:
            facts = self.compute_facts(plan, data, today=today)
        except Exception as exc:  # noqa: BLE001 - analytics must never break the app
            logger.exception("Analytics failed: %s", exc)
            return AgentAnswer(
                answer=(
                    "I retrieved the Monday.com data but could not complete the analysis for "
                    "this question. Try rephrasing, or ask about a single board "
                    "(pipeline or work orders)."
                ),
                plan=plan,
                narration_source="error",
                data_fetched_at=data.fetched_at.strftime("%Y-%m-%d %H:%M:%S"),
                warnings=data.warnings,
                error=str(exc),
            )

        validation = validate_facts(facts)
        facts["validation"] = validation

        if plan.intent == "leadership_update":
            answer, source = self.writer.write_leadership_update(
                facts, period=facts.get("period", "the current period"), today=today
            )
        else:
            answer, source = self.writer.write(question, plan, facts)

        warnings = list(data.warnings)
        if data.is_stale:
            warnings.append(
                f"Showing data cached at {data.fetched_at.strftime('%H:%M:%S')} because the "
                f"live refresh failed."
            )
        warnings.extend(validation.get("warnings", []))

        return AgentAnswer(
            answer=answer,
            plan=plan,
            facts=facts,
            narration_source=source,
            data_fetched_at=data.fetched_at.strftime("%Y-%m-%d %H:%M:%S"),
            is_stale=data.is_stale,
            warnings=warnings,
        )

    # -- analytics routing -------------------------------------------------
    def compute_facts(
        self, plan: QueryPlan, data: BusinessData, *, today: pd.Timestamp | None = None
    ) -> dict:
        """Run the deterministic analytics required by ``plan``."""
        today = today or pd.Timestamp.today().normalize()
        fy_start = self.settings.fiscal_year_start_month
        date_range = resolve_date_range(plan.date_range, today=today, fy_start_month=fy_start)

        facts: dict = {
            "question_intent": plan.intent,
            "period": date_range.label if date_range else "all time",
            "period_bounds": date_range.as_dict() if date_range else None,
            "as_of": today.date().isoformat(),
            "data_fetched_at": data.fetched_at.strftime("%Y-%m-%d %H:%M:%S"),
            "filters": {
                "sector": plan.sector,
                "owner": plan.owner,
                "status": plan.status_filter,
                "group_by": plan.group_by,
            },
        }

        if plan.intent == "leadership_update":
            # A leadership update is anchored to a quarter even when the user did
            # not name one (see DECISION_LOG.md).
            if date_range is None:
                date_range = resolve_date_range(
                    "current_quarter", today=today, fy_start_month=fy_start
                )
                facts["period"] = date_range.label
                facts["period_bounds"] = date_range.as_dict()
            return {**facts, **self._leadership_facts(data, date_range, today=today)}
        if plan.intent == "data_quality":
            return {**facts, **self._data_quality_facts(data)}

        wants_deals = "deals" in plan.boards
        wants_work_orders = "work_orders" in plan.boards

        if wants_deals and not data.deals.empty:
            group_by = _deal_group_by(plan.group_by)
            deal_facts = analyze_deals(
                data.deals,
                sector=plan.sector,
                owner=plan.owner,
                date_range=date_range if plan.intent != "revenue_analysis" else date_range,
                status_filter=plan.status_filter,
                group_by=group_by,
                today=today,
                fy_start_month=fy_start,
            )
            if plan.intent in {"deal_analysis", "pipeline_analysis", "general_business_summary"}:
                scope = deal_scope(
                    data.deals, sector=plan.sector, owner=plan.owner,
                    date_range=date_range, status_filter=plan.status_filter,
                )
                deal_facts["at_risk"] = deals_at_risk(data.deals, scope["frame"], today=today)
            if plan.intent == "sales_rep_analysis":
                deal_facts["by_owner"] = pipeline_by_dimension(
                    data.deals, dimension="owner", status="Open"
                )
            facts["deals"] = deal_facts
        elif wants_deals:
            facts["deals"] = {"available": False,
                              "reason": "The Deals board returned no usable rows."}

        if wants_work_orders and not data.work_orders.empty:
            facts["work_orders"] = analyze_work_orders(
                data.work_orders,
                sector=plan.sector,
                owner=plan.owner,
                date_range=date_range if plan.intent in {"work_order_analysis",
                                                         "operational_health"} else None,
                status_filter=plan.status_filter,
                group_by=_wo_group_by(plan.group_by),
                today=today,
            )
        elif wants_work_orders:
            facts["work_orders"] = {"available": False,
                                    "reason": "The Work Orders board returned no usable rows."}

        if plan.requires_cross_board:
            facts["cross_board"] = analyze_cross_board(data.deals, data.work_orders)

        return facts

    def _leadership_facts(
        self, data: BusinessData, date_range: DateRange | None, *, today: pd.Timestamp
    ) -> dict:
        """Assemble the full picture a leadership update needs.

        Interpretation (documented in DECISION_LOG.md): the pipeline section is
        *not* period-filtered (a founder wants the whole open book), while the
        expected-close breakdown supplies the period view.
        """
        fy_start = self.settings.fiscal_year_start_month
        deals = analyze_deals(
            data.deals, date_range=None, status_filter=None, group_by="sector",
            today=today, fy_start_month=fy_start,
        ) if not data.deals.empty else {"available": False}
        if not data.deals.empty:
            deals["at_risk"] = deals_at_risk(data.deals, today=today)
            deals["by_stage_group"] = pipeline_by_dimension(
                data.deals, dimension="stage_group", status="Open"
            )
        work_orders = analyze_work_orders(
            data.work_orders, date_range=None, group_by="sector", today=today,
        ) if not data.work_orders.empty else {"available": False}
        cross = analyze_cross_board(data.deals, data.work_orders)
        return {
            "deals": deals,
            "work_orders": work_orders,
            "cross_board": cross,
            "scope_note": (
                "Pipeline and workload figures cover the whole open book, not just "
                f"{date_range.label if date_range else 'the period'}. The "
                "'expected_close' block shows how open pipeline falls across quarters."
            ),
        }

    @staticmethod
    def _data_quality_facts(data: BusinessData) -> dict:
        return {
            "deals": {"data_quality": data.deals.quality.as_dict(),
                      "rows": len(data.deals.frame),
                      "available_fields": data.deals.available_fields},
            "work_orders": {"data_quality": data.work_orders.quality.as_dict(),
                            "rows": len(data.work_orders.frame),
                            "available_fields": data.work_orders.available_fields},
            "board_mapping": {
                "deals_unmapped": data.deals_mapping.unmapped_fields if data.deals_mapping else [],
                "work_orders_unmapped": (
                    data.work_orders_mapping.unmapped_fields if data.work_orders_mapping else []
                ),
            },
        }


_DEAL_DIMENSIONS = {
    "sector": "sector", "stage": "stage", "stage_group": "stage_group",
    "owner": "owner", "product": "product", "client": "client", "none": "none",
}
_WO_DIMENSIONS = {
    "sector": "sector", "execution_status": "execution_status", "owner": "owner",
    "nature_of_work": "nature_of_work", "client": "client", "none": "none",
}


def _deal_group_by(group_by: str | None) -> str:
    return _DEAL_DIMENSIONS.get((group_by or "sector"), "sector")


def _wo_group_by(group_by: str | None) -> str:
    return _WO_DIMENSIONS.get((group_by or "sector"), "sector")


def validate_facts(facts: dict) -> dict:
    """Sanity-check computed facts before they are narrated.

    This is a guard against silently shipping an impossible figure (negative
    pipeline, shares that do not add up, a breakdown that exceeds its total).
    """
    warnings: list[str] = []
    checks: list[str] = []

    deals = facts.get("deals") or {}
    summary = deals.get("summary") or {}
    open_value = (summary.get("open_pipeline_value") or {}).get("amount")
    total_value = (summary.get("total_value") or {}).get("amount")

    if open_value is not None:
        checks.append("open_pipeline_non_negative")
        if open_value < 0:
            warnings.append("Computed open pipeline is negative; check the source deal values.")
    if open_value is not None and total_value is not None and open_value - total_value > 1:
        warnings.append("Open pipeline exceeds total deal value; the status field may be inconsistent.")

    breakdown = deals.get("breakdown") or {}
    rows = breakdown.get("rows") or []
    if rows and breakdown.get("total_value"):
        checks.append("breakdown_within_total")
        summed = sum((r.get("value") or {}).get("amount", 0) for r in rows)
        total = breakdown["total_value"]["amount"]
        if total and summed - total > max(1.0, abs(total) * 0.001) and not breakdown.get("truncated"):
            warnings.append("Sector breakdown exceeds the pipeline total; grouping may be duplicated.")

    work_orders = facts.get("work_orders") or {}
    wo_summary = work_orders.get("summary") or {}
    count = wo_summary.get("work_order_count")
    active = wo_summary.get("active_work_orders")
    completed = wo_summary.get("completed_work_orders")
    if count is not None and active is not None and completed is not None:
        checks.append("work_order_counts_consistent")
        if active + completed > count:
            warnings.append("Active plus completed work orders exceed the total; statuses overlap.")

    return {"checks_run": checks, "warnings": warnings, "passed": not warnings}
