"""Pydantic schemas for the structured query plan produced by the LLM."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Intent = Literal[
    "pipeline_analysis",
    "revenue_analysis",
    "deal_analysis",
    "sector_analysis",
    "sales_rep_analysis",
    "work_order_analysis",
    "operational_health",
    "cross_board_analysis",
    "leadership_update",
    "data_quality",
    "general_business_summary",
    "greeting",
    "out_of_scope",
]

ALL_INTENTS: tuple[str, ...] = (
    "pipeline_analysis", "revenue_analysis", "deal_analysis", "sector_analysis",
    "sales_rep_analysis", "work_order_analysis", "operational_health",
    "cross_board_analysis", "leadership_update", "data_quality",
    "general_business_summary", "greeting", "out_of_scope",
)

#: Tokens the date resolver understands. The planner is constrained to these.
DATE_RANGE_TOKENS: tuple[str, ...] = (
    "all_time", "current_quarter", "next_quarter", "last_quarter", "current_month",
    "last_month", "current_fy", "last_fy", "current_year", "ytd",
    "next_30_days", "next_90_days", "last_30_days", "last_90_days", "overdue",
)

GROUP_BY_OPTIONS: tuple[str, ...] = (
    "none", "sector", "stage_group", "stage", "owner", "product", "execution_status",
    "nature_of_work", "client",
)


class QueryPlan(BaseModel):
    """Structured interpretation of a founder's question.

    Produced by the LLM (validated here) or by the deterministic keyword fallback
    when the LLM is unavailable or returns unusable JSON.
    """

    intent: Intent = "general_business_summary"
    boards: list[Literal["deals", "work_orders"]] = Field(default_factory=lambda: ["deals"])
    metric: str | None = None
    sector: str | None = None
    owner: str | None = None
    date_range: str = "all_time"
    status_filter: str | None = None
    group_by: str | None = "sector"
    requires_cross_board: bool = False
    needs_clarification: bool = False
    clarification_question: str | None = None
    clarification_options: list[str] = Field(default_factory=list)
    #: Free-text note from the planner about how it read the question.
    reasoning: str | None = None
    #: ``llm`` | ``fallback`` | ``llm_repaired``
    source: str = "llm"

    @field_validator("date_range", mode="before")
    @classmethod
    def _clean_date_range(cls, value: object) -> str:
        if value in (None, "", "null"):
            return "all_time"
        token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        return token

    @field_validator("group_by", mode="before")
    @classmethod
    def _clean_group_by(cls, value: object) -> str | None:
        if value in (None, "", "null", "none"):
            return "none"
        return str(value).strip().lower().replace(" ", "_")

    @field_validator("sector", "owner", "metric", "status_filter", mode="before")
    @classmethod
    def _clean_optional(cls, value: object) -> str | None:
        if value in (None, "", "null", "any", "all"):
            return None
        return str(value).strip()

    @field_validator("boards", mode="before")
    @classmethod
    def _clean_boards(cls, value: object) -> list[str]:
        if not value:
            return ["deals"]
        if isinstance(value, str):
            value = [value]
        allowed = {"deals", "work_orders"}
        cleaned = [
            str(v).strip().lower().replace(" ", "_").replace("workorders", "work_orders")
            for v in value
        ]
        cleaned = [v for v in cleaned if v in allowed]
        return cleaned or ["deals"]

    def with_defaults(self) -> QueryPlan:
        """Fill board selection implied by the intent, so the plan is always runnable."""
        boards = set(self.boards)
        if self.intent in {"work_order_analysis", "operational_health"}:
            boards.add("work_orders")
        if self.intent in {"pipeline_analysis", "revenue_analysis", "deal_analysis",
                           "sales_rep_analysis"}:
            boards.add("deals")
        if self.intent in {"cross_board_analysis", "leadership_update",
                           "general_business_summary", "data_quality", "sector_analysis"}:
            boards.update({"deals", "work_orders"})
        requires_cross = self.requires_cross_board or self.intent in {
            "cross_board_analysis", "leadership_update"
        }
        return self.model_copy(update={
            "boards": sorted(boards),
            "requires_cross_board": requires_cross,
        })
