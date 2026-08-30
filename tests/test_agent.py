"""Planner, narration, caching and end-to-end orchestration (all without a live API)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from agent.data_service import DataService
from agent.llm import GroqLLM, LLMError, extract_json
from agent.orchestrator import BIAgent, validate_facts
from agent.planner import QueryPlanner, heuristic_plan
from agent.response import ResponseWriter, render_facts_markdown
from agent.schemas import QueryPlan
from monday.client import MondayClient
from tests.conftest import TODAY, FakeMondayAPI, make_settings


# --- stubs ------------------------------------------------------------------

class StubGroq:
    """Minimal stand-in for the Groq SDK client."""

    def __init__(self, responses: list[str] | None = None, *, raise_on_call: bool = False):
        self.responses = list(responses or [])
        self.raise_on_call = raise_on_call
        self.calls: list[dict] = []
        self.chat = self  # groq client shape: client.chat.completions.create

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_on_call:
            raise RuntimeError("groq is down")
        text = self.responses.pop(0) if self.responses else "{}"

        class _Message:
            content = text

        class _Choice:
            message = _Message()

        class _Completion:
            choices = [_Choice()]

        return _Completion()


def build_agent(llm_client=None, *, api=None):
    settings = make_settings()
    api = api or FakeMondayAPI()
    service = DataService(
        settings,
        client=MondayClient(settings, transport=api.transport(), max_retries=1),
        today=TODAY,
    )
    llm = GroqLLM(make_settings(groq_api_key="test-key"), client=llm_client)
    return BIAgent(settings=settings, data_service=service, llm=llm), api


# --- planner ----------------------------------------------------------------

@pytest.mark.parametrize(
    "question,intent",
    [
        ("How's our pipeline looking this quarter?", "pipeline_analysis"),
        ("How many work orders are delayed?", "work_order_analysis"),
        ("Compare pipeline vs operational workload.", "cross_board_analysis"),
        ("Prepare a leadership update.", "leadership_update"),
        ("Give me an executive summary", "leadership_update"),
        ("Which sectors have the strongest pipeline?", "sector_analysis"),
        ("What's our expected revenue this quarter?", "revenue_analysis"),
        ("How is data quality on the boards?", "data_quality"),
    ],
)
def test_heuristic_planner_classifies_core_questions(question, intent):
    assert heuristic_plan(question).intent == intent


def test_heuristic_planner_extracts_period_and_sector():
    plan = heuristic_plan("What's the energy pipeline this quarter?")
    assert plan.sector == "Energy"
    assert plan.date_range == "current_quarter"


def test_heuristic_planner_asks_for_clarification_only_when_vague():
    vague = heuristic_plan("How are we doing?")
    assert vague.needs_clarification is True
    assert len(vague.clarification_options) >= 3

    specific = heuristic_plan("What's our energy pipeline this quarter?")
    assert specific.needs_clarification is False


def test_follow_up_inherits_previous_context():
    history = [{"question": "How is energy doing?",
                "plan": {"intent": "sector_analysis", "sector": "Energy",
                         "date_range": "current_quarter", "status_filter": "open"}}]
    plan = heuristic_plan("What about infrastructure?", history)
    assert plan.intent == "sector_analysis"
    assert plan.sector == "Infrastructure"
    assert plan.date_range == "current_quarter"


def test_llm_planner_uses_valid_json():
    payload = json.dumps({
        "intent": "sector_analysis", "boards": ["deals"], "metric": "pipeline_value",
        "sector": "energy", "date_range": "current_quarter", "status_filter": "open",
        "group_by": "sector", "requires_cross_board": False,
        "needs_clarification": False, "clarification_question": None,
    })
    llm = GroqLLM(make_settings(groq_api_key="k"), client=StubGroq([payload]))
    plan = QueryPlanner(llm).plan("How is energy doing this quarter?")
    assert plan.intent == "sector_analysis"
    assert plan.sector == "energy"
    assert plan.date_range == "current_quarter"


def test_llm_planner_recovers_from_fenced_json():
    payload = "```json\n{\"intent\": \"pipeline_analysis\", \"boards\": [\"deals\"]}\n```"
    llm = GroqLLM(make_settings(groq_api_key="k"), client=StubGroq([payload]))
    plan = QueryPlanner(llm).plan("pipeline please")
    assert plan.intent == "pipeline_analysis"


def test_llm_planner_falls_back_on_malformed_json():
    llm = GroqLLM(make_settings(groq_api_key="k"), client=StubGroq(["I think you want pipeline"]))
    plan = QueryPlanner(llm).plan("How's our pipeline this quarter?")
    assert plan.source == "fallback"
    assert plan.intent == "pipeline_analysis"


def test_llm_planner_repairs_partially_invalid_json():
    payload = json.dumps({"intent": "not_a_real_intent", "sector": "mining",
                          "date_range": "current_quarter"})
    llm = GroqLLM(make_settings(groq_api_key="k"), client=StubGroq([payload]))
    plan = QueryPlanner(llm).plan("How is mining doing this quarter?")
    assert plan.sector == "mining"
    assert plan.intent in {"sector_analysis", "general_business_summary"}


def test_llm_planner_falls_back_when_groq_raises():
    llm = GroqLLM(make_settings(groq_api_key="k"), client=StubGroq(raise_on_call=True))
    plan = QueryPlanner(llm).plan("How many work orders are delayed?")
    assert plan.source == "fallback"
    assert plan.intent == "work_order_analysis"


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('prefix {"a": 1} suffix') == {"a": 1}
    assert extract_json("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert extract_json("no json here") is None
    assert extract_json("") is None


# --- orchestration ----------------------------------------------------------

def test_end_to_end_answer_without_groq():
    """No Groq key: the app must still produce a correct, computed answer."""
    settings = make_settings()
    api = FakeMondayAPI()
    service = DataService(
        settings, client=MondayClient(settings, transport=api.transport(), max_retries=1),
        today=TODAY,
    )
    agent = BIAgent(settings=settings, data_service=service, llm=GroqLLM(settings, client=None))
    answer = agent.ask("How's our pipeline looking?", today=TODAY)
    assert answer.narration_source == "deterministic"
    assert "₹10.25 Cr" in answer.answer
    assert answer.error is None


def test_end_to_end_answer_with_groq_narration():
    plan_json = json.dumps({"intent": "pipeline_analysis", "boards": ["deals"],
                            "status_filter": "open", "group_by": "sector"})
    narration = "### Pipeline\n\n**₹10.25 Cr** open pipeline.\n\n**What this means:** healthy."
    stub = StubGroq([plan_json, narration])
    agent, _ = build_agent(stub)
    answer = agent.ask("How's our pipeline?", today=TODAY)
    assert answer.narration_source == "llm"
    assert answer.answer == narration
    # The narrator must be handed pre-computed facts, never raw rows.
    narrator_call = stub.calls[1]["messages"][1]["content"]
    assert "Computed facts" in narrator_call
    assert "10.25 Cr" in narrator_call


def test_empty_narration_degrades_to_the_deterministic_renderer():
    """A too-short/empty completion must not become the user's answer."""
    plan_json = json.dumps({"intent": "pipeline_analysis", "boards": ["deals"]})
    agent, _ = build_agent(StubGroq([plan_json, "   "]))
    answer = agent.ask("How's our pipeline?", today=TODAY)
    assert answer.narration_source == "deterministic"
    assert "₹10.25 Cr" in answer.answer


def test_clarification_short_circuits_before_fetching_data():
    settings = make_settings()
    api = FakeMondayAPI()
    service = DataService(
        settings, client=MondayClient(settings, transport=api.transport(), max_retries=1),
        today=TODAY,
    )
    agent = BIAgent(settings=settings, data_service=service, llm=GroqLLM(settings, client=None))
    answer = agent.ask("How are we doing?", today=TODAY)
    assert answer.needs_clarification is True
    assert api.request_count == 0, "no Monday.com call should be made for a clarification"


def test_monday_failure_returns_a_user_safe_message():
    settings = make_settings()
    api = FakeMondayAPI()
    api.fail_with = (401, {"errors": [{"message": "unauthorized"}]})
    service = DataService(
        settings, client=MondayClient(settings, transport=api.transport(), max_retries=1),
        today=TODAY,
    )
    agent = BIAgent(settings=settings, data_service=service, llm=GroqLLM(settings, client=None))
    answer = agent.ask("How's our pipeline?", today=TODAY)
    assert answer.narration_source == "error"
    assert "token" in answer.answer.lower()
    assert "Traceback" not in answer.answer


def test_stale_cache_is_used_and_labelled_when_a_refresh_fails():
    settings = make_settings()
    api = FakeMondayAPI()
    service = DataService(
        settings, client=MondayClient(settings, transport=api.transport(), max_retries=1),
        today=TODAY,
    )
    service.get_data()  # warm the cache
    api.fail_with = (503, {})
    data, error = service.get_data_or_stale(force_refresh=True)
    assert data is not None
    assert data.is_stale is True
    assert error is not None
    assert any("refresh failed" in w.lower() for w in data.warnings)


def test_cache_prevents_repeat_fetches_within_the_ttl():
    settings = make_settings(cache_ttl_seconds=300)
    api = FakeMondayAPI()
    service = DataService(
        settings, client=MondayClient(settings, transport=api.transport(), max_retries=1),
        today=TODAY,
    )
    service.get_data()
    first = api.request_count
    service.get_data()
    assert api.request_count == first, "the second read must be served from cache"
    service.get_data(force_refresh=True)
    assert api.request_count > first, "an explicit refresh must bypass the cache"


def test_zero_ttl_always_refetches():
    settings = make_settings(cache_ttl_seconds=0)
    api = FakeMondayAPI()
    service = DataService(
        settings, client=MondayClient(settings, transport=api.transport(), max_retries=1),
        today=TODAY,
    )
    service.get_data()
    first = api.request_count
    service.get_data()
    assert api.request_count > first


def test_leadership_update_covers_every_required_section():
    settings = make_settings()
    api = FakeMondayAPI()
    service = DataService(
        settings, client=MondayClient(settings, transport=api.transport(), max_retries=1),
        today=TODAY,
    )
    agent = BIAgent(settings=settings, data_service=service, llm=GroqLLM(settings, client=None))
    answer = agent.ask("Prepare a leadership update", today=TODAY)
    for heading in ("## Executive Summary", "## Pipeline", "## Operations",
                    "## Key Risks", "## Opportunities", "## Data Quality"):
        assert heading in answer.answer
    assert answer.facts["cross_board"]["available"] is True


def test_out_of_scope_question_is_declined_politely():
    payload = json.dumps({"intent": "out_of_scope", "boards": ["deals"]})
    agent, api = build_agent(StubGroq([payload]))
    answer = agent.ask("What is the capital of France?", today=TODAY)
    assert "Deals and Work Orders" in answer.answer
    assert api.request_count == 0


def test_validation_flags_impossible_figures():
    result = validate_facts({
        "work_orders": {"summary": {"work_order_count": 5, "active_work_orders": 4,
                                    "completed_work_orders": 3}}
    })
    assert result["passed"] is False
    assert any("exceed the total" in w for w in result["warnings"])


def test_validation_passes_on_consistent_figures(deals_dataset):
    from analytics.deals import analyze_deals

    result = validate_facts({"deals": analyze_deals(deals_dataset, today=TODAY)})
    assert result["passed"] is True
    assert "open_pipeline_non_negative" in result["checks_run"]


def test_facts_sent_to_the_llm_stay_json_serialisable():
    agent, _ = build_agent()
    from agent.schemas import QueryPlan

    data, _ = agent.data_service.get_data_or_stale()
    facts = agent.compute_facts(QueryPlan(intent="leadership_update").with_defaults(),
                                data, today=TODAY)
    json.dumps(facts, default=str)


def test_deterministic_renderer_quotes_precomputed_displays():
    agent, _ = build_agent()
    data, _ = agent.data_service.get_data_or_stale()
    plan = QueryPlan(intent="pipeline_analysis").with_defaults()
    facts = agent.compute_facts(plan, data, today=TODAY)
    markdown = render_facts_markdown(plan, facts)
    assert "₹" in markdown
    assert "Data quality" in markdown
