"""Planner, narration, caching and end-to-end orchestration (all without a live API)."""
from __future__ import annotations

import json

import pytest

from agent.data_service import DataService
from agent.llm import GroqLLM, extract_json
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
    assert "Deals" in answer.answer and "Work Orders" in answer.answer
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


# --- greetings and tone -----------------------------------------------------

@pytest.mark.parametrize(
    "greeting",
    ["hi", "Hi!", "hello", "Hey there", "good morning", "thanks", "thank you",
     "who are you", "what can you do?", "help", "how does this work"],
)
def test_greetings_are_recognised_not_refused(greeting):
    plan = heuristic_plan(greeting)
    assert plan.intent == "greeting"


@pytest.mark.parametrize(
    "question",
    ["hi, how is the mining pipeline?", "hello - which deals are at risk?",
     "hey what's our biggest opportunity"],
)
def test_a_greeting_prefix_does_not_swallow_the_real_question(question):
    plan = heuristic_plan(question)
    assert plan.intent != "greeting"


def test_greeting_answers_warmly_without_touching_monday_or_groq():
    """A 'hi' must cost no API call and must not read as a refusal."""
    settings = make_settings()
    api = FakeMondayAPI()
    stub = StubGroq()
    service = DataService(
        settings, client=MondayClient(settings, transport=api.transport(), max_retries=1),
        today=TODAY,
    )
    agent = BIAgent(settings=settings, data_service=service,
                    llm=GroqLLM(make_settings(groq_api_key="k"), client=stub))
    answer = agent.ask("hi", today=TODAY)

    assert answer.narration_source == "greeting"
    assert api.request_count == 0, "a greeting must not fetch board data"
    assert stub.calls == [], "a greeting must not call the LLM"

    text = answer.answer.lower()
    assert "hello" in text
    assert "only answer" not in text and "not able to help" not in text
    # It should orient the user, not just decline.
    assert "pipeline" in text and "leadership update" in text


def test_out_of_scope_declines_but_stays_helpful():
    payload = json.dumps({"intent": "out_of_scope", "boards": ["deals"]})
    agent, api = build_agent(StubGroq([payload]))
    answer = agent.ask("What is the capital of France?", today=TODAY)

    text = answer.answer
    assert api.request_count == 0
    assert "Deals" in text and "Work Orders" in text
    # Declines, but offers a way forward rather than ending the conversation.
    assert "glad to help" in text.lower()
    assert "example questions" in text.lower()


@pytest.mark.parametrize(
    "question",
    ["what is the capital of France?", "write me a poem", "what's the weather like",
     "tell me a joke"],
)
def test_fallback_planner_declines_unrelated_questions(question):
    """Without Groq, an unrelated question must not get a business summary."""
    assert heuristic_plan(question).intent == "out_of_scope"


@pytest.mark.parametrize(
    "question",
    ["How's our pipeline looking this quarter?", "Which deals are at risk?",
     "How many work orders are delayed?", "How is energy doing?",
     "How is OWNER_002 performing?", "What are our biggest opportunities?",
     "Prepare a leadership update.", "How are we doing?"],
)
def test_fallback_planner_never_declines_a_legitimate_question(question):
    assert heuristic_plan(question).intent != "out_of_scope"


def test_short_follow_up_is_not_mistaken_for_out_of_scope():
    history = [{"question": "How is energy doing?",
                "plan": {"intent": "sector_analysis", "sector": "Energy",
                         "date_range": "all_time"}}]
    assert heuristic_plan("and infrastructure?", history).intent != "out_of_scope"


def test_unrelated_question_without_groq_is_declined_end_to_end():
    settings = make_settings()
    api = FakeMondayAPI()
    service = DataService(
        settings, client=MondayClient(settings, transport=api.transport(), max_retries=1),
        today=TODAY,
    )
    agent = BIAgent(settings=settings, data_service=service, llm=GroqLLM(settings, client=None))
    answer = agent.ask("what is the capital of France?", today=TODAY)
    assert api.request_count == 0
    assert "₹" not in answer.answer, "an unrelated question must not return figures"
    assert "glad to help" in answer.answer.lower()


# --- resilience regressions -------------------------------------------------

def test_narration_failure_does_not_escape_ask(monkeypatch):
    """A defect in the narrator must degrade, not crash the app.

    ``ResponseWriter`` already handles Groq errors; this covers everything else
    (an unexpected SDK error, a renderer bug on an unusual fact shape). The
    computed figures are correct at that point and must still reach the user.
    """
    agent, _ = build_agent(StubGroq(['{"intent":"pipeline_analysis","boards":["deals"]}']))

    def _boom(*args, **kwargs):
        raise RuntimeError("narrator exploded")

    monkeypatch.setattr(agent.writer, "write", _boom)

    answer = agent.ask("How's our pipeline?", today=TODAY)
    assert answer.narration_source == "error"
    assert answer.facts, "the computed facts must survive a narration failure"
    assert "could not be written up" in answer.answer


def test_leadership_narration_failure_does_not_escape_ask(monkeypatch):
    agent, _ = build_agent(StubGroq(['{"intent":"leadership_update","boards":["deals"]}']))
    monkeypatch.setattr(
        agent.writer, "write_leadership_update",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    answer = agent.ask("Prepare a leadership update.", today=TODAY)
    assert answer.narration_source == "error"
    assert answer.facts


def test_empty_board_is_reported_as_no_data_not_zero_deals():
    """An unreachable/empty board must not read as a real figure of zero."""
    agent, _ = build_agent(api=FakeMondayAPI(deals=[], work_orders=[]))
    agent.llm = GroqLLM(make_settings(groq_api_key=None))
    agent.writer = ResponseWriter(agent.llm)

    answer = agent.ask("How's our pipeline?", today=TODAY)
    assert "no usable rows" in answer.answer
    assert "0 open deals" not in answer.answer


# --- replying to a clarification --------------------------------------------

def _clarified(options):
    """A history whose last turn offered ``options``."""
    return [{
        "question": "How are we doing?",
        "answer": "Which would you like?",
        "plan": {"intent": "general_business_summary", "needs_clarification": True,
                 "clarification_options": options, "date_range": "all_time"},
    }]


OPTIONS = ["Sales / pipeline", "Revenue (won and billed)",
           "Operations / work orders", "Overall business health"]


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("1", "Sales / pipeline"),
        ("2.", "Revenue (won and billed)"),
        ("(3)", "Operations / work orders"),
        ("#4", "Overall business health"),
        ("option 2", "Revenue (won and billed)"),
        ("first", "Sales / pipeline"),
        ("the second one", "Revenue (won and billed)"),
        ("last", "Overall business health"),
        ("revenue", "Revenue (won and billed)"),
        ("9", None),                      # out of range
        ("how is mining doing?", None),    # a real question, not a selection
    ],
)
def test_clarification_reply_resolves_to_the_chosen_option(reply, expected):
    from agent.planner import resolve_clarification_reply

    assert resolve_clarification_reply(reply, _clarified(OPTIONS)) == expected


def test_a_selection_is_only_read_against_a_pending_clarification():
    from agent.planner import resolve_clarification_reply

    answered = [{"question": "x", "answer": "y",
                 "plan": {"intent": "pipeline_analysis", "needs_clarification": False}}]
    assert resolve_clarification_reply("1", answered) is None


def test_numbered_reply_is_honoured_instead_of_re_clarifying():
    """The defect: replying "1" to numbered options produced another question."""
    plan = heuristic_plan("1", _clarified(OPTIONS))
    assert not plan.needs_clarification
    assert plan.intent == "pipeline_analysis"

    operations = heuristic_plan("3", _clarified(OPTIONS))
    assert operations.intent in {"work_order_analysis", "operational_health"}


def test_the_agent_never_clarifies_twice_in_a_row():
    """One clarification is helpful; two is a loop the user cannot escape."""
    agent, _ = build_agent()
    history, clarifications = [], 0
    for reply in ["How are we doing?", "1", "2", "3", "4"]:
        answer = agent.ask(reply, history=history, today=TODAY)
        if answer.needs_clarification:
            clarifications += 1
            assert clarifications == 1 or not history[-1]["plan"]["needs_clarification"], (
                "clarified twice in succession"
            )
        history.append({"question": reply, "answer": answer.answer,
                        "plan": answer.plan.model_dump() if answer.plan else None})


def test_llm_planner_asking_again_is_overridden(monkeypatch):
    """Even if the model re-clarifies, the turn after a clarification answers."""
    agent, _ = build_agent(StubGroq([
        '{"intent":"general_business_summary","boards":["deals"],'
        '"needs_clarification":true,"clarification_question":"which?",'
        '"clarification_options":["a","b"]}'
    ]))
    plan = agent.planner.plan("pipeline please", _clarified(OPTIONS))
    assert not plan.needs_clarification


def test_bare_number_without_a_pending_question_asks_rather_than_guessing():
    answered = [{"question": "pipeline", "answer": "...",
                 "plan": {"intent": "pipeline_analysis", "needs_clarification": False}}]
    plan = heuristic_plan("2", answered)
    assert plan.needs_clarification
    assert "not sure what that refers to" in (plan.clarification_question or "")


def test_offered_options_reach_the_planner_prompt():
    """The model cannot resolve "1" unless it is told what was offered."""
    from agent.planner import _format_history

    rendered = _format_history(_clarified(OPTIONS))
    assert "1. Sales / pipeline" in rendered
    assert "2. Revenue (won and billed)" in rendered


def test_internal_field_slugs_never_reach_the_user_as_options():
    plan = QueryPlan(clarification_options=["pipeline_by_sector", "deal_details"])
    assert plan.clarification_options == ["Pipeline by sector", "Deal details"]
