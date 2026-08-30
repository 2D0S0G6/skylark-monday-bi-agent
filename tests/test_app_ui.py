"""UI smoke tests using Streamlit's AppTest harness (no browser, no live APIs)."""
from __future__ import annotations

import pathlib

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import FakeMondayAPI

APP_PATH = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def _clear_config_cache(monkeypatch):
    import config

    config.reset_settings_cache()
    yield
    config.reset_settings_cache()


def test_app_shows_setup_screen_when_unconfigured(monkeypatch):
    for key in ("MONDAY_API_TOKEN", "MONDAY_DEALS_BOARD_ID",
                "MONDAY_WORK_ORDERS_BOARD_ID", "GROQ_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("config._from_streamlit_secrets", lambda key: None)

    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    assert not app.exception
    assert any("not configured yet" in w.value for w in app.warning)
    assert any("MONDAY_API_TOKEN" in m.value for m in app.markdown)


def test_app_renders_an_answer_against_the_fake_board(monkeypatch):
    """Drive the whole UI with a mocked Monday transport and no Groq key."""
    api = FakeMondayAPI()

    monkeypatch.setenv("MONDAY_API_TOKEN", "test-token")
    monkeypatch.setenv("MONDAY_DEALS_BOARD_ID", "1111")
    monkeypatch.setenv("MONDAY_WORK_ORDERS_BOARD_ID", "2222")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("config._from_streamlit_secrets", lambda key: None)

    import agent.data_service as data_service_module
    from monday.client import MondayClient

    def _patched_client(self):
        if self._client is None:
            self._client = MondayClient(self.settings, transport=api.transport(), max_retries=1)
        return self._client

    monkeypatch.setattr(
        data_service_module.DataService, "client", property(_patched_client)
    )

    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["pending_question"] = "How's our pipeline looking?"
    app.run()

    assert not app.exception
    body = "\n".join(m.value for m in app.markdown)
    assert "₹" in body
    assert "Skylark Drones" in "\n".join(t.value for t in app.title)


def test_example_questions_cover_the_required_topics():
    import app as app_module

    questions = " ".join(app_module.EXAMPLE_QUESTIONS).lower()
    assert len(app_module.EXAMPLE_QUESTIONS) >= 10
    for topic in ("pipeline", "sector", "risk", "work order", "delayed",
                  "leadership update", "opportunit"):
        assert topic in questions, f"missing an example question about {topic}"


def test_headline_metric_cards_only_use_computed_values():
    import app as app_module

    frame = app_module._flatten_rows([
        {"sector": "Mining", "value": {"amount": 1.0, "display": "₹1.00 Cr"},
         "risk_signals": ["a", "b"]}
    ])
    assert isinstance(frame, pd.DataFrame)
    assert frame.loc[0, "value"] == "₹1.00 Cr"
    assert frame.loc[0, "risk_signals"] == "a, b"


def test_setup_screen_diagnoses_each_missing_setting(monkeypatch):
    """The setup screen must show which settings are visible and from where."""
    for key in ("MONDAY_API_TOKEN", "MONDAY_DEALS_BOARD_ID",
                "MONDAY_WORK_ORDERS_BOARD_ID", "GROQ_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("config._from_streamlit_secrets", lambda key: None)

    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    assert not app.exception

    # A per-setting status table, not just prose.
    assert app.dataframe, "the setup screen should render a diagnosis table"
    table = app.dataframe[0].value
    assert set(table["Setting"]) == {
        "MONDAY_API_TOKEN", "MONDAY_DEALS_BOARD_ID",
        "MONDAY_WORK_ORDERS_BOARD_ID", "GROQ_API_KEY",
    }
    assert (table["Status"] == "❌ missing").all()


def test_diagnose_reports_the_source_of_each_setting(monkeypatch):
    import config

    # Start from a clean slate so a developer's real .env cannot mask the result.
    for key in config.REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MONDAY_API_TOKEN", "from-env")
    monkeypatch.setattr(
        "config._from_streamlit_secrets",
        lambda key: "from-secrets" if key == "GROQ_API_KEY" else None,
    )
    report = config.diagnose()
    by_key = {v["key"]: v for v in report["variables"]}
    assert by_key["MONDAY_API_TOKEN"]["source"] == "environment / .env"
    assert by_key["GROQ_API_KEY"]["source"] == "Streamlit secrets"
    # The two board IDs are supplied by neither source here.
    assert by_key["MONDAY_DEALS_BOARD_ID"]["source"] == "not found"
    assert report["all_present"] is False


def test_diagnose_never_exposes_a_secret_value(monkeypatch):
    import json

    import config

    monkeypatch.setenv("MONDAY_API_TOKEN", "super-secret-value")
    report = config.diagnose()
    assert "super-secret-value" not in json.dumps(report)


# --- output escaping --------------------------------------------------------

def test_metric_cards_escape_untrusted_text():
    """KPI cards are rendered with ``unsafe_allow_html``, so they must escape.

    Card text is derived from board data, which the app does not control.
    """
    import app as app_module

    card = app_module._metric_card(
        "<b>label</b>", "<img src=x onerror=alert(1)>", "<script>alert(2)</script>"
    )
    assert "<img" not in card
    assert "<script>" not in card
    assert "&lt;img src=x onerror=alert(1)&gt;" in card


def test_analysis_pills_escape_planner_output(monkeypatch):
    """``plan.sector`` is a free string produced by the LLM and must be escaped."""
    import app as app_module
    from agent.orchestrator import AgentAnswer
    from agent.schemas import QueryPlan

    rendered: list[str] = []
    monkeypatch.setattr(app_module.st, "markdown", lambda body, **kw: rendered.append(str(body)))
    monkeypatch.setattr(app_module.st, "caption", lambda *a, **kw: None)
    monkeypatch.setattr(app_module.st, "json", lambda *a, **kw: None)
    monkeypatch.setattr(app_module.st, "expander", lambda *a, **kw: _NullContext())

    plan = QueryPlan(intent="pipeline_analysis", boards=["deals"],
                     sector="<script>alert(1)</script>", source="llm")
    app_module.render_details(AgentAnswer(answer="x", plan=plan, facts={}))

    body = "\n".join(rendered)
    assert "<script>" not in body
    assert "&lt;script&gt;" in body


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_reruns_do_not_repeat_the_question_or_refetch(monkeypatch):
    """A Streamlit rerun must not re-run the agent or re-hit Monday.com.

    Every widget interaction (opening an expander, clicking a sidebar button)
    replays the whole script, so the answer path has to be driven only by a fresh
    submission -- not by whatever is left in session state.
    """
    api = FakeMondayAPI()

    monkeypatch.setenv("MONDAY_API_TOKEN", "test-token")
    monkeypatch.setenv("MONDAY_DEALS_BOARD_ID", "1111")
    monkeypatch.setenv("MONDAY_WORK_ORDERS_BOARD_ID", "2222")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("config._from_streamlit_secrets", lambda key: None)

    import agent.data_service as data_service_module
    from monday.client import MondayClient

    def _patched_client(self):
        if self._client is None:
            self._client = MondayClient(self.settings, transport=api.transport(), max_retries=1)
        return self._client

    monkeypatch.setattr(
        data_service_module.DataService, "client", property(_patched_client)
    )

    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["pending_question"] = "How's our pipeline looking?"
    app.run()

    requests_after_answer = api.request_count
    assert len(app.session_state["history"]) == 1

    app.run()  # a plain rerun, as any widget interaction would cause
    app.run()

    assert not app.exception
    assert len(app.session_state["history"]) == 1, "the question was answered again"
    assert api.request_count == requests_after_answer, "the boards were refetched"
