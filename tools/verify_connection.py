"""Verify the Monday.com and Groq configuration end to end.

    python -m tools.verify_connection

Prints what the agent can see: board names, item counts, the resolved column
mapping, the data-quality summary and a sample computed answer. No secret is
printed. Useful as a first step after configuring `.env`, and as the smoke test
before deploying.
"""
from __future__ import annotations

import sys

import pandas as pd

from agent.data_service import DataService
from agent.llm import GroqLLM
from agent.orchestrator import BIAgent
from config import get_settings
from monday.client import MondayError
from utils.logging import configure_logging


def main() -> int:
    configure_logging()
    settings = get_settings()

    print("Configuration")
    print("-" * 70)
    for key, value in settings.safe_summary().items():
        print(f"  {key:<22} {value}")
    missing = settings.missing_settings()
    if [m for m in missing if m != "GROQ_API_KEY"]:
        print(f"\n!! Missing required settings: {', '.join(missing)}")
        return 1

    print("\nFetching boards from Monday.com…")
    service = DataService(settings)
    try:
        data = service.get_data(force_refresh=True)
    except MondayError as exc:
        print(f"\n!! {exc.user_message}")
        print(f"   (developer detail: {exc})")
        return 2

    for label, dataset, mapping in (
        ("Deals", data.deals, data.deals_mapping),
        ("Work Orders", data.work_orders, data.work_orders_mapping),
    ):
        print(f"\n{label}: '{mapping.board_name}' (ID {mapping.board_id}) — "
              f"{len(dataset.frame)} rows")
        print("  resolved columns:")
        for entry in mapping.describe():
            print(f"    {entry['field']:<22} -> {entry['monday_column']!r} "
                  f"[{entry['type']}, matched by {entry['matched_by']}]")
        if mapping.unmapped_fields:
            print(f"  unmapped fields: {', '.join(mapping.unmapped_fields)}")
        if mapping.unused_columns:
            print(f"  board columns not used: {', '.join(mapping.unused_columns[:10])}")
        print("  data quality:")
        for issue in dataset.quality.top(6):
            print(f"    [{issue.severity.value}] {issue.message}")

    for warning in data.warnings:
        print(f"\n!! {warning}")

    print("\nRunning a sample question through the full pipeline…")
    agent = BIAgent(settings=settings, data_service=service, llm=GroqLLM(settings))
    print(f"  Groq available: {agent.llm.available}")
    answer = agent.ask("How's our pipeline looking this quarter?",
                       today=pd.Timestamp.today().normalize())
    print("-" * 70)
    print(answer.answer)
    print("-" * 70)
    print(f"  planner: {answer.plan.source if answer.plan else 'n/a'} | "
          f"narration: {answer.narration_source}")
    validation = (answer.facts or {}).get("validation", {})
    print(f"  validation: {'passed' if validation.get('passed') else validation}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
