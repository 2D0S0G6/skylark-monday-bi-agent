"""List the Groq models this account can actually use.

    python -m tools.list_models

Groq retires models periodically, so a hard-coded default eventually 404s. Run
this to find a current model ID, then set `GROQ_MODEL` in `.env`.
"""
from __future__ import annotations

import sys

from config import get_settings
from utils.logging import configure_logging


def main() -> int:
    configure_logging()
    settings = get_settings()
    if not settings.groq_api_key:
        print("!! GROQ_API_KEY is not configured.")
        return 1

    try:
        from groq import Groq
    except ImportError:
        print("!! The groq package is not installed: pip install -r requirements.txt")
        return 1

    client = Groq(api_key=settings.groq_api_key)
    try:
        models = sorted(m.id for m in client.models.list().data)
    except Exception as exc:  # noqa: BLE001 - surface any auth/network problem plainly
        print(f"!! Could not list models: {exc}")
        return 2

    # Audio and guard models cannot serve chat completions.
    excluded = ("whisper", "prompt-guard", "orpheus", "tts", "safeguard")
    chat_models = [m for m in models if not any(x in m.lower() for x in excluded)]

    print(f"Chat-capable models available to this account ({len(chat_models)}):\n")
    for model in chat_models:
        marker = "  <- current GROQ_MODEL" if model == settings.groq_model else ""
        print(f"  {model}{marker}")

    if settings.groq_model not in models:
        print(
            f"\n!! GROQ_MODEL is set to '{settings.groq_model}', which this account "
            f"cannot use. Set GROQ_MODEL in .env to one of the IDs above."
        )
        return 3
    print(f"\nGROQ_MODEL='{settings.groq_model}' is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
