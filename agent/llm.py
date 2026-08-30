"""Thin Groq client wrapper with typed failures and JSON-mode support.

Isolating Groq here means the planner and narrator can be unit-tested with a
stub, and a Groq outage degrades the app to deterministic behaviour instead of
crashing it.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from config import Settings, get_settings
from utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "LLMError",
    "LLMUnavailableError",
    "LLMModelNotFoundError",
    "GroqLLM",
    "extract_json",
]


class LLMError(Exception):
    """A Groq call failed. Callers degrade gracefully rather than surfacing this."""

    user_message = (
        "The language model is temporarily unavailable, so this answer is based on "
        "the computed figures without the usual narrative polish."
    )


class LLMUnavailableError(LLMError):
    user_message = (
        "Groq is not configured. Set GROQ_API_KEY to enable natural-language "
        "question understanding and executive narration."
    )


class LLMModelNotFoundError(LLMError):
    """The configured GROQ_MODEL does not exist on this account.

    Worth its own type: Groq retires models, and a silent fallback to keyword
    planning would hide a one-line configuration fix.
    """

    user_message = (
        "The configured GROQ_MODEL is not available on this Groq account. "
        "Run `python -m tools.list_models` to see the models you can use, then set "
        "GROQ_MODEL in your .env (or Streamlit secrets)."
    )


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_ms: int


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from a model response.

    Handles raw JSON, fenced JSON, and JSON with leading/trailing prose. Returns
    ``None`` when nothing parseable is found; callers then fall back.
    """
    if not text:
        return None
    candidates: list[str] = []
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate.strip())
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


class GroqLLM:
    """Wrapper around ``groq.Groq`` chat completions."""

    def __init__(self, settings: Settings | None = None, *, client=None) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._import_error: str | None = None
        #: Set when a call fails for a reason the operator should act on.
        self.last_error: str | None = None
        if client is None and self.settings.groq_api_key:
            try:
                from groq import Groq  # noqa: PLC0415 - optional at import time

                self._client = Groq(api_key=self.settings.groq_api_key)
            except Exception as exc:  # noqa: BLE001 - SDK/network issues must not crash import
                self._import_error = str(exc)
                logger.error("Could not initialise the Groq client: %s", exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1400,
        retries: int = 2,
    ) -> LLMResponse:
        """Run a chat completion, retrying transient failures."""
        if not self._client:
            raise LLMUnavailableError(self._import_error or "GROQ_API_KEY is not configured")

        kwargs: dict = {
            "model": self.settings.groq_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            started = time.perf_counter()
            try:
                completion = self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # the Groq SDK raises many error types
                last_error = exc
                message = str(exc).lower()
                if "model_not_found" in message or "does not exist" in message:
                    self.last_error = f"model '{self.settings.groq_model}' is unavailable"
                    logger.error(
                        "GROQ_MODEL '%s' is not available on this account", self.settings.groq_model
                    )
                    raise LLMModelNotFoundError(str(exc)) from exc
                if "json" in message and json_mode:
                    # Some models reject response_format; retry without it.
                    kwargs.pop("response_format", None)
                elif attempt < retries:
                    time.sleep(0.6 * (attempt + 1))
                logger.warning("Groq call failed (attempt %s): %s", attempt + 1, exc)
                continue

            latency = int((time.perf_counter() - started) * 1000)
            try:
                text = completion.choices[0].message.content or ""
            except (AttributeError, IndexError) as exc:  # pragma: no cover - defensive
                raise LLMError(f"Unexpected Groq response shape: {exc}") from exc
            return LLMResponse(text=text, model=self.settings.groq_model, latency_ms=latency)

        self.last_error = str(last_error)
        raise LLMError(f"Groq request failed after {retries + 1} attempts: {last_error}")
