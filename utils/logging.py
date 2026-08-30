"""Application logging.

Logs are for developers. User-facing surfaces never receive stack traces, and no
secret is ever written to a log record (see :func:`redact`).
"""
from __future__ import annotations

import logging
import os
import re
import sys

_CONFIGURED = False

# Anything that looks like a Monday / Groq credential is masked before it can
# reach a log handler.
_SECRET_PATTERNS = [
    re.compile(r"eyJ[A-Za-z0-9_\-\.]{20,}"),          # Monday JWT-style tokens
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),               # Groq API keys
    re.compile(r"(?i)(authorization|api[_-]?key|token)\s*[:=]\s*\S+"),
]


def redact(text: str) -> str:
    """Replace anything that resembles a credential with ``***``."""
    out = str(text)
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub("***redacted***", out)
    return out


class _RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            try:
                record.args = tuple(redact(a) if isinstance(a, str) else a for a in record.args)
            except TypeError:  # dict-style args
                pass
        return True


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once. Safe to call repeatedly."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s | %(message)s")
    )
    handler.addFilter(_RedactingFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level_name, logging.INFO))
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
