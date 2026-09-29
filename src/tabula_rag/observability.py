"""Structured logging, request correlation and Prometheus metrics.

Same design as mini-challenge 2's observability module: JSON logs by default,
a request id bound once per request via a context variable, and metrics with
deliberately low label cardinality.
"""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Mapping, MutableMapping
from contextvars import ContextVar
from typing import Any

import structlog
from prometheus_client import Counter, Gauge, Histogram

__all__ = [
    "ANSWERS_ABSTAINED",
    "CLAIMS_TOTAL",
    "CLAIMS_UNSUPPORTED",
    "CONFLICTS_DETECTED",
    "INFLIGHT",
    "LLM_FAILURES",
    "LLM_LATENCY",
    "QUERIES",
    "QUERY_LATENCY",
    "bind_request_id",
    "configure_logging",
    "current_request_id",
    "get_logger",
    "new_request_id",
]

_request_id: ContextVar[str] = ContextVar("request_id", default="-")

QUERIES = Counter("tabula_rag_queries_total", "Queries handled.", ["outcome"])
QUERY_LATENCY = Histogram(
    "tabula_rag_query_latency_seconds",
    "End-to-end query latency.",
    buckets=(0.25, 0.5, 1, 2, 4, 8, 16, 32),
)
LLM_LATENCY = Histogram(
    "tabula_rag_llm_latency_seconds",
    "Latency of a single LLM call.",
    ["operation"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 16),
)
LLM_FAILURES = Counter("tabula_rag_llm_failures_total", "Upstream model failures.", ["reason"])
CLAIMS_TOTAL = Counter("tabula_rag_claims_total", "Claims produced.", ["status"])
CLAIMS_UNSUPPORTED = Counter(
    "tabula_rag_claims_unsupported_total", "Claims that failed the coverage gate."
)
ANSWERS_ABSTAINED = Counter(
    "tabula_rag_answers_abstained_total", "Answers withheld.", ["reason"]
)
CONFLICTS_DETECTED = Counter(
    "tabula_rag_conflicts_detected_total", "Cross-source numeric/date disagreements flagged."
)
INFLIGHT = Gauge("tabula_rag_inflight_queries", "Queries in flight.")


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure structlog and the stdlib root logger once, at startup."""
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=getattr(logging, level))
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _inject_request_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _inject_request_id(
    _logger: Any, _name: str, event: MutableMapping[str, Any]
) -> Mapping[str, Any]:
    event.setdefault("request_id", _request_id.get())
    return event


def get_logger(name: str) -> Any:
    """Return a bound logger for ``name``."""
    return structlog.get_logger(name)


def new_request_id() -> str:
    """Generate a short, collision-resistant request id."""
    return uuid.uuid4().hex[:16]


def bind_request_id(request_id: str) -> None:
    """Bind ``request_id`` for the current async context."""
    _request_id.set(request_id)


def current_request_id() -> str:
    """Return the request id bound to the current context."""
    return _request_id.get()
