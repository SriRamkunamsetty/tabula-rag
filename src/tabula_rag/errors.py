"""Typed errors, one HTTP status each — same pattern as mini-challenge 2."""

from __future__ import annotations


class TabulaRagError(Exception):
    """Base class for all service errors."""

    code = "internal_error"
    http_status = 500

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        """Create the error with a human message and optional machine detail."""
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, str | None]:
        """Render the error as a stable JSON body."""
        return {"code": self.code, "message": self.message, "detail": self.detail}


class ConfigurationError(TabulaRagError):
    """The deployment is misconfigured; retrying will not help."""

    code = "configuration_error"
    http_status = 500


class DocumentNotFoundError(TabulaRagError):
    """A referenced document id is not in the index."""

    code = "document_not_found"
    http_status = 404


class EmptyCorpusError(TabulaRagError):
    """A query was issued before any document was ingested."""

    code = "empty_corpus"
    http_status = 409


class LLMTimeoutError(TabulaRagError):
    """The language model did not answer inside the configured budget."""

    code = "llm_timeout"
    http_status = 504


class LLMProtocolError(TabulaRagError):
    """The language model answered, but not in the contracted shape."""

    code = "llm_protocol_error"
    http_status = 502


class CircuitOpenError(TabulaRagError):
    """The upstream model is failing; requests are being shed deliberately."""

    code = "upstream_unavailable"
    http_status = 503
