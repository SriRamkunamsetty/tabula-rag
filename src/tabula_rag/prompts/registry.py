"""Versioned generation prompts.

Mirrors mini-challenge 2's prompt registry: prompts are named, immutable
versions rather than strings at the call site, so an evaluation run recorded
against one version stays meaningful after a later version ships.

The versions here form the ablation ladder the brief asks for — "leveraging
RAG to improve accuracy and reduce hallucination" is only demonstrable by
comparing against what the model does *without* that leverage:

* ``no-retrieval`` — the model answers from its own knowledge, no context at
  all. This is the delta baseline: since the corpus is a game invented for
  this project, a model that has never seen it should score close to zero
  here, and any real answer is either a lucky guess or a hallucination.
* ``naive`` — context is stuffed into the prompt with no citation
  requirement. This is what most RAG demos ship.
* ``cited`` *(default)* — every claim must carry a chunk id. This is the
  version whose output ``verification.py`` can actually check.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DEFAULT_VERSION",
    "PROMPTS",
    "PromptVersion",
    "answer_schema",
    "get_prompt",
    "list_versions",
]


@dataclass(frozen=True, slots=True)
class PromptVersion:
    """One generation strategy, identified by a stable version string."""

    version: str
    summary: str
    system: str
    requires_context: bool
    requires_citations: bool

    def render_user(self, query: str, context: str) -> str:
        """Fill the user turn for this prompt version."""
        if not self.requires_context:
            return (
                f"Question: {query}\n\n"
                "Answer from your own knowledge. There is no reference material "
                "provided for this question."
            )
        return (
            f"Reference material (each block is one citable chunk, labelled with its id):\n\n"
            f"{context}\n\n"
            f"Question: {query}"
        )


_CITED_SYSTEM = (
    "You are a rules oracle. You answer questions using ONLY the reference "
    "material provided in the user message — never your own outside "
    "knowledge, even if you believe you know the answer.\n\n"
    "Respond with a JSON object of this exact shape:\n"
    '{"claims": [{"text": "one factual sentence", "cited_chunk_ids": ["id1"]}], '
    '"abstained": false, "abstain_reason": null}\n\n'
    "Rules:\n"
    "1. Every claim's text must be a single factual sentence, and must cite "
    "at least one chunk id from the reference material that supports it.\n"
    "2. Never state a fact that is not directly supported by a cited chunk. "
    "If the reference material does not answer the question, set "
    '"abstained": true, give a one-sentence "abstain_reason", and return an '
    "empty claims list.\n"
    "3. Do not combine two unrelated facts into one claim — split them so "
    "each claim can be checked independently."
)

_NAIVE_SYSTEM = (
    "You are a helpful assistant. Use the reference material provided to "
    "answer the question as helpfully as possible.\n\n"
    'Respond with JSON: {"claims": [{"text": "...", "cited_chunk_ids": []}], '
    '"abstained": false, "abstain_reason": null}. Citations are optional.'
)

_NO_RETRIEVAL_SYSTEM = (
    "You are a helpful assistant. Answer the question as best you can from "
    "your own training.\n\n"
    'Respond with JSON: {"claims": [{"text": "...", "cited_chunk_ids": []}], '
    '"abstained": false, "abstain_reason": null}.'
)

NO_RETRIEVAL = PromptVersion(
    version="no-retrieval",
    summary="Baseline: answers from parametric knowledge alone, no context given.",
    system=_NO_RETRIEVAL_SYSTEM,
    requires_context=False,
    requires_citations=False,
)

NAIVE = PromptVersion(
    version="naive",
    summary="Context is provided but citations are not required or checked.",
    system=_NAIVE_SYSTEM,
    requires_context=True,
    requires_citations=False,
)

CITED = PromptVersion(
    version="cited",
    summary="Every claim must cite a chunk id; the only version verification.py can check.",
    system=_CITED_SYSTEM,
    requires_context=True,
    requires_citations=True,
)

PROMPTS: dict[str, PromptVersion] = {p.version: p for p in (NO_RETRIEVAL, NAIVE, CITED)}
DEFAULT_VERSION = CITED.version


def get_prompt(version: str | None = None) -> PromptVersion:
    """Return a prompt version, falling back to the current default."""
    key = version or DEFAULT_VERSION
    if key not in PROMPTS:
        raise KeyError(
            f"unknown prompt version '{key}'; available: {', '.join(sorted(PROMPTS))}"
        )
    return PROMPTS[key]


def list_versions() -> list[tuple[str, str]]:
    """Return ``(version, summary)`` for every registered prompt."""
    return [(p.version, p.summary) for p in PROMPTS.values()]


def answer_schema() -> dict[str, object]:
    """JSON Schema for the cited-answer contract, passed as ``guided_json``."""
    return {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "cited_chunk_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["text", "cited_chunk_ids"],
                    "additionalProperties": False,
                },
            },
            "abstained": {"type": "boolean"},
            "abstain_reason": {"type": ["string", "null"]},
        },
        "required": ["claims", "abstained", "abstain_reason"],
        "additionalProperties": False,
    }
