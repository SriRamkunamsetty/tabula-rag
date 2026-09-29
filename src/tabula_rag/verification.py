"""Claim verification — the gate that keeps generation honest.

A citation-enforced prompt (see ``prompts/registry.py``) makes the generator
attach chunk ids to every sentence. That is necessary but not sufficient: a
model can cite a real chunk and still say something that chunk doesn't
actually support. This module is the check that catches that — it is the
RAG equivalent of mini-challenge 2's grounding veto, applied to full
sentences instead of single field values.

The check is a **coverage score**: what fraction of the claim's meaningful
(non-stopword) terms are also present in the text of the chunk(s) it cites.
This is deliberately the same family of metric as the "Unsupported Sentence
Ratio" used in the RAG evaluation literature — a lexical-overlap proxy for
entailment, cheap enough to run on every claim with no extra model call, and
honest about being a proxy rather than a true NLI judgment (see
``docs/adr/003-lexical-verification.md`` for that trade-off argued in full).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tabula_rag.models import Chunk

__all__ = ["CoverageReport", "content_terms", "score_claim_coverage", "split_into_claims"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

# A short, domain-agnostic stopword list. Kept intentionally small: the goal
# is to stop "the", "a", "is" from diluting the coverage score, not to build
# a linguistically complete stopword list — over-aggressive filtering would
# strip exactly the content words (numbers, named entities) coverage most
# needs to check.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "but",
        "if",
        "then",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "by",
        "with",
        "from",
        "into",
        "up",
        "down",
        "not",
        "no",
        "do",
        "does",
        "did",
        "can",
        "may",
        "must",
        "will",
        "shall",
        "has",
        "have",
        "had",
    }
)


def content_terms(text: str) -> set[str]:
    """Lowercase, stopword-filtered, single-character-filtered term set."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1}


def split_into_claims(answer: str) -> list[str]:
    """Split a generated answer into sentence-level candidate claims.

    A simple punctuation-boundary split is used rather than a model call: the
    generator is prompted (see ``prompts/registry.py``) to produce one claim
    per line with an explicit citation marker already attached, so by the
    time text reaches this function it is already close to one sentence per
    line. This function is the fallback for free-form text and for the
    no-citation-prompt ablation rung, where no such structure exists yet.
    """
    text = answer.strip()
    if not text:
        return []
    pieces = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in pieces if p.strip()]


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Result of checking one claim against its cited chunk(s)."""

    score: float
    supported: bool
    best_chunk_id: str | None
    supporting_quote: str | None


def score_claim_coverage(
    claim_text: str,
    cited_chunks: list[Chunk],
    *,
    threshold: float,
) -> CoverageReport:
    """Score how well ``cited_chunks`` cover the content terms in a claim.

    Coverage is computed against each cited chunk independently and the best
    result is kept — a claim that draws on one chunk should not be penalised
    for a second, less relevant citation dragging its score down. A claim
    citing zero chunks scores 0.0 unconditionally: there is nothing to check
    it against, which is exactly the uncited case the schema in ``models.py``
    refuses to let a served answer contain.

    Args:
        claim_text: The candidate sentence.
        cited_chunks: The chunk(s) the generator attached to this claim.
        threshold: Coverage at or above this value counts as supported.

    Returns:
        A :class:`CoverageReport`. When supported, ``supporting_quote`` is the
        sentence from the winning chunk with the highest term overlap — the
        nearest thing to "point at the exact line" this lexical method can
        offer, for a reviewer to fact-check by eye.
    """
    claim_terms = content_terms(claim_text)
    if not claim_terms or not cited_chunks:
        return CoverageReport(
            score=0.0, supported=False, best_chunk_id=None, supporting_quote=None
        )

    best_score = 0.0
    best_chunk_id: str | None = None
    best_quote: str | None = None

    for chunk in cited_chunks:
        chunk_terms = content_terms(chunk.text)
        if not chunk_terms:
            continue
        overlap = len(claim_terms & chunk_terms) / len(claim_terms)
        if overlap > best_score:
            best_score = overlap
            best_chunk_id = chunk.chunk_id
            best_quote = _best_matching_sentence(claim_terms, chunk.text)

    return CoverageReport(
        score=round(best_score, 4),
        supported=best_score >= threshold,
        best_chunk_id=best_chunk_id,
        supporting_quote=best_quote,
    )


def _best_matching_sentence(claim_terms: set[str], chunk_text: str) -> str | None:
    """Return the sentence of ``chunk_text`` with the most claim-term overlap."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(chunk_text) if s.strip()]
    if not sentences:
        return chunk_text.strip()[:200] or None
    best_sentence, best_overlap = None, -1
    for sentence in sentences:
        overlap = len(claim_terms & content_terms(sentence))
        if overlap > best_overlap:
            best_overlap, best_sentence = overlap, sentence
    return best_sentence
