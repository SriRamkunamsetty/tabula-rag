"""Domain models for the TABULA RAG rules oracle.

The central discipline of this service is that a claim in an answer is
worthless without a chunk it can be checked against. Every model here exists
to keep that link intact from ingestion through to the final answer: a
:class:`Chunk` always knows which document and section it came from, a
:class:`Claim` always names the chunk(s) it cites, and an :class:`AnswerResult`
can never contain a claim without a verification outcome attached to it.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "AnswerResult",
    "Chunk",
    "Claim",
    "ClaimStatus",
    "IngestRequest",
    "IngestResult",
    "QueryRequest",
    "RetrievedChunk",
    "SourceConflict",
    "UsageStats",
]


class _Strict(BaseModel):
    """Base model with production-safe defaults."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Chunk(_Strict):
    """One retrievable unit of a source document.

    ``chunk_id`` is stable across re-ingestion of the same document (derived
    from document id + section + position), which is what lets citations in a
    previously generated answer still resolve after the corpus is re-indexed.
    """

    chunk_id: str = Field(min_length=1, max_length=128)
    document_id: str = Field(min_length=1, max_length=128)
    document_title: str = ""
    section: str = ""
    position: int = Field(ge=0)
    text: str = Field(min_length=1)

    @property
    def citation_label(self) -> str:
        """Human-readable pointer used when an answer cites this chunk."""
        where = (
            f"{self.document_title} §{self.section}" if self.section else self.document_title
        )
        return where or self.document_id


class RetrievedChunk(_Strict):
    """A chunk plus how strongly retrieval ranked it for a given query."""

    chunk: Chunk
    lexical_rank: int | None = None
    dense_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None


class ClaimStatus(enum.StrEnum):
    """Outcome of checking one claim against the chunk(s) it cites."""

    SUPPORTED = "supported"
    """Cited a real chunk, and that chunk's text covers the claim."""
    UNSUPPORTED = "unsupported"
    """Cited a real chunk, but that chunk's text does not cover the claim."""
    UNCITED = "uncited"
    """Context was available to cite, but the claim cited nothing."""
    UNGROUNDED = "ungrounded"
    """No context was available at all (a no-retrieval query) -- there was
    nothing to check this claim against, so its truth rests entirely on the
    model's own unverified assertion. Distinct from SUPPORTED on purpose:
    an ungrounded claim may still ship (see Claim.is_trustworthy) but must
    never be mistaken for a verified one."""


class Claim(_Strict):
    """One atomic statement from a generated answer, with its citations."""

    text: str = Field(min_length=1)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    status: ClaimStatus = ClaimStatus.UNCITED
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    supporting_quote: str | None = None

    @property
    def is_trustworthy(self) -> bool:
        """True when this claim may be surfaced to the caller as-is.

        SUPPORTED and UNGROUNDED both ship: a verified claim because it
        checked out, an ungrounded one because there was nothing to check it
        against in the first place, and withholding it would conflate "no
        retrieval was requested" with "retrieval found a problem." The two
        remain distinguishable by status for any caller that cares -- see
        AnswerResult.grounded_ratio, which counts SUPPORTED only.
        """
        return self.status in (ClaimStatus.SUPPORTED, ClaimStatus.UNGROUNDED)


class SourceConflict(_Strict):
    """Two retrieved chunks that appear to state different values for the same fact."""

    claim_text: str
    value_a: str
    source_a: str
    value_b: str
    source_b: str


class UsageStats(_Strict):
    """Token and timing accounting for one query."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieved_chunks: int = 0
    latency_ms: float = 0.0
    model: str = "unknown"

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens

    def cost_usd(self, gpu_hourly_rate: float, throughput_tokens_per_s: float) -> float:
        """Amortised GPU cost, derived the same way as in mini-challenge 2."""
        if throughput_tokens_per_s <= 0:
            return 0.0
        return (gpu_hourly_rate / 3600.0) * (self.total_tokens / throughput_tokens_per_s)


class AnswerResult(_Strict):
    """The response envelope returned by the service."""

    query: str
    answer: str
    abstained: bool
    claims: list[Claim] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    conflicts: list[SourceConflict] = Field(default_factory=list)
    usage: UsageStats = Field(default_factory=UsageStats)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _abstained_answer_has_no_unverified_claims(self) -> AnswerResult:
        if not self.abstained and any(c.status is ClaimStatus.UNCITED for c in self.claims):
            raise ValueError("a served (non-abstained) answer must not contain uncited claims")
        return self

    @property
    def unsupported_sentence_ratio(self) -> float:
        """Share of claims that failed verification — the USR metric."""
        if not self.claims:
            return 0.0
        bad = sum(1 for c in self.claims if not c.is_trustworthy)
        return bad / len(self.claims)

    @property
    def faithfulness(self) -> float:
        """1 - unsupported_sentence_ratio.

        Reads as "how much of this answer was not actively refused." An
        UNGROUNDED claim counts as faithful here (it was never checked, so
        it cannot have failed a check) -- use :attr:`grounded_ratio` for the
        stricter question of how much was actually verified.
        """
        return 1.0 - self.unsupported_sentence_ratio

    @property
    def grounded_ratio(self) -> float:
        """Share of claims that were actually checked against a source and passed.

        Strictly narrower than :attr:`faithfulness`: an UNGROUNDED claim
        (produced with no retrieval, so nothing existed to check it against)
        counts as faithful but does NOT count here. A no-retrieval answer
        reports ``faithfulness == 1.0`` (nothing was refused) alongside
        ``grounded_ratio == 0.0`` (nothing was verified either) -- the two
        numbers together are what make "this answer shipped, but zero of it
        was checked against anything" visible to a caller at a glance.
        """
        if not self.claims:
            return 0.0
        supported = sum(1 for c in self.claims if c.status is ClaimStatus.SUPPORTED)
        return supported / len(self.claims)


class IngestRequest(_Strict):
    """Input contract for ``POST /v1/ingest``."""

    document_id: str = Field(min_length=1, max_length=128)
    title: str = ""
    text: str = Field(min_length=1)


class IngestResult(_Strict):
    """Response for an ingest call."""

    document_id: str
    chunks_indexed: int
    warnings: list[str] = Field(default_factory=list)


class QueryRequest(_Strict):
    """Input contract for ``POST /v1/query``."""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    use_retrieval: bool = True
    """False runs the "no-retrieval" ablation rung — see eval/harness.py."""
    prompt_version: str | None = None
