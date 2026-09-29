"""The RAG pipeline.

One query flows through five stages:

1. **Retrieve.** Hybrid (BM25 + dense) search against the corpus, fused by
   reciprocal rank. Skipped entirely when ``request.use_retrieval`` is
   false, which is what makes the no-retrieval ablation rung possible.
2. **Rerank.** The fused top-k is narrowed to the generation context by a
   cross-encoder reranker, which reads query and candidate text together
   rather than relying on rank alone.
3. **Generate.** The LLM answers using only the reranked context, under a
   citation-enforced, schema-locked prompt.
4. **Verify.** Every claim is checked for coverage against the chunk(s) it
   cited (see ``verification.py``). Claims that fail are never surfaced.
5. **Assemble.** If too large a share of claims failed verification, or
   retrieval found nothing, the service abstains outright rather than
   returning a thin, mostly-unsupported answer. Cross-source conflicts
   (``conflicts.py``) are attached alongside the answer regardless of
   outcome, since a conflict is useful information even in an abstention.

This is the RAG mirror of mini-challenge 2's ``OCRService.extract``: neither
service trusts a single model call to police itself, and both would rather
say less than say something ungrounded.
"""

from __future__ import annotations

import time
from typing import Any

from tabula_rag.config import Settings
from tabula_rag.conflicts import find_conflicts
from tabula_rag.corpus_index import CorpusIndex
from tabula_rag.errors import EmptyCorpusError, LLMProtocolError
from tabula_rag.llm.client import LLMClient
from tabula_rag.models import (
    AnswerResult,
    Chunk,
    Claim,
    ClaimStatus,
    QueryRequest,
    RetrievedChunk,
    UsageStats,
)
from tabula_rag.observability import (
    ANSWERS_ABSTAINED,
    CLAIMS_TOTAL,
    CLAIMS_UNSUPPORTED,
    CONFLICTS_DETECTED,
    get_logger,
)
from tabula_rag.prompts.registry import answer_schema, get_prompt
from tabula_rag.retrieval.embeddings import EmbeddingClient
from tabula_rag.retrieval.reranker import RerankerClient
from tabula_rag.verification import score_claim_coverage

__all__ = ["RAGService"]

_log = get_logger(__name__)


class RAGService:
    """Coordinates ingestion and the retrieve -> rerank -> generate -> verify loop."""

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        embeddings: EmbeddingClient,
        reranker: RerankerClient,
        corpus: CorpusIndex | None = None,
    ) -> None:
        """Wire the pipeline to its model clients and a corpus index."""
        self._settings = settings
        self._llm = llm
        self._reranker = reranker
        self._corpus = corpus or CorpusIndex(settings, embeddings)

    @property
    def corpus(self) -> CorpusIndex:
        """The corpus index backing this service."""
        return self._corpus

    async def ingest(self, document_id: str, text: str, title: str = "") -> int:
        """Chunk and index one document. Returns the number of chunks added."""
        chunks = await self._corpus.ingest(document_id, text, title=title)
        _log.info("document_ingested", document_id=document_id, chunks=len(chunks))
        return len(chunks)

    async def query(self, request: QueryRequest) -> AnswerResult:
        """Answer one question, end to end.

        Raises:
            EmptyCorpusError: Retrieval was requested but the corpus has no
                documents ingested at all -- a query at this point cannot
                mean anything, so it is an error rather than a silent
                abstention.
        """
        started = time.perf_counter()
        top_k = request.top_k or self._settings.retrieval_top_k
        usage = UsageStats(model=self._settings.llm_model)
        warnings: list[str] = []

        retrieved: list[RetrievedChunk] = []
        if request.use_retrieval:
            if len(self._corpus) == 0:
                raise EmptyCorpusError(
                    "no documents have been ingested",
                    detail="call POST /v1/ingest before querying",
                )
            retrieved = await self._corpus.hybrid_search(request.query, top_k=top_k)
            retrieved = await self._rerank(request.query, retrieved)

        conflicts = find_conflicts(retrieved) if retrieved else []
        if conflicts:
            CONFLICTS_DETECTED.inc(len(conflicts))

        if request.use_retrieval and not retrieved:
            usage.latency_ms = round((time.perf_counter() - started) * 1000, 2)
            ANSWERS_ABSTAINED.labels(reason="no_retrieval_results").inc()
            return AnswerResult(
                query=request.query,
                answer="",
                abstained=True,
                claims=[],
                retrieved=[],
                conflicts=[],
                usage=usage,
                warnings=["retrieval found nothing above the minimum relevance score"],
            )

        prompt = get_prompt(request.prompt_version)
        context = self._format_context(retrieved) if prompt.requires_context else ""
        guided = answer_schema() if prompt.requires_citations else None

        response = await self._llm.complete(
            system=prompt.system,
            user=prompt.render_user(request.query, context),
            temperature=0.0,
            guided_json=guided,
            max_tokens=1024,
            operation="generate",
        )
        usage.prompt_tokens += response.prompt_tokens
        usage.completion_tokens += response.completion_tokens
        usage.retrieved_chunks = len(retrieved)

        try:
            payload = response.as_json()
        except LLMProtocolError as exc:
            warnings.append(f"generation returned unparseable output: {exc.message}")
            usage.latency_ms = round((time.perf_counter() - started) * 1000, 2)
            ANSWERS_ABSTAINED.labels(reason="unparseable_generation").inc()
            return AnswerResult(
                query=request.query,
                answer="",
                abstained=True,
                claims=[],
                retrieved=retrieved,
                conflicts=conflicts,
                usage=usage,
                warnings=warnings,
            )

        if payload.get("abstained"):
            usage.latency_ms = round((time.perf_counter() - started) * 1000, 2)
            ANSWERS_ABSTAINED.labels(reason="model_abstained").inc()
            reason = payload.get("abstain_reason") or "the model declined to answer"
            return AnswerResult(
                query=request.query,
                answer="",
                abstained=True,
                claims=[],
                retrieved=retrieved,
                conflicts=conflicts,
                usage=usage,
                warnings=[str(reason)],
            )

        claims = self._verify_claims(
            payload.get("claims") or [], has_context=prompt.requires_context
        )
        if not prompt.requires_context and claims:
            warnings.append(
                "answer produced without retrieval; not verified against any source "
                "(see each claim's status and AnswerResult.grounded_ratio)"
            )
        for claim in claims:
            CLAIMS_TOTAL.labels(status=claim.status.value).inc()
            if not claim.is_trustworthy:
                CLAIMS_UNSUPPORTED.inc()

        unsupported_ratio = (
            sum(1 for c in claims if not c.is_trustworthy) / len(claims) if claims else 1.0
        )
        usage.latency_ms = round((time.perf_counter() - started) * 1000, 2)

        if not claims or unsupported_ratio > self._settings.max_unsupported_ratio:
            ANSWERS_ABSTAINED.labels(reason="unsupported_ratio_exceeded").inc()
            warnings.append(
                f"{unsupported_ratio:.0%} of claims failed verification "
                f"(threshold {self._settings.max_unsupported_ratio:.0%})"
            )
            return AnswerResult(
                query=request.query,
                answer="",
                abstained=True,
                claims=claims,
                retrieved=retrieved,
                conflicts=conflicts,
                usage=usage,
                warnings=warnings,
            )

        trustworthy = [c for c in claims if c.is_trustworthy]
        answer_text = " ".join(c.text for c in trustworthy)
        result = AnswerResult(
            query=request.query,
            answer=answer_text,
            abstained=False,
            claims=trustworthy,
            retrieved=retrieved,
            conflicts=conflicts,
            usage=usage,
            warnings=warnings,
        )
        ANSWERS_ABSTAINED.labels(reason="none").inc(0)  # register the label with count 0
        _log.info(
            "query_complete",
            claims=len(trustworthy),
            faithfulness=round(result.faithfulness, 3),
            conflicts=len(conflicts),
            latency_ms=usage.latency_ms,
        )
        return result

    async def _rerank(
        self, query: str, retrieved: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Score the fused candidates with the cross-encoder and keep the top ones."""
        if not retrieved:
            return []
        scores = await self._reranker.rerank(query, [r.chunk.text for r in retrieved])
        scored = [
            r.model_copy(update={"rerank_score": round(score, 4)})
            for r, score in zip(retrieved, scores, strict=True)
        ]
        scored.sort(key=lambda r: r.rerank_score or 0.0, reverse=True)
        kept = [r for r in scored if (r.fused_score >= self._settings.min_retrieval_score)]
        return kept[: self._settings.rerank_top_k] or scored[: self._settings.rerank_top_k]

    def _verify_claims(self, raw_claims: list[Any], *, has_context: bool) -> list[Claim]:
        """Turn generator output into verified :class:`Claim` objects.

        When ``has_context`` is false (the no-retrieval ablation rung), there
        is no corpus to check a claim against -- demanding a citation would
        be a category error, not a verification failure. Every non-empty
        claim is marked ``UNGROUNDED`` and passed through unchecked; the
        caller is warned separately (see ``pipeline.query``) so this is never
        silently mistaken for a verified answer.
        """
        claims: list[Claim] = []
        for raw in raw_claims:
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text", "")).strip()
            if not text:
                continue

            if not has_context:
                claims.append(
                    Claim(text=text, cited_chunk_ids=[], status=ClaimStatus.UNGROUNDED)
                )
                continue

            cited_ids = [str(c) for c in (raw.get("cited_chunk_ids") or [])]

            if not cited_ids:
                claims.append(Claim(text=text, cited_chunk_ids=[], status=ClaimStatus.UNCITED))
                continue

            cited_chunks: list[Chunk] = [
                c for cid in cited_ids if (c := self._corpus.get(cid)) is not None
            ]
            if not cited_chunks:
                claims.append(
                    Claim(text=text, cited_chunk_ids=cited_ids, status=ClaimStatus.UNSUPPORTED)
                )
                continue

            report = score_claim_coverage(
                text, cited_chunks, threshold=self._settings.claim_coverage_threshold
            )
            claims.append(
                Claim(
                    text=text,
                    cited_chunk_ids=cited_ids,
                    status=(
                        ClaimStatus.SUPPORTED if report.supported else ClaimStatus.UNSUPPORTED
                    ),
                    coverage_score=report.score,
                    supporting_quote=report.supporting_quote,
                )
            )
        return claims

    @staticmethod
    def _format_context(retrieved: list[RetrievedChunk]) -> str:
        """Render retrieved chunks as labelled, citable blocks for the prompt."""
        blocks = []
        for item in retrieved:
            blocks.append(
                f"[{item.chunk.chunk_id}] ({item.chunk.citation_label})\n{item.chunk.text}"
            )
        return "\n\n".join(blocks)
