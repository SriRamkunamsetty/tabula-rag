"""Cross-encoder reranking.

Fusion (RRF) is a good coarse filter but knows nothing about the query and
candidate *text* — only their ranks in two lists. A cross-encoder reranker
reads the query and each candidate together and scores relevance directly,
which is what narrows the fused top-k down to the handful of chunks that
actually go into the generation prompt.

The production client speaks the Cohere-style rerank contract
(``{"query", "documents"} -> {"results": [{"index", "relevance_score"}]}``),
which vLLM implements for cross-encoder models such as BAAI/bge-reranker-v2-m3.
Tests and the offline demo use :class:`LexicalOverlapReranker`, a deterministic
stand-in scored by token overlap between the query and each candidate.
"""

from __future__ import annotations

import re
from typing import Protocol

import httpx

from tabula_rag.config import Settings
from tabula_rag.errors import LLMProtocolError, LLMTimeoutError

__all__ = ["LexicalOverlapReranker", "OpenAICompatibleReranker", "RerankerClient"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class RerankerClient(Protocol):
    """Interface the retrieval layer depends on."""

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score per document, in input order."""
        ...

    async def aclose(self) -> None:
        """Release transport resources."""
        ...


class OpenAICompatibleReranker:
    """Production client for a vLLM-served cross-encoder reranker."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        """Build a client; an injected transport is used by tests and benchmarks."""
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            base_url=settings.reranker_base_url,
            timeout=httpx.Timeout(
                settings.llm_timeout_s, connect=settings.llm_connect_timeout_s
            ),
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
        )

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Call the rerank endpoint and return scores in input order."""
        try:
            response = await self._client.post(
                "/rerank",
                json={
                    "model": self._settings.reranker_model,
                    "query": query,
                    "documents": documents,
                },
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("rerank request timed out") from exc
        if response.status_code >= 400:
            raise LLMProtocolError(
                f"rerank endpoint rejected the request ({response.status_code})",
                detail=response.text[:500],
            )
        body = response.json()
        try:
            scores = [0.0] * len(documents)
            for item in body["results"]:
                scores[item["index"]] = float(item["relevance_score"])
            return scores
        except (KeyError, TypeError, IndexError) as exc:
            raise LLMProtocolError(
                "rerank response was missing expected fields", detail=str(body)[:500]
            ) from exc

    async def aclose(self) -> None:
        """Close the underlying HTTP transport."""
        await self._client.aclose()


class LexicalOverlapReranker:
    """Deterministic, GPU-free reranker: Jaccard token overlap with the query.

    Not a substitute for a trained cross-encoder's semantic judgment — it is
    a stand-in that lets fusion → rerank → generation be exercised end to end
    in tests without a served model, in the same spirit as
    :class:`tabula_rag.retrieval.embeddings.HashingEmbeddingClient`.
    """

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Score each document by token-set overlap with the query."""
        query_terms = set(_TOKEN_RE.findall(query.lower()))
        if not query_terms:
            return [0.0] * len(documents)
        scores = []
        for doc in documents:
            doc_terms = set(_TOKEN_RE.findall(doc.lower()))
            if not doc_terms:
                scores.append(0.0)
                continue
            overlap = len(query_terms & doc_terms) / len(query_terms | doc_terms)
            scores.append(overlap)
        return scores

    async def aclose(self) -> None:
        """No-op; present so the fake satisfies the client protocol."""
        return None
