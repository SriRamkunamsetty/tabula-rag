"""Dense embedding clients.

Production embeddings are served the same way as everything else in this
system: an OpenAI-compatible `/v1/embeddings` endpoint on the MI300X (a model
such as BAAI/bge-m3 served via vLLM). The pipeline depends only on the
:class:`EmbeddingClient` protocol, so tests and the offline demo run against
:class:`HashingEmbeddingClient` — a deterministic, dependency-free stand-in —
without any network access or downloaded weights.

``HashingEmbeddingClient`` is explicitly a lexical proxy, not a semantic
embedding: it hashes character n-grams into a fixed-width vector, so it
captures token and substring overlap well enough to exercise fusion and
reranking logic deterministically, but it does not capture paraphrase or
synonymy the way a trained embedding model does. That gap is real and is
called out in ADR 2 rather than hidden.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

import httpx

from tabula_rag.config import Settings
from tabula_rag.errors import LLMProtocolError, LLMTimeoutError

__all__ = ["EmbeddingClient", "HashingEmbeddingClient", "OpenAICompatibleEmbeddings", "Vector"]

Vector = list[float]


class EmbeddingClient(Protocol):
    """Interface the retrieval layer depends on."""

    async def embed(self, texts: list[str]) -> list[Vector]:
        """Return one embedding vector per input text, in order."""
        ...

    async def aclose(self) -> None:
        """Release transport resources."""
        ...


class OpenAICompatibleEmbeddings:
    """Production client for a vLLM-served embedding model."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        """Build a client; an injected transport is used by tests and benchmarks."""
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            base_url=settings.embedding_base_url,
            timeout=httpx.Timeout(
                settings.llm_timeout_s, connect=settings.llm_connect_timeout_s
            ),
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
        )

    async def embed(self, texts: list[str]) -> list[Vector]:
        """Call the embeddings endpoint for a batch of texts."""
        try:
            response = await self._client.post(
                "/embeddings",
                json={"model": self._settings.embedding_model, "input": texts},
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError("embedding request timed out") from exc
        if response.status_code >= 400:
            raise LLMProtocolError(
                f"embedding endpoint rejected the request ({response.status_code})",
                detail=response.text[:500],
            )
        body = response.json()
        try:
            items = sorted(body["data"], key=lambda d: d["index"])
            return [item["embedding"] for item in items]
        except (KeyError, TypeError) as exc:
            raise LLMProtocolError(
                "embedding response was missing expected fields", detail=str(body)[:500]
            ) from exc

    async def aclose(self) -> None:
        """Close the underlying HTTP transport."""
        await self._client.aclose()


class HashingEmbeddingClient:
    """Deterministic, GPU-free lexical proxy for a real embedding model.

    Each text is lowercased, split into overlapping character trigrams, and
    each trigram is hashed into one of ``dim`` buckets with its sign
    determined by a second hash — the standard "hashing trick" for a
    dependency-free bag-of-n-grams vector. The result is L2-normalised so
    cosine similarity behaves sensibly.
    """

    def __init__(self, dim: int = 256) -> None:
        """Create a hashing embedder with a ``dim``-wide output vector."""
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[Vector]:
        """Compute a hashed n-gram vector for each text."""
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> Vector:
        vector = [0.0] * self._dim
        normalized = text.lower()
        grams = [normalized[i : i + 3] for i in range(max(1, len(normalized) - 2))]
        if not grams:
            grams = [normalized or " "]
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    async def aclose(self) -> None:
        """No-op; present so the fake satisfies the client protocol."""
        return None


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity between two vectors of equal length."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
