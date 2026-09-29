"""A flat, in-memory dense vector index.

A linear scan over cosine similarity is the right choice at this corpus
scale — a rulebook and its supporting documents are hundreds of chunks, not
millions — and it keeps the index dependency-free and fully inspectable. See
``docs/adr/002-flat-dense-index.md`` for the scaling argument and the upgrade
path (an ANN index such as HNSW or a managed vector store) once corpus size
would make a linear scan the bottleneck.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tabula_rag.retrieval.embeddings import Vector, cosine_similarity

__all__ = ["DenseIndex"]


@dataclass
class DenseIndex:
    """An in-memory, brute-force cosine-similarity index."""

    _ids: list[str] = field(default_factory=list, init=False)
    _vectors: list[Vector] = field(default_factory=list, init=False)

    def __len__(self) -> int:
        """Number of vectors currently indexed."""
        return len(self._ids)

    def add(self, doc_id: str, vector: Vector) -> None:
        """Add one vector to the index."""
        self._ids.append(doc_id)
        self._vectors.append(vector)

    def search(self, query_vector: Vector, top_k: int = 10) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(doc_id, similarity)`` pairs, highest first.

        A cosine similarity of exactly 0.0 (orthogonal or a zero vector) is
        excluded from results, matching :class:`BM25Index`'s convention of
        never returning a result with no genuine signal behind it.
        """
        if not self._ids or not query_vector:
            return []
        scored = [
            (doc_id, cosine_similarity(query_vector, vector))
            for doc_id, vector in zip(self._ids, self._vectors, strict=True)
        ]
        ranked = sorted((s for s in scored if s[1] > 0.0), key=lambda s: s[1], reverse=True)
        return ranked[:top_k]
