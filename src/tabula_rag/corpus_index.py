"""The corpus: chunk storage plus its two retrieval indexes.

Kept as one class because the three pieces — the chunk store, the BM25
index, and the dense index — must always stay in sync: every chunk added to
one is added to all three in the same call, so ``hybrid_search`` never risks
returning a chunk id one index knows about and another doesn't.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tabula_rag.chunking import chunk_document
from tabula_rag.config import Settings
from tabula_rag.models import Chunk, RetrievedChunk
from tabula_rag.retrieval.bm25 import BM25Index
from tabula_rag.retrieval.dense import DenseIndex
from tabula_rag.retrieval.embeddings import EmbeddingClient
from tabula_rag.retrieval.fusion import reciprocal_rank_fusion

__all__ = ["CorpusIndex"]


@dataclass
class CorpusIndex:
    """Holds every ingested chunk and both retrieval indexes over them."""

    settings: Settings
    embeddings: EmbeddingClient
    _chunks: dict[str, Chunk] = field(default_factory=dict, init=False)
    _bm25: BM25Index = field(init=False)
    _dense: DenseIndex = field(default_factory=DenseIndex, init=False)

    def __post_init__(self) -> None:
        """Build the BM25 index using the configured k1/b parameters."""
        self._bm25 = BM25Index(k1=self.settings.bm25_k1, b=self.settings.bm25_b)

    def __len__(self) -> int:
        """Total chunks currently indexed."""
        return len(self._chunks)

    @property
    def document_ids(self) -> set[str]:
        """Distinct source document ids currently indexed."""
        return {c.document_id for c in self._chunks.values()}

    async def ingest(self, document_id: str, text: str, title: str = "") -> list[Chunk]:
        """Chunk a document and add every chunk to both indexes."""
        chunks = chunk_document(
            document_id,
            text,
            title=title,
            target_tokens=self.settings.chunk_target_tokens,
            overlap_tokens=self.settings.chunk_overlap_tokens,
        )
        if not chunks:
            return []
        vectors = await self.embeddings.embed([c.text for c in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._chunks[chunk.chunk_id] = chunk
            self._bm25.add(chunk.chunk_id, chunk.text)
            self._dense.add(chunk.chunk_id, vector)
        return chunks

    def get(self, chunk_id: str) -> Chunk | None:
        """Look up a chunk by id, or ``None`` if not indexed."""
        return self._chunks.get(chunk_id)

    async def hybrid_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Lexical + dense retrieval, fused by reciprocal rank.

        Returns an empty list for an empty corpus or a query with no signal
        in either index — retrieval finding nothing is a legitimate outcome
        the pipeline must be able to act on (abstain), not an error.
        """
        if not self._chunks:
            return []
        lexical = self._bm25.search(query, top_k=top_k * 2)
        query_vector = (await self.embeddings.embed([query]))[0]
        dense = self._dense.search(query_vector, top_k=top_k * 2)
        fused = reciprocal_rank_fusion(lexical, dense, k=self.settings.rrf_k)

        results: list[RetrievedChunk] = []
        for item in fused[:top_k]:
            chunk = self._chunks.get(item.doc_id)
            if chunk is None:  # pragma: no cover - defensive; indexes are kept in sync
                continue
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    lexical_rank=item.lexical_rank,
                    dense_rank=item.dense_rank,
                    fused_score=round(item.fused_score, 6),
                )
            )
        return results
