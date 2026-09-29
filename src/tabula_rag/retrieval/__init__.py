"""Hybrid retrieval: BM25 + dense embeddings, fused by RRF, then reranked."""

from tabula_rag.retrieval.bm25 import BM25Index
from tabula_rag.retrieval.dense import DenseIndex
from tabula_rag.retrieval.embeddings import (
    EmbeddingClient,
    HashingEmbeddingClient,
    OpenAICompatibleEmbeddings,
)
from tabula_rag.retrieval.fusion import FusedResult, reciprocal_rank_fusion
from tabula_rag.retrieval.reranker import (
    LexicalOverlapReranker,
    OpenAICompatibleReranker,
    RerankerClient,
)

__all__ = [
    "BM25Index",
    "DenseIndex",
    "EmbeddingClient",
    "FusedResult",
    "HashingEmbeddingClient",
    "LexicalOverlapReranker",
    "OpenAICompatibleEmbeddings",
    "OpenAICompatibleReranker",
    "RerankerClient",
    "reciprocal_rank_fusion",
]
