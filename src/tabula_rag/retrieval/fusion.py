"""Reciprocal Rank Fusion (RRF).

Combines two independently ranked lists — BM25's lexical ranking and the
dense index's semantic ranking — into one fused ranking, using each item's
*rank* rather than its raw score. This sidesteps the score-scale mismatch
between BM25 (unbounded) and cosine similarity (-1 to 1): RRF only ever asks
"how high did this item rank," which is comparable across arbitrarily
different scoring functions.

Reference: Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion Outperforms
Condorcet and Individual Rank Learning Methods," SIGIR 2009.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["FusedResult", "reciprocal_rank_fusion"]


@dataclass(frozen=True, slots=True)
class FusedResult:
    """One item's fused rank and the individual ranks that produced it."""

    doc_id: str
    fused_score: float
    lexical_rank: int | None
    dense_rank: int | None


def reciprocal_rank_fusion(
    lexical: list[tuple[str, float]],
    dense: list[tuple[str, float]],
    *,
    k: int = 60,
) -> list[FusedResult]:
    """Fuse two ranked result lists by reciprocal rank.

    Args:
        lexical: ``(doc_id, score)`` pairs from BM25, highest score first.
        dense: ``(doc_id, score)`` pairs from the dense index, highest first.
        k: RRF's damping constant. Higher values flatten the contribution of
            rank differences further down each list; 60 is the value from the
            original paper and a reasonable default absent corpus-specific
            tuning data.

    Returns:
        Fused results sorted by descending fused score. An item retrieved by
        both lists scores higher than one retrieved by only one, which is
        exactly the "agreement between independent signals" property that
        makes hybrid retrieval more robust than either signal alone.
    """
    scores: dict[str, float] = {}
    lexical_ranks: dict[str, int] = {}
    dense_ranks: dict[str, int] = {}

    for rank, (doc_id, _score) in enumerate(lexical, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        lexical_ranks[doc_id] = rank

    for rank, (doc_id, _score) in enumerate(dense, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        dense_ranks[doc_id] = rank

    fused = [
        FusedResult(
            doc_id=doc_id,
            fused_score=score,
            lexical_rank=lexical_ranks.get(doc_id),
            dense_rank=dense_ranks.get(doc_id),
        )
        for doc_id, score in scores.items()
    ]
    fused.sort(key=lambda r: r.fused_score, reverse=True)
    return fused
