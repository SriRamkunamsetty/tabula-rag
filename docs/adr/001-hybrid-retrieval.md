# ADR 1 — Hybrid retrieval (BM25 + dense), fused by reciprocal rank, then reranked

## Status
Accepted

## Context
Mini-challenge 3 asks for RAG that improves accuracy and reduces
hallucination on proprietary domain knowledge. The retrieval stage decides
what the generator is even allowed to see, so its quality caps everything
downstream — no amount of citation-checking recovers from the right chunk
never being retrieved in the first place.

Lexical (BM25) and dense (embedding) retrieval fail in complementary ways.
BM25 misses a paraphrase that shares no vocabulary with the query ("what
stops a piece from being taken back" vs. the rulebook's "Leaping over it").
Dense retrieval misses an exact distinctive term it has no reason to weight
specially (a rule ID, a capitalised game term like "Surge" or "the Well").
A system relying on only one inherits that one's blind spot.

## Decision
Run both independently, then fuse with Reciprocal Rank Fusion (RRF) rather
than a weighted blend of raw scores. BM25's scores are unbounded and
corpus-size-dependent; cosine similarity is bounded in [-1, 1]. Averaging
those two directly means whichever one happens to produce larger numbers on
a given corpus silently dominates. RRF sidesteps this by using each item's
*rank* in each list, not its score, which is exactly the kind of
scale-mismatch problem `tabula_ocr`'s consensus module (mini-challenge 2)
solved a different way (normalised comparison in `normalize.py`) — same
underlying issue, the appropriate fix differs by domain.

A cross-encoder reranker then narrows the fused top-k to the generation
context, reading query and candidate text together rather than relying on
either retriever's rank alone.

## Consequences
- **Positive.** A chunk retrieved by both signals ranks above one retrieved
  by only one — agreement between independent retrieval methods is treated
  as evidence, the same principle mini-challenge 2's multi-pass consensus
  applies to OCR decoding.
- **Positive.** No score-scale tuning is needed when the embedding model or
  its dimensionality changes; RRF's `k` constant is the only knob, and 60
  (the value from the original paper) is a defensible default absent
  corpus-specific tuning data.
- **Negative.** RRF discards score *magnitude* entirely — a fused result
  says "ranked well by these signals" but not "how confident either signal
  was." The reranker's score is what carries that information back in for
  the final top-k selection.

## Alternatives considered
- **Dense-only retrieval.** Rejected: misses exact-term queries on rule
  names and numbers, which this corpus (a rulebook, dense with specific
  terminology) has many of.
- **Weighted score blending.** Rejected for the scale-mismatch reason above;
  would need per-corpus recalibration to stay meaningful.
- **A single reranker pass with no prior retrieval stage** (cross-encode
  every chunk against the query). Rejected as the corpus grows: a
  cross-encoder scores query-document pairs one at a time and does not scale
  to scoring an entire corpus per query, which is exactly why it is used to
  narrow a retrieved candidate set rather than to search one directly.
