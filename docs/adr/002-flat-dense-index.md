# ADR 2 — A flat, brute-force dense index, with a named upgrade path

## Status
Accepted

## Context
Dense retrieval needs a way to find the vectors nearest a query vector. At
scale, that means an approximate nearest-neighbour (ANN) index — HNSW,
IVF-PQ, or a managed vector database. At the scale this service actually
operates at — a rulebook and its supporting documents, on the order of tens
to a few hundred chunks — none of that machinery pays for itself.

## Decision
`DenseIndex` (`retrieval/dense.py`) is a linear scan: compute cosine
similarity against every stored vector, sort, return the top-k. No ANN
structure, no external vector database, no additional deployment
dependency.

## Consequences
- **Positive.** Zero operational surface: no vector database to provision,
  version, or keep in sync with the chunk store. `CorpusIndex` owns the
  chunks, the BM25 index, and the dense index as one in-memory object that
  cannot drift out of sync with itself.
- **Positive.** Exact, not approximate — a linear scan cannot return a
  false-negative nearest neighbour the way an ANN index occasionally can,
  which matters for a service whose entire premise is not silently missing
  the one chunk that would have grounded the answer correctly.
- **Positive.** Fully inspectable: `cosine_similarity` and the scan loop are
  a dozen lines each, unit-tested directly, with no black-box index
  structure to trust.
- **Negative.** O(n) per query in the number of indexed chunks. At corpus
  sizes in the tens of thousands of chunks and beyond, this becomes the
  latency bottleneck.

## Upgrade path
If corpus size grows past the point where a linear scan is the bottleneck
(measurable directly: `benchmarks/bench_pipeline.py` reports retrieval
latency separately from generation latency), the fix is local: `DenseIndex`
is accessed only through `CorpusIndex.hybrid_search`, so swapping the linear
scan for an HNSW index (e.g. via `hnswlib` or a managed store) changes one
class's internals and nothing in `pipeline.py`, `api.py`, or the eval
harness. This is the same "protocol now, swap the implementation later"
pattern used for the LLM, embedding, and reranker clients throughout this
service — the interface is designed once, and the implementation behind it
scales when there's a measured reason to.
