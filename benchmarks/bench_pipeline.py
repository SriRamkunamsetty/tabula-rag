r"""Throughput and latency benchmark for the RAG pipeline.

Run against live vLLM endpoints on MI300X to populate the deployment's real
numbers::

    TABULA_RAG_LLM_BASE_URL=http://<ip>:8000/v1 \
    TABULA_RAG_EMBEDDING_BASE_URL=http://<ip>:8001/v1 \
    TABULA_RAG_RERANKER_BASE_URL=http://<ip>:8002/v1 \
        python benchmarks/bench_pipeline.py --live --requests 40 --concurrency 8

With no ``--live`` flag, benchmarks the pipeline's own overhead -- corpus
indexing, hybrid retrieval, reranking, claim verification -- against
deterministic offline clients (:class:`HashingEmbeddingClient`,
:class:`LexicalOverlapReranker`, :class:`ScriptedLLM`), which needs no GPU
and no network. Every run is labelled with its own mode so a number from one
is never mistaken for the other, following the same convention as
mini-challenge 2's benchmark script.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from tabula_rag.config import Settings, get_settings
from tabula_rag.corpus_index import CorpusIndex
from tabula_rag.llm.client import OpenAICompatibleLLM
from tabula_rag.llm.fake import ScriptedLLM
from tabula_rag.models import QueryRequest
from tabula_rag.pipeline import RAGService
from tabula_rag.retrieval.embeddings import HashingEmbeddingClient, OpenAICompatibleEmbeddings
from tabula_rag.retrieval.reranker import LexicalOverlapReranker, OpenAICompatibleReranker

REPO_ROOT = Path(__file__).resolve().parents[1]
RULEBOOK = (REPO_ROOT / "corpus" / "hexfall_rulebook_v1.md").read_text(encoding="utf-8")

SCRIPTED_ANSWER = json.dumps({
    "claims": [{"text": "A Tower's Surge captures every enemy Runner in a straight line "
                         "of sight up to and including the third cell.",
                "cited_chunk_ids": ["hexfall-rulebook::0005"]}],
    "abstained": False, "abstain_reason": None,
})


@dataclass(frozen=True, slots=True)
class BenchResult:
    """One benchmark run's summary statistics."""

    mode: str
    llm_model: str
    requests: int
    concurrency: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    throughput_req_s: float
    total_tokens: int
    tokens_per_s: float
    error_rate: float
    cost_per_1k_queries_usd: float


async def _run_one(service: RAGService, query_text: str) -> tuple[float, int, bool]:
    started = time.perf_counter()
    try:
        result = await service.query(QueryRequest(query=query_text))
        return (time.perf_counter() - started) * 1000, result.usage.total_tokens, True
    except Exception:
        return (time.perf_counter() - started) * 1000, 0, False


async def run_benchmark(
    settings: Settings, *, total_requests: int, concurrency: int, live: bool
) -> BenchResult:
    """Fire ``total_requests`` queries at ``concurrency`` and summarise."""
    if live:
        llm = OpenAICompatibleLLM(settings)
        embeddings = OpenAICompatibleEmbeddings(settings)
        reranker = OpenAICompatibleReranker(settings)
    else:
        llm = ScriptedLLM([SCRIPTED_ANSWER] * total_requests)
        embeddings = HashingEmbeddingClient()
        reranker = LexicalOverlapReranker()

    corpus = CorpusIndex(settings, embeddings)
    await corpus.ingest("hexfall-rulebook", RULEBOOK, title="Hexfall Rulebook")
    service = RAGService(settings, llm, embeddings, reranker, corpus)

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(_i: int) -> tuple[float, int, bool]:
        async with semaphore:
            return await _run_one(service, "What does a Tower's Surge ability do?")

    wall_start = time.perf_counter()
    outcomes = await asyncio.gather(*[bounded(i) for i in range(total_requests)])
    wall_elapsed = time.perf_counter() - wall_start
    await llm.aclose()

    latencies = sorted(ms for ms, _, ok in outcomes if ok)
    tokens = sum(t for _, t, ok in outcomes if ok)
    errors = sum(1 for *_r, ok in outcomes if not ok)
    if not latencies:
        raise RuntimeError("every benchmark request failed; nothing to report")

    tokens_per_s = tokens / wall_elapsed if wall_elapsed > 0 else 0.0
    cost_per_query = (
        (settings.gpu_hourly_rate_usd / 3600.0) * (tokens / max(len(latencies), 1)) / max(tokens_per_s, 1e-9)
    ) if tokens_per_s > 0 else 0.0

    return BenchResult(
        mode="live-endpoints" if live else "harness-only (no GPU)",
        llm_model=settings.llm_model, requests=total_requests, concurrency=concurrency,
        p50_ms=round(_percentile(latencies, 0.50), 2),
        p95_ms=round(_percentile(latencies, 0.95), 2),
        p99_ms=round(_percentile(latencies, 0.99), 2),
        mean_ms=round(statistics.fmean(latencies), 2),
        throughput_req_s=round(len(latencies) / wall_elapsed, 3) if wall_elapsed > 0 else 0.0,
        total_tokens=tokens, tokens_per_s=round(tokens_per_s, 2),
        error_rate=round(errors / total_requests, 4),
        cost_per_1k_queries_usd=round(cost_per_query * 1000, 4),
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = min(len(sorted_values) - 1, round(fraction * (len(sorted_values) - 1)))
    return sorted_values[index]


def main() -> None:
    """Parse CLI arguments, run the benchmark, print and append the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results.json"))
    args = parser.parse_args()

    settings = get_settings()
    result = asyncio.run(
        run_benchmark(settings, total_requests=args.requests, concurrency=args.concurrency, live=args.live)
    )
    print(json.dumps(asdict(result), indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else []
    existing.append(asdict(result))
    args.output.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nappended to {args.output}")


if __name__ == "__main__":
    main()
