"""Offline evaluation harness — the no-retrieval / naive / cited ablation ladder.

Runs a labelled question set through the full pipeline once per prompt
version and reports the delta. The ``no-retrieval`` rung is the proof the
brief specifically asks for: since the corpus (the Hexfall rulebook) was
invented for this project, a model answering with retrieval switched off has
never seen this material, so its accuracy on that rung should sit near zero.
Any real lift on the ``naive`` and ``cited`` rungs is unambiguously
attributable to retrieval, not to the model recalling training data — a
distinction most "does RAG help" demos cannot actually make because they
evaluate on public documents the base model has plausibly already seen.

Dataset format is JSONL, one question per line::

    {"id": "q1", "query": "What does a Tower's Surge do?",
     "expected_answer": "captures every enemy Runner ... third cell",
     "expects_abstain": false}
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tabula_rag.config import Settings
from tabula_rag.corpus_index import CorpusIndex
from tabula_rag.eval.metrics import EvalCounters, score_case
from tabula_rag.llm.client import LLMClient
from tabula_rag.models import QueryRequest
from tabula_rag.observability import get_logger
from tabula_rag.pipeline import RAGService
from tabula_rag.retrieval.embeddings import EmbeddingClient
from tabula_rag.retrieval.reranker import RerankerClient

__all__ = ["EvalCase", "RunConfig", "RunReport", "load_dataset", "render_markdown", "run_suite"]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One labelled question."""

    case_id: str
    query: str
    expected_answer: str | None
    expects_abstain: bool

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> EvalCase:
        """Build a case from one decoded JSONL row."""
        return cls(
            case_id=str(data["id"]),
            query=str(data["query"]),
            expected_answer=data.get("expected_answer"),
            expects_abstain=bool(data.get("expects_abstain", False)),
        )


@dataclass(frozen=True, slots=True)
class RunConfig:
    """One point in the ablation grid: which prompt version to evaluate."""

    prompt_version: str

    @property
    def label(self) -> str:
        """Short identifier used as the ablation table row header."""
        return self.prompt_version


@dataclass
class RunReport:
    """Metrics for one configuration across the whole dataset."""

    config: RunConfig
    metrics: dict[str, float | int]
    outcomes: list[Any] = field(default_factory=list)
    """Per-case :class:`~tabula_rag.eval.metrics.CaseOutcome` results, kept
    alongside the aggregate metrics so a caller can break results down by
    category (e.g. answerable vs. expected-abstain) without re-running the
    suite."""
    wall_clock_s: float = 0.0
    cost_usd: float = 0.0
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render the report as JSON for the committed results file."""
        return {
            "config": {"prompt_version": self.config.prompt_version},
            "metrics": self.metrics,
            "outcomes": [
                {
                    "case_id": o.case_id,
                    "expects_abstain": o.expects_abstain,
                    "abstained": o.abstained,
                    "correct": o.correct,
                    "faithfulness": round(o.faithfulness, 4),
                }
                for o in self.outcomes
            ],
            "wall_clock_s": round(self.wall_clock_s, 3),
            "cost_usd": round(self.cost_usd, 6),
            "failures": self.failures,
        }


def load_dataset(path: Path) -> list[EvalCase]:
    """Read a JSONL question set."""
    cases: list[EvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(EvalCase.from_json(json.loads(line)))
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_number} is not a usable case: {exc}") from exc
    if not cases:
        raise ValueError(f"dataset {path} contains no cases")
    return cases


async def run_suite(
    cases: list[EvalCase],
    settings: Settings,
    configs: list[RunConfig],
    *,
    llm_factory: Any = None,
    embeddings: EmbeddingClient | None = None,
    reranker: RerankerClient | None = None,
    corpus_documents: list[tuple[str, str, str]] | None = None,
) -> list[RunReport]:
    """Evaluate every prompt-version configuration over every case.

    Args:
        cases: Labelled questions.
        settings: Base settings shared by every configuration.
        configs: The ablation grid — one entry per prompt version.
        llm_factory: Zero-argument callable returning a fresh
            :class:`LLMClient`, called once per configuration. Defaults to
            :class:`~tabula_rag.llm.client.OpenAICompatibleLLM`.
        embeddings: Shared embedding client; defaults to a fresh
            :class:`~tabula_rag.retrieval.embeddings.OpenAICompatibleEmbeddings`.
        reranker: Shared reranker client; defaults to
            :class:`~tabula_rag.retrieval.reranker.OpenAICompatibleReranker`.
        corpus_documents: ``(document_id, text, title)`` tuples to ingest into
            a fresh corpus before each configuration runs. Required in
            practice — an empty corpus can only ever abstain.

    Returns:
        One :class:`RunReport` per configuration, in input order.
    """
    from tabula_rag.llm.client import OpenAICompatibleLLM
    from tabula_rag.retrieval.embeddings import OpenAICompatibleEmbeddings
    from tabula_rag.retrieval.reranker import OpenAICompatibleReranker

    resolved_embeddings = embeddings or OpenAICompatibleEmbeddings(settings)
    resolved_reranker = reranker or OpenAICompatibleReranker(settings)
    resolved_llm_factory = llm_factory or (lambda: OpenAICompatibleLLM(settings))

    reports: list[RunReport] = []
    for config in configs:
        corpus = CorpusIndex(settings, resolved_embeddings)
        for doc_id, text, title in corpus_documents or []:
            await corpus.ingest(doc_id, text, title=title)

        llm: LLMClient = resolved_llm_factory()
        service = RAGService(settings, llm, resolved_embeddings, resolved_reranker, corpus)
        prompt = config.prompt_version
        use_retrieval = prompt != "no-retrieval"

        counters = EvalCounters()
        failures: list[str] = []
        tokens = 0
        started = time.perf_counter()

        for case in cases:
            request = QueryRequest(
                query=case.query, use_retrieval=use_retrieval, prompt_version=prompt
            )
            try:
                result = await service.query(request)
            except Exception as exc:
                failures.append(f"{case.case_id}: {type(exc).__name__}: {exc}")
                _log.warning("eval_case_failed", case_id=case.case_id, error=str(exc))
                continue
            tokens += result.usage.total_tokens
            counters.add(
                score_case(case.case_id, case.expected_answer, case.expects_abstain, result)
            )

        elapsed = time.perf_counter() - started
        cost = (settings.gpu_hourly_rate_usd / 3600.0) * (
            tokens / settings.measured_tokens_per_s
        )
        reports.append(
            RunReport(
                config=config,
                metrics=counters.to_dict(),
                outcomes=counters.outcomes,
                wall_clock_s=elapsed,
                cost_usd=cost,
                failures=failures,
            )
        )

    return reports


def render_markdown(reports: list[RunReport]) -> str:
    """Render the ablation table that goes into the write-up."""
    header = (
        "| prompt version | cases | accuracy | faithfulness | grounded | abstain recall | "
        "abstain precision | mean latency | tokens |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for r in reports:
        m = r.metrics
        rows.append(
            f"| {r.config.label} | {m['cases']} | {m['accuracy']:.1%} | "
            f"{m['mean_faithfulness']:.1%} | {m['mean_grounded_ratio']:.1%} | "
            f"{m['abstention_recall']:.1%} | "
            f"{m['abstention_precision']:.1%} | {m['mean_latency_ms']:.1f} ms | "
            f"{m['total_tokens']:,} |"
        )
    return header + "\n".join(rows) + "\n"
