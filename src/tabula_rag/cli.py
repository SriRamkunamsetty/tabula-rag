"""Command-line interface."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import typer

from tabula_rag import __version__
from tabula_rag.config import Settings, get_settings
from tabula_rag.corpus_index import CorpusIndex
from tabula_rag.eval.harness import RunConfig, load_dataset, render_markdown, run_suite
from tabula_rag.llm.fake import ScriptedLLM
from tabula_rag.models import QueryRequest
from tabula_rag.observability import configure_logging
from tabula_rag.pipeline import RAGService
from tabula_rag.prompts.registry import DEFAULT_VERSION, list_versions
from tabula_rag.retrieval.embeddings import HashingEmbeddingClient, OpenAICompatibleEmbeddings
from tabula_rag.retrieval.reranker import LexicalOverlapReranker

app = typer.Typer(
    add_completion=False,
    help="TABULA RAG - grounded question answering with claim-level verification.",
    no_args_is_help=True,
)


def _settings(overrides: dict[str, Any] | None = None) -> Settings:
    settings = get_settings()
    return settings.model_copy(update=overrides) if overrides else settings


@app.command()
def version() -> None:
    """Print the service version."""
    typer.echo(f"tabula-rag {__version__}")


@app.command("prompts")
def list_prompts() -> None:
    """List available generation prompt versions."""
    typer.echo(f"Prompt versions (default: {DEFAULT_VERSION}):")
    for version_id, summary in list_versions():
        typer.echo(f"  {version_id:<14} {summary}")


@app.command()
def ingest(
    path: Path = typer.Option(
        ..., exists=True, readable=True, help="Markdown/text file to ingest."
    ),
    document_id: str = typer.Option(None, help="Defaults to the file stem."),
    title: str = typer.Option("", help="Human-readable document title."),
) -> None:
    """Ingest one document into a fresh in-process corpus and report chunk count.

    This command is for quick inspection; a running service persists its
    corpus for the process lifetime via POST /v1/ingest instead.
    """
    settings = _settings()
    configure_logging(settings.log_level, "console")
    doc_id = document_id or path.stem
    embeddings = OpenAICompatibleEmbeddings(settings)
    corpus = CorpusIndex(settings, embeddings)
    chunks = asyncio.run(corpus.ingest(doc_id, path.read_text(encoding="utf-8"), title=title))
    typer.echo(f"indexed {len(chunks)} chunks from {path} as document '{doc_id}'")
    for c in chunks[:5]:
        typer.echo(f"  {c.chunk_id}  §{c.section or '(no heading)'}  {len(c.text)} chars")


@app.command()
def evaluate(
    dataset: Path = typer.Option(..., exists=True, help="JSONL evaluation set."),
    report: Path = typer.Option(Path("reports/eval.json"), help="Where to write JSON."),
    prompts: str = typer.Option(
        "no-retrieval,naive,cited", help="Comma-separated prompt versions to sweep."
    ),
) -> None:
    """Run the ablation ladder over a labelled dataset."""
    settings = _settings()
    configure_logging(settings.log_level, "console")
    cases = load_dataset(dataset)
    grid = [RunConfig(prompt_version=p) for p in prompts.split(",")]
    reports = asyncio.run(run_suite(cases, settings, grid))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps([r.to_dict() for r in reports], indent=2), encoding="utf-8")
    typer.echo(render_markdown(reports))
    typer.echo(f"wrote {report}")


@app.command()
def demo() -> None:
    """Run the pipeline end to end with no GPU, using scripted models.

    Ingests the Hexfall rulebook and its errata sheet, then asks three
    questions: one the corpus answers cleanly, one it cannot answer at all
    (must abstain), and one where the base rulebook and the errata disagree
    (must be flagged as a conflict).
    """
    settings = _settings(
        {"environment": "dev", "max_unsupported_ratio": 0.20, "claim_coverage_threshold": 0.4}
    )
    configure_logging("INFO", "console")

    corpus_dir = Path(__file__).resolve().parents[2] / "corpus"
    rulebook = (corpus_dir / "hexfall_rulebook_v1.md").read_text(encoding="utf-8")
    errata = (corpus_dir / "hexfall_errata.md").read_text(encoding="utf-8")

    embeddings = HashingEmbeddingClient()
    corpus = CorpusIndex(settings, embeddings)
    asyncio.run(corpus.ingest("hexfall-rulebook", rulebook, title="Hexfall Rulebook"))
    asyncio.run(corpus.ingest("hexfall-errata", errata, title="Hexfall Errata"))
    typer.echo(f"indexed {len(corpus)} chunks across {len(corpus.document_ids)} documents\n")

    scripted = ScriptedLLM(
        responses=[
            # Q1: answerable, well-grounded
            json.dumps(
                {
                    "claims": [
                        {
                            "text": (
                                "A Tower's Surge captures every enemy Runner in a "
                                "straight line of sight up to and including the third cell."
                            ),
                            "cited_chunk_ids": [_first_chunk_with(corpus, "Surge")],
                        }
                    ],
                    "abstained": False,
                    "abstain_reason": None,
                }
            ),
            # Q2: unanswerable — must abstain
            json.dumps(
                {
                    "claims": [],
                    "abstained": True,
                    "abstain_reason": ("The rulebook does not specify a board colour scheme."),
                }
            ),
            # Q3: turn-limit question — base rulebook chunk cited, but the
            # errata (also retrieved) revises this number, so conflicts.py
            # should flag it regardless of which chunk generation drew from.
            json.dumps(
                {
                    "claims": [
                        {
                            "text": (
                                "A game that reaches move 60 without a winner "
                                "is declared a draw."
                            ),
                            "cited_chunk_ids": [_first_chunk_with(corpus, "move 60")],
                        }
                    ],
                    "abstained": False,
                    "abstain_reason": None,
                }
            ),
        ],
    )
    reranker = LexicalOverlapReranker()
    service = RAGService(settings, scripted, embeddings, reranker, corpus)

    questions = [
        "What does a Tower's Surge ability do?",
        "What colour is the Hexfall board?",
        "After how many moves is Hexfall declared a draw?",
    ]
    for q in questions:
        result = asyncio.run(service.query(QueryRequest(query=q)))
        typer.echo(f"Q: {q}")
        if result.abstained:
            typer.echo(f"   [ABSTAIN] {result.warnings[0] if result.warnings else ''}")
        else:
            typer.echo(f"   [ANSWER]  {result.answer}")
            typer.echo(
                f"             faithfulness={result.faithfulness:.2f}  "
                f"claims={len(result.claims)}"
            )
        if result.conflicts:
            for c in result.conflicts:
                typer.echo(
                    f"   [CONFLICT] {c.source_a} says {c.value_a}, "
                    f"{c.source_b} says {c.value_b}  (re: {c.claim_text})"
                )
        typer.echo("")


def _first_chunk_with(corpus: CorpusIndex, needle: str) -> str:
    """Find a chunk id containing ``needle``.

    Used only to script the demo deterministically.
    """
    for chunk_id in list(corpus._chunks):
        chunk = corpus.get(chunk_id)
        if chunk and needle.lower() in chunk.text.lower():
            return chunk_id
    return ""


if __name__ == "__main__":  # pragma: no cover
    app()
