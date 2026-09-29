"""The harness entrypoint: ``python3 /app/app.py ...``.

Exactly the two invocations the challenge defines::

    python3 /app/app.py --index /app/corpus
    python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "What is ...?"

Unrecognised flags are ignored with a warning: a crash scores zero, a surplus flag should not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tabula_rag.contract.config import ContractSettings
from tabula_rag.contract.runner import output_path, run_index, run_query

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="TABULA RAG, Mini-Challenge 3 contract mode",
        allow_abbrev=False,
    )
    parser.add_argument("--index", metavar="CORPUS_DIR", help="index this corpus (run once)")
    parser.add_argument(
        "--corpus", metavar="CORPUS_DIR", help="corpus directory for a question"
    )
    parser.add_argument("--query-id", help="id given by the harness; names the output file")
    parser.add_argument("--query", help="the question to answer")
    parser.add_argument(
        "--output-dir", help="override the output directory (default /app/output)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one invocation. Always exits 0: the output file, not the exit code, is the result."""
    args, unknown = _parser().parse_known_args(argv)
    if unknown:
        print(f"app.py: ignoring unrecognized arguments: {unknown}", file=sys.stderr)
    settings = ContractSettings.from_env()
    if args.output_dir:
        settings = ContractSettings(
            **{**settings.__dict__, "output_dir": Path(args.output_dir)}
        )

    if args.index:
        summary = run_index(Path(args.index), settings)
        print(json.dumps({"indexed": summary}))
        return 0
    if args.query is not None and args.query_id:
        corpus = Path(args.corpus) if args.corpus else None
        result = run_query(corpus, args.query_id, args.query, settings)
        print(
            json.dumps(
                {"output": str(output_path(settings, args.query_id)), **result.to_output()}
            )
        )
        return 0
    print(
        "usage: app.py --index CORPUS | --corpus CORPUS --query-id ID --query TEXT",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
