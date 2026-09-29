r"""Replay the grader's invocation model against this repository, then score like the grader.

    python scripts/contract_selfcheck.py                    # synthetic corpus + scripted model server
    python scripts/contract_selfcheck.py --live             # your real model (TABULA_RAG_LLM_BASE_URL)
    python scripts/contract_selfcheck.py --live \\
        --corpus mc3-starter-kit/corpus --questions mc3-starter-kit/questions.json

What it does that in-process tests cannot: every step is a separate OS process, exactly as the
harness runs them (``app.py --index`` once, then one ``app.py --query-id ...`` per question), the
model is reached over real HTTP with real base64 images, and each question's wall-clock time is
checked against the 30 second limit. Scoring follows the published rule: 20 points per question,
answer must match after normalisation AND the citation set must match exactly.

Question files may be JSON or JSONL; recognised keys are ``question``/``query``,
``answer``/``expected_answer`` and ``citations``/``expected_citations``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from tabula_rag.contract.llm import prepare_image  # noqa: E402
from tabula_rag.contract.text import grader_normalize  # noqa: E402

PER_QUESTION_LIMIT_S = 30.0
POINTS = 20


def _load_questions(path: Path) -> list[tuple[str, str, list[str]]]:
    raw = path.read_text(encoding="utf-8").strip()
    rows: list[Any] = (
        json.loads(raw)
        if raw.startswith("[")
        else [json.loads(x) for x in raw.splitlines() if x]
    )
    if isinstance(rows, dict):
        rows = rows.get("questions", [])
    out = []
    for row in rows:
        question = row.get("question") or row.get("query") or ""
        answer = row.get("answer", row.get("expected_answer", "")) or ""
        cites = row.get("citations", row.get("expected_citations", [])) or []
        out.append((question, str(answer), [str(c) for c in cites]))
    return out


class _ScriptedServer:
    """A tiny OpenAI-compatible server backed by the test suite's scripted model."""

    def __init__(self, corpus: Path) -> None:
        from contract_corpus import IMAGE_TRANSCRIPTS, good_replies

        replies = good_replies()
        transcripts = {}
        for relative, text in IMAGE_TRANSCRIPTS.items():
            png, _ = prepare_image((corpus / relative).read_bytes(), 1600)
            transcripts[base64.b64encode(png).decode("ascii")] = text

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:  # silence
                return

            def _send(self, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                self._send({"data": [{"id": "scripted-vlm"}]})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                content = body["messages"][-1]["content"]
                if isinstance(content, list):  # an image transcription request
                    uri = content[1]["image_url"]["url"].split(",", 1)[1]
                    text = transcripts.get(uri, "")
                else:
                    question = content.split("\n", 1)[0]
                    reply = next(
                        (r for k, r in replies.items() if k in question),
                        {"reasoning": "", "answerable": False, "answer": "", "sources": []},
                    )
                    text = json.dumps(reply)
                self._send({"choices": [{"message": {"content": text}}]})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--live", action="store_true", help="use the model at TABULA_RAG_LLM_BASE_URL"
    )
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--questions", type=Path)
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="tabula-selfcheck-"))
    server: _ScriptedServer | None = None
    if args.corpus:
        corpus = args.corpus
    else:
        from contract_corpus import build_corpus

        corpus = build_corpus(work / "corpus")
    if args.questions:
        questions = _load_questions(args.questions)
    else:
        from contract_corpus import QUESTIONS

        questions = list(QUESTIONS)

    env = {
        **os.environ,
        "TABULA_RAG_INDEX_DIR": str(work / "index"),
        "TABULA_RAG_OUTPUT_DIR": str(work / "out"),
    }
    if not args.live:
        if args.corpus:
            print(
                "--corpus without --live has no scripted model for it; add --live",
                file=sys.stderr,
            )
            return 2
        server = _ScriptedServer(corpus)
        env["TABULA_RAG_LLM_BASE_URL"] = f"http://127.0.0.1:{server.port}/v1"
    env.setdefault("TABULA_RAG_INDEX_LLM_WAIT_S", "60")

    def run(*extra: str) -> tuple[float, subprocess.CompletedProcess[str]]:
        started = time.monotonic()
        done = subprocess.run(
            [sys.executable, str(REPO / "app.py"), *extra],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        return time.monotonic() - started, done

    print(f"corpus: {corpus}")
    elapsed, done = run("--index", str(corpus))
    print(f"index:  {elapsed:6.1f}s  exit={done.returncode}  {done.stdout.strip()[-120:]}")

    score, problems = 0, []
    print(f"\n{'#':>2} {'time':>6}  {'result':6} question")
    for number, (question, expected, citations) in enumerate(questions, start=1):
        elapsed, done = run(
            "--corpus", str(corpus), "--query-id", f"query_{number:02d}", "--query", question
        )
        target = work / "out" / f"query_{number:02d}_output.json"
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            assert set(payload) >= {"answer", "citations", "confidence"}
        except (OSError, ValueError, AssertionError):
            payload = None
            problems.append(f"q{number}: missing or malformed output file")
        if elapsed > PER_QUESTION_LIMIT_S:
            problems.append(
                f"q{number}: took {elapsed:.1f}s (limit {PER_QUESTION_LIMIT_S:.0f}s)"
            )
        ok = (
            payload is not None
            and grader_normalize(str(payload["answer"])) == grader_normalize(expected)
            and set(payload["citations"]) == set(citations)
        )
        score += POINTS if ok else 0
        print(f"{number:>2} {elapsed:5.1f}s  {'PASS' if ok else 'FAIL':6} {question[:70]}")
        if not ok and payload is not None:
            print(f"     got      {payload['answer']!r} {payload['citations']}")
            print(f"     expected {expected!r} {citations}")

    total = POINTS * len(questions)
    print(f"\nscore: {score}/{total}")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    if server:
        server.close()
    return 0 if score == total and not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
