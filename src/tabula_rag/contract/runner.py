"""Index building and query running, with every budget and failure mode accounted for.

Two invocations, mirroring the harness:

* ``--index CORPUS``: parse every file (text first, so a usable index exists within seconds),
  persist it, then read images with the vision model and persist again. Charged to the
  10-minute startup budget, so image reading waits for the model server for a bounded time
  and gives up gracefully rather than blocking forever.
* ``--query``: a brand-new process per question. Load the persisted index from disk (or, if it
  is missing or stale, rebuild the text part inline), answer, and *always* leave a valid output
  file behind: a placeholder is written first, so a crash or timeout still scores as a refusal
  rather than a malformed response.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path

from tabula_rag.contract.answer import QueryResult, answer_query
from tabula_rag.contract.config import ContractSettings
from tabula_rag.contract.corpus import (
    DocRecord,
    corpus_fingerprint,
    family_and_revision,
    walk_corpus,
)
from tabula_rag.contract.index import ContractIndex
from tabula_rag.contract.llm import ModelClient, OpenAIModelClient
from tabula_rag.contract.parsers import Segment, detect_kind, parse_file
from tabula_rag.observability import get_logger

__all__ = [
    "build_text_index",
    "index_corpus",
    "load_or_build",
    "output_path",
    "run_index",
    "run_query",
]

_log = get_logger(__name__)
EMPTY_OUTPUT = {"answer": "", "citations": [], "confidence": 0.0}


def build_text_index(corpus: Path) -> tuple[ContractIndex, list[DocRecord]]:
    """Parse every non-image file. Images are registered and returned for the vision pass."""
    index = ContractIndex(fingerprint=corpus_fingerprint(corpus) if corpus.is_dir() else "")
    pending: list[DocRecord] = []
    if not corpus.is_dir():
        index.finalize()
        return index, pending
    for path in walk_corpus(corpus):
        relative = path.relative_to(corpus).as_posix()
        family, revision, withdrawn = family_and_revision(relative)
        kind = detect_kind(path)
        record = DocRecord(
            relative, kind, family=family, revision=revision, withdrawn=withdrawn
        )
        if kind == "image":
            record.reason = "image not read yet"
            index.docs[relative] = record
            pending.append(record)
            continue
        result = parse_file(path)
        record.reason = result.skipped
        record.withdrawn = record.withdrawn or result.withdrawn
        index.add_document(record, result.segments)
    index.finalize()
    return index, pending


async def _read_images(
    corpus: Path,
    pending: list[DocRecord],
    model: ModelClient,
    settings: ContractSettings,
    reuse: ContractIndex | None,
) -> list[tuple[DocRecord, str]]:
    """Transcribe images concurrently. Returns ``(record, text)`` for the ones that worked."""
    semaphore = asyncio.Semaphore(max(1, settings.vision_concurrency))
    done: list[tuple[DocRecord, str]] = []

    async def read_one(record: DocRecord) -> None:
        previous = reuse.docs.get(record.path) if reuse else None
        if previous is not None and previous.indexed and previous.text:
            done.append((record, previous.text))
            return
        async with semaphore:
            try:
                data = (corpus / record.path).read_bytes()
            except OSError:
                record.reason = "unreadable image"
                return
            text = await model.transcribe(data, timeout_s=settings.vision_timeout_s)
        if text:
            done.append((record, text))
        else:
            record.reason = "vision model returned nothing"

    await asyncio.gather(*(read_one(r) for r in pending))
    return done


async def index_corpus(
    corpus: Path,
    model: ModelClient | None,
    settings: ContractSettings,
    *,
    save: bool = True,
    reuse: ContractIndex | None = None,
) -> ContractIndex:
    """Build (and optionally persist) the full index, images included when a model is given."""
    started = time.monotonic()
    index, pending = build_text_index(corpus)
    if save:
        _try_save(index, settings.index_dir)  # a usable index exists from here on
    if pending and model is not None:
        if await model.wait_ready(settings.index_llm_wait_s):
            for record, text in await _read_images(corpus, pending, model, settings, reuse):
                index.add_document(record, [Segment(text, "image transcription")])
                record.reason = None
        else:
            for record in pending:
                record.reason = "vision model unavailable"
        index.finalize()
        if save:
            _try_save(index, settings.index_dir)
    _log.info(
        "index_built",
        files=len(index.docs),
        indexed=sum(1 for r in index.docs.values() if r.indexed),
        chunks=len(index.chunks),
        seconds=round(time.monotonic() - started, 2),
    )
    return index


def _try_save(index: ContractIndex, directory: Path) -> None:
    try:
        index.save(directory)
    except OSError as exc:
        _log.warning("index_save_failed", error=str(exc))


def run_index(
    corpus: Path, settings: ContractSettings, model: ModelClient | None = None
) -> dict[str, int]:
    """CLI half of ``--index``. Returns summary counts; never raises."""
    owned = model is None
    client = model if model is not None else OpenAIModelClient(settings)

    async def go() -> ContractIndex:
        try:
            return await index_corpus(corpus, client, settings)
        finally:
            if owned:
                await client.aclose()

    try:
        index = asyncio.run(go())
    except Exception as exc:
        _log.error("index_failed", error=f"{type(exc).__name__}: {exc}")
        return {"files": 0, "indexed": 0, "chunks": 0}
    return {
        "files": len(index.docs),
        "indexed": sum(1 for r in index.docs.values() if r.indexed),
        "chunks": len(index.chunks),
    }


def load_or_build(corpus: Path | None, settings: ContractSettings) -> ContractIndex:
    """The persisted index if it matches the corpus, otherwise a text-only rebuild."""
    saved: ContractIndex | None = None
    try:
        saved = ContractIndex.load(settings.index_dir)
    except (OSError, ValueError, KeyError):
        saved = None
    if corpus is None or not corpus.is_dir():
        return saved or ContractIndex()
    if saved is not None and saved.fingerprint == corpus_fingerprint(corpus):
        return saved
    _log.warning("index_missing_or_stale_rebuilding_text_only")
    index, pending = build_text_index(corpus)
    if saved is not None:  # keep image transcripts we already paid for
        for record in pending:
            previous = saved.docs.get(record.path)
            if previous is not None and previous.indexed and previous.text:
                index.add_document(record, [Segment(previous.text, "image transcription")])
        index.finalize()
    return index


def output_path(settings: ContractSettings, query_id: str) -> Path:
    """``<output-dir>/<query-id>_output.json``, with path separators removed from the id."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", query_id) or "query"
    return settings.output_dir / f"{safe}_output.json"


def write_output(path: Path, payload: dict[str, object]) -> None:
    """Write JSON atomically so the harness can never read a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)
        Path(temp).replace(path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def run_query(
    corpus: Path | None,
    query_id: str,
    query: str,
    settings: ContractSettings,
    model: ModelClient | None = None,
    *,
    watchdog: bool = True,
) -> QueryResult:
    """CLI half of a question. Always leaves ``<query-id>_output.json`` behind."""
    started = time.monotonic()
    target = output_path(settings, query_id)
    write_output(
        target, dict(EMPTY_OUTPUT)
    )  # a well-formed refusal exists from the first moment
    timer: threading.Timer | None = None
    if watchdog:  # last line of defence: exit cleanly, with the placeholder already on disk
        timer = threading.Timer(settings.query_budget_s + 4.0, lambda: os._exit(0))
        timer.daemon = True
        timer.start()

    result = QueryResult()
    owned = model is None
    client = model if model is not None else OpenAIModelClient(settings)

    async def go() -> QueryResult:
        try:
            index = load_or_build(corpus, settings)
            return await asyncio.wait_for(
                answer_query(
                    index,
                    client,
                    query,
                    top_k=settings.top_k,
                    llm_timeout_s=settings.llm_timeout_s,
                    deadline=started + settings.query_budget_s,
                ),
                timeout=max(1.0, settings.query_budget_s - (time.monotonic() - started)),
            )
        finally:
            if owned:
                await client.aclose()

    try:
        result = asyncio.run(go())
    except Exception as exc:
        _log.error("query_failed", error=f"{type(exc).__name__}: {exc}")
        result.trace.append(f"failed: {type(exc).__name__}")
    finally:
        if timer is not None:
            timer.cancel()
    write_output(target, result.to_output())
    return result
