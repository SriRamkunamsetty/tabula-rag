"""Answering one question in contract mode: retrieve, ask, then verify deterministically.

The model proposes; this module disposes. Three of the challenge's rules are enforced here
in plain code rather than trusted to a prompt:

* **Grounded or empty.** The answer must literally occur (after the grader's own
  normalisation) in a file the model is citing. A value the model "remembers" or invents
  fails this and becomes an empty answer, which is the required refusal.
* **Citations are an exact set.** The model is asked to cite only files whose removal would
  make the answer impossible; the code then removes superseded revisions and files that add
  nothing, while keeping genuine chain links (a file that supplied an identifier used to
  look the value up elsewhere).
* **Superseded documents lose.** Retrieval demotes them and the prompt labels them.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote

from tabula_rag.contract.index import ContractIndex, Hit
from tabula_rag.contract.llm import ModelClient
from tabula_rag.contract.text import (
    content_terms,
    extract_identifiers,
    grader_normalize,
    normalized_contains,
    strip_trailing_unit,
)

__all__ = [
    "ANSWER_SCHEMA",
    "QueryResult",
    "Verdict",
    "answer_query",
    "build_prompt",
    "clean_answer",
    "resolve_paths",
    "verify_and_cite",
]

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reasoning", "answerable", "answer", "sources"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You answer questions about a private collection of documents, using ONLY the excerpts provided.
The collection describes products and systems that do not exist in your training data. Anything you "remember" about them is invented, so never use it.

Rules
1. Answer with the VALUE ONLY: a number, part number, version, quarter, code. No sentence, no restating the question, no explanation, and no unit unless the unit is part of the identifier. Give the COMPLETE value: keep any qualifier that identifies it (a quarter keeps its fiscal year, for example "Q3 FY27"; a revision keeps its prefix, for example "REV-C2").
2. Copy the value exactly as written in the excerpt.
3. If the excerpts do not contain the answer, set answerable=false, answer="" and sources=[]. Never guess. A value that is redacted, missing, or described as unavailable is not an answer.
4. Prefer files marked "current". A file marked "superseded" is only correct if the question explicitly asks about that older revision.
5. "sources" lists the file paths, exactly as written after FILE, that the answer depends on. For every file ask: "if this file were removed, could I still give the same answer?" If yes, leave it out. A file that merely discusses the same topic, or repeats the same value, is not a source. If you needed one file to obtain an identifier (a ticket number, error code, part number) and a second file to look up the value for that identifier, list BOTH files.
6. "reasoning" is one or two short sentences."""


@dataclass
class QueryResult:
    """What gets written to ``<query-id>_output.json``."""

    answer: str = ""
    citations: list[str] = field(default_factory=list)
    confidence: float = 0.0
    trace: list[str] = field(default_factory=list)

    def to_output(self) -> dict[str, Any]:
        """The exact JSON shape the harness reads: all three keys, every time."""
        if not self.answer or not self.citations:
            return {"answer": "", "citations": [], "confidence": 0.0}
        return {
            "answer": self.answer,
            "citations": list(self.citations),
            "confidence": round(max(0.0, min(1.0, self.confidence)), 3),
        }


@dataclass
class Verdict:
    """Result of the deterministic checks, before anything is written."""

    answer: str
    citations: list[str]
    confidence: float
    reason: str


_QUOTES = "`'\"" + "".join(chr(c) for c in (0x201C, 0x201D, 0x2018, 0x2019)) + " "


def clean_answer(raw: str) -> str:
    """Reduce a model answer to a bare value: one line, no wrapping quotes, no unit."""
    text = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    previous = None
    while text != previous:  # quotes and sentence punctuation can nest: "94 C".
        previous = text
        text = text.strip(_QUOTES).rstrip(".;, ")
    text = re.sub(r"^(?:answer|value)\s*[:=]\s*", "", text, flags=re.IGNORECASE)
    return strip_trailing_unit(text)


def resolve_paths(index: ContractIndex, cited: list[str]) -> list[str]:
    """Map whatever paths the model wrote onto real, indexed corpus paths (in order, unique)."""
    indexed = [p for p, r in index.docs.items() if r.indexed]
    exact = set(indexed)
    lowered = {p.lower(): p for p in indexed}
    by_name: dict[str, list[str]] = {}
    for path in indexed:
        by_name.setdefault(path.rsplit("/", 1)[-1].lower(), []).append(path)

    def resolve_one(raw: str) -> str | None:
        candidate = unquote(raw).strip().strip("`'\" ").replace("\\", "/")
        candidate = re.sub(r"^(?:/?app/)?(?:corpus/)?(?:\./)?", "", candidate).lstrip("/")
        if candidate in exact:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
        same_name = by_name.get(candidate.rsplit("/", 1)[-1].lower(), [])
        return same_name[0] if len(same_name) == 1 else None

    resolved: list[str] = []
    for raw in cited:
        found = resolve_one(str(raw))
        if found and found not in resolved:
            resolved.append(found)
    return resolved


def _file_ids(index: ContractIndex, path: str) -> set[str]:
    return {i for i, files in index.identifier_files.items() if path in files}


def build_prompt(index: ContractIndex, question: str, hits: list[Hit]) -> str:
    """Excerpts grouped by file (best file first), each file labelled current/superseded."""
    order: list[str] = []
    grouped: dict[str, list[Hit]] = {}
    for hit in hits:
        if hit.chunk.path not in grouped:
            order.append(hit.chunk.path)
            grouped[hit.chunk.path] = []
        grouped[hit.chunk.path].append(hit)
    blocks: list[str] = []
    for path in order:
        record = index.docs.get(path)
        if record is not None and record.superseded:
            newer = f" by {record.superseded_by}" if record.superseded_by else ""
            status = f"superseded{newer}"
        else:
            status = "current"
        parts = [f"[FILE {path} | {status}]"]
        for hit in grouped[path]:
            where = f" ({hit.chunk.locator})" if hit.chunk.locator else ""
            parts.append(f"--{where}\n{hit.chunk.text}")
        blocks.append("\n".join(parts))
    return f"Question: {question}\n\nExcerpts:\n\n" + "\n\n".join(blocks)


def expand_with_identifiers(
    index: ContractIndex, question: str, hits: list[Hit], top_k: int
) -> list[Hit]:
    """Second retrieval round for multi-step questions.

    Identifiers found in the best excerpts (a ticket number in a log line, say) that the
    question did not already contain are searched for directly, so the file that holds the
    *value for that identifier* is retrieved even though the question never mentions it.
    """
    question_norm = grader_normalize(question)
    found: dict[str, int] = {}
    for hit in hits[:5]:
        for identifier in extract_identifiers(hit.chunk.text):
            if identifier in question_norm:
                continue
            files = index.identifier_files.get(identifier)
            if files:  # only identifiers that occur somewhere else too can link anything
                found[identifier] = len(files)
    if not found:
        return hits
    chosen = sorted(found, key=lambda i: (found[i], i))[:6]
    extra = index.search(" ".join(chosen) + " " + question, top_k=top_k)
    merged = list(hits)
    seen = {h.chunk.chunk_id for h in merged}
    for hit in extra:
        if hit.chunk.chunk_id not in seen:
            merged.append(hit)
            seen.add(hit.chunk.chunk_id)
    return merged[: top_k + 6]


def verify_and_cite(
    index: ContractIndex,
    question: str,
    hits: list[Hit],
    raw_answer: str,
    raw_sources: list[str],
) -> Verdict:
    """Apply the grounding veto and prune citations to the necessary set."""
    answer = clean_answer(raw_answer)
    if not answer:
        return Verdict("", [], 0.0, "empty answer")
    question_terms = content_terms(question)
    needed_overlap = 1 if len(question_terms) <= 3 else 2

    def supports(text: str) -> bool:
        return (
            normalized_contains(text, answer)
            and len(question_terms & content_terms(text)) >= needed_overlap
        )

    # 1. Which files really contain the value, judged on the excerpts the model was shown?
    supported: list[str] = []
    for hit in hits:
        if hit.chunk.path not in supported and supports(hit.chunk.text):
            supported.append(hit.chunk.path)
    grounded_in_excerpt = bool(supported)
    if not supported:
        # Fall back to the whole text of retrieved files (the value may sit in a chunk
        # that was not shown).
        for hit in hits:
            record = index.docs.get(hit.chunk.path)
            if record and hit.chunk.path not in supported and supports(record.text):
                supported.append(hit.chunk.path)
    if not supported:
        return Verdict("", [], 0.0, "answer not found in any retrieved file")
    if grounded_in_excerpt is False and normalized_contains(question, answer):
        return Verdict("", [], 0.0, "answer only echoes the question")

    # 2. Files the model cited that really exist, and which of them carry the value.
    cited = resolve_paths(index, raw_sources)
    value_files = [p for p in cited if p in supported] or supported[:1]
    current = [p for p in value_files if not (index.docs[p].superseded)]
    if current:
        value_files = current

    # 3. Chain links: cited files that hold an identifier shared with a value file and not in
    #    the question, i.e. files the answer needed to *find* the value.
    question_norm = grader_normalize(question)
    value_ids: set[str] = set().union(*(_file_ids(index, p) for p in value_files))
    links: list[str] = []
    for path in cited:
        if path in value_files or index.docs[path].superseded:
            continue
        shared = _file_ids(index, path) & value_ids
        if any(identifier not in question_norm for identifier in shared):
            links.append(path)

    citations = [*value_files, *links]
    confidence = 0.9 if grounded_in_excerpt else 0.7
    if any(index.docs[p].superseded for p in citations):
        confidence -= 0.2
    return Verdict(answer, citations, confidence, "grounded")


async def answer_query(
    index: ContractIndex,
    model: ModelClient,
    question: str,
    *,
    top_k: int,
    llm_timeout_s: float,
    deadline: float,
) -> QueryResult:
    """Answer one question. Never raises; failure of any kind yields an empty answer."""
    result = QueryResult()
    hits = index.search(question, top_k=top_k)
    result.trace.append(f"retrieved {len(hits)} chunks")
    if not hits:
        result.trace.append("nothing retrieved: abstain")
        return result
    hits = expand_with_identifiers(index, question, hits, top_k)
    result.trace.append(
        f"context: {len({h.chunk.path for h in hits})} files, {len(hits)} chunks"
    )

    remaining = deadline - time.monotonic()
    if remaining < 2.0:
        result.trace.append("no time left for the model: abstain")
        return result
    reply = await model.chat_json(
        SYSTEM_PROMPT,
        build_prompt(index, question, hits),
        ANSWER_SCHEMA,
        max_tokens=400,
        timeout_s=min(llm_timeout_s, remaining - 1.0),
    )
    if reply is None:
        result.trace.append("model gave no usable output: abstain")
        return result
    if not reply.get("answerable") or not str(reply.get("answer", "")).strip():
        result.trace.append("model says the collection does not answer this: abstain")
        return result
    sources = [str(s) for s in reply.get("sources") or []]
    verdict = verify_and_cite(index, question, hits, str(reply.get("answer", "")), sources)
    result.trace.append(f"verification: {verdict.reason}")
    if verdict.answer:
        result.answer, result.citations, result.confidence = (
            verdict.answer,
            verdict.citations,
            verdict.confidence,
        )
    return result
