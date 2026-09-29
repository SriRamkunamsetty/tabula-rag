"""The contract-mode index: chunks, an inverted BM25 index, and disk persistence.

Why a new index rather than the service's ``CorpusIndex``: the graded harness runs a
*fresh process per question*, so the index must load from disk in well under a second,
and it must ingest logs and tables of tens of thousands of rows. This one keeps postings
lists (search touches only documents that contain a query term), builds in linear time,
and round-trips through three plain files.

File names are indexed alongside content: a question about "the asset label" should be
able to find ``support/asset_label.jpg`` even when the transcription never says "asset".
"""

from __future__ import annotations

import json
import math
import pickle
import re
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tabula_rag.contract.corpus import DocRecord, mark_superseded
from tabula_rag.contract.parsers import Segment
from tabula_rag.contract.text import extract_identifiers, tokenize

__all__ = ["ContractIndex", "Hit", "IndexedChunk", "LexicalIndex", "chunk_segments"]

INDEX_VERSION = 1
SUPERSEDED_PENALTY = 0.4
_WORD = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class IndexedChunk:
    """A retrievable slice of one file."""

    chunk_id: int
    path: str
    locator: str
    text: str


@dataclass(frozen=True, slots=True)
class Hit:
    """A chunk with its retrieval score."""

    chunk: IndexedChunk
    score: float


def chunk_segments(
    path: str, segments: list[Segment], *, target_tokens: int = 170, start_id: int = 0
) -> list[IndexedChunk]:
    """Split segments into line-aligned, token-budgeted chunks.

    Line-aligned because logs, code and tables are line-oriented: splitting on blank
    lines (as prose chunkers do) would turn a 20 MB log into a single chunk. One line of
    overlap keeps a fact that straddles a boundary retrievable from either side.
    """
    chunks: list[IndexedChunk] = []
    next_id = start_id
    for segment in segments:
        # (line number, text) pieces; an over-long line is split into word windows
        pieces: list[tuple[int, str]] = []
        for number, line in enumerate((ln for ln in segment.text.split("\n") if ln.strip()), 1):
            words = _WORD.findall(line)
            if len(words) > target_tokens:
                pieces.extend(
                    (number, " ".join(words[i : i + target_tokens]))
                    for i in range(0, len(words), target_tokens)
                )
            else:
                pieces.append((number, line))
        groups: list[list[tuple[int, str]]] = []
        current: list[tuple[int, str]] = []
        current_tokens = 0
        fresh = 0  # pieces in ``current`` that are not carried-over overlap
        for number, piece in pieces:
            size = len(_WORD.findall(piece))
            if fresh and current_tokens + size > target_tokens:
                groups.append(current)
                current = current[-1:]  # one piece of overlap
                current_tokens = len(_WORD.findall(current[0][1]))
                fresh = 0
            current.append((number, piece))
            current_tokens += size
            fresh += 1
        if fresh:
            groups.append(current)
        for group in groups:
            where = segment.locator
            if len(groups) > 1:
                where = f"{segment.locator} lines {group[0][0]}-{group[-1][0]}"
            chunks.append(
                IndexedChunk(next_id, path, where.strip(), "\n".join(text for _, text in group))
            )
            next_id += 1
    return chunks


@dataclass
class LexicalIndex:
    """BM25 over an inverted index."""

    k1: float = 1.4
    b: float = 0.75
    _postings: dict[str, list[tuple[int, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _lengths: list[int] = field(default_factory=list)
    _avg_length: float = 0.0

    def __len__(self) -> int:
        """Number of indexed documents."""
        return len(self._lengths)

    def state(self) -> dict[str, Any]:
        """Everything needed to restore this index without re-tokenising the corpus."""
        return {
            "k1": self.k1,
            "b": self.b,
            "postings": dict(self._postings),
            "lengths": self._lengths,
            "avg": self._avg_length,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> LexicalIndex:
        """Rebuild an index from :meth:`state`."""
        index = cls(k1=state["k1"], b=state["b"])
        index._postings = defaultdict(list, state["postings"])
        index._lengths = list(state["lengths"])
        index._avg_length = float(state["avg"])
        return index

    def add(self, tokens: list[str]) -> int:
        """Index one document and return its position."""
        doc = len(self._lengths)
        counts: dict[str, int] = defaultdict(int)
        for token in tokens:
            counts[token] += 1
        for token, freq in counts.items():
            self._postings[token].append((doc, freq))
        self._lengths.append(len(tokens))
        return doc

    def finalize(self) -> None:
        """Compute corpus statistics. Call once after the last ``add``."""
        self._avg_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0

    def search(self, query_tokens: list[str], top_k: int) -> list[tuple[int, float]]:
        """Return ``(doc, score)`` pairs, best first. No matching term means no results."""
        total = len(self._lengths)
        if not total:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = math.log(1.0 + (total - len(postings) + 0.5) / (len(postings) + 0.5))
            for doc, freq in postings:
                norm = 1.0 - self.b + self.b * (self._lengths[doc] / (self._avg_length or 1.0))
                scores[doc] += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * norm)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]


@dataclass
class ContractIndex:
    """Documents, chunks and lookups for one indexed corpus."""

    docs: dict[str, DocRecord] = field(default_factory=dict)
    chunks: list[IndexedChunk] = field(default_factory=list)
    fingerprint: str = ""
    identifier_files: dict[str, list[str]] = field(default_factory=dict)
    _lexical: LexicalIndex = field(default_factory=LexicalIndex, repr=False)

    # ------------------------------------------------------------------ building
    def add_document(self, record: DocRecord, segments: list[Segment]) -> None:
        """Register a document; if it produced text, chunk and index it."""
        self.docs[record.path] = record
        if not segments:
            return
        record.indexed = True
        record.text = "\n".join(s.text for s in segments)
        self.chunks.extend(chunk_segments(record.path, segments, start_id=len(self.chunks)))

    def finalize(self) -> None:
        """Compute revision relations, identifier map and the lexical index."""
        mark_superseded(list(self.docs.values()))
        by_id: dict[str, set[str]] = defaultdict(set)
        for record in self.docs.values():
            if record.indexed:
                for identifier in extract_identifiers(record.text):
                    by_id[identifier].add(record.path)
        self.identifier_files = {i: sorted(p) for i, p in by_id.items() if len(p) <= 12}
        self._lexical = LexicalIndex()
        for chunk in self.chunks:
            self._lexical.add(tokenize(f"{chunk.path}\n{chunk.text}"))
        self._lexical.finalize()

    # ------------------------------------------------------------------ retrieval
    def search(self, query: str, top_k: int = 12, per_file: int = 3) -> list[Hit]:
        """Best chunks for ``query``, at most ``per_file`` from any one file.

        Chunks of a superseded document are demoted, not removed: a question that really
        is about the old revision can still reach it, but the current one wins ties.
        """
        raw = self._lexical.search(tokenize(query), top_k=top_k * 6)
        hits: list[Hit] = []
        for position, score in raw:
            chunk = self.chunks[position]
            record = self.docs.get(chunk.path)
            if record is not None and record.superseded:
                score *= SUPERSEDED_PENALTY
            hits.append(Hit(chunk, score))
        hits.sort(key=lambda h: h.score, reverse=True)
        taken: dict[str, int] = defaultdict(int)
        result: list[Hit] = []
        for hit in hits:
            if taken[hit.chunk.path] >= per_file:
                continue
            taken[hit.chunk.path] += 1
            result.append(hit)
            if len(result) >= top_k:
                break
        return result

    # ------------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        """Write the index atomically (a reader never sees a half-written index)."""
        directory.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".index-", dir=directory.parent))
        manifest = {
            "version": INDEX_VERSION,
            "fingerprint": self.fingerprint,
            "docs": [
                {k: v for k, v in asdict(r).items() if k != "text"}
                | {"revision": list(r.revision)}
                for r in self.docs.values()
            ],
        }
        (staging / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with (staging / "texts.jsonl").open("w", encoding="utf-8") as handle:
            for record in self.docs.values():
                if record.indexed:
                    handle.write(json.dumps({"path": record.path, "text": record.text}) + "\n")
        with (staging / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(asdict(chunk)) + "\n")
        (staging / "identifiers.json").write_text(
            json.dumps(self.identifier_files), encoding="utf-8"
        )
        # The postings are the slow part to rebuild (tokenising every chunk took ~7 s for a
        # 16 MB corpus, on *every* question). The index directory is written and read only by
        # this container, so a pickle is an acceptable, compact and fast format for it.
        with (staging / "lexical.pkl").open("wb") as handle:
            pickle.dump(self._lexical.state(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        if directory.exists():
            _remove_tree(directory)
        staging.rename(directory)

    @classmethod
    def load(cls, directory: Path) -> ContractIndex:
        """Load a saved index; raises ``FileNotFoundError``/``ValueError`` if unusable."""
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("version") != INDEX_VERSION:
            raise ValueError("index version mismatch")
        index = cls(fingerprint=manifest["fingerprint"])
        texts: dict[str, str] = {}
        with (directory / "texts.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                texts[row["path"]] = row["text"]
        for doc in manifest["docs"]:
            record = DocRecord(**{**doc, "revision": tuple(doc["revision"])})
            record.text = texts.get(record.path, "")
            index.docs[record.path] = record
        with (directory / "chunks.jsonl").open(encoding="utf-8") as handle:
            index.chunks = [IndexedChunk(**json.loads(line)) for line in handle]
        index.identifier_files = json.loads(
            (directory / "identifiers.json").read_text(encoding="utf-8")
        )
        index._lexical = _load_lexical(directory, len(index.chunks)) or _rebuild_lexical(
            index.chunks
        )
        return index


def _rebuild_lexical(chunks: list[IndexedChunk]) -> LexicalIndex:
    lexical = LexicalIndex()
    for chunk in chunks:
        lexical.add(tokenize(f"{chunk.path}\n{chunk.text}"))
    lexical.finalize()
    return lexical


def _load_lexical(directory: Path, expected_chunks: int) -> LexicalIndex | None:
    """The persisted lexical index, or ``None`` if it is missing, damaged or out of step."""
    try:
        with (directory / "lexical.pkl").open("rb") as handle:
            lexical = LexicalIndex.from_state(pickle.load(handle))
    except Exception:
        return None
    return lexical if len(lexical) == expected_chunks else None


def _remove_tree(directory: Path) -> None:
    import shutil

    shutil.rmtree(directory, ignore_errors=True)
