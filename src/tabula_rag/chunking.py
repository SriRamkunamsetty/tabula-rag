"""Turn a source document into retrievable chunks.

Two things matter for the rest of the pipeline and are both handled here:

* **Section awareness.** Markdown-style headings become the ``section``
  metadata on every chunk beneath them, so a citation can say "Hexfall
  Rulebook §6 Towers" instead of just a document id — this is what makes a
  citation something a person can actually go check.
* **Token-budget splitting.** A section longer than the target chunk size is
  split on paragraph boundaries with a small overlap, so a rule that spans a
  paragraph break is not silently cut in half between two chunks that never
  retrieve together.

A whitespace tokenizer approximation is used for the token budget rather than
a model-specific tokenizer: chunk boundaries only need to be *roughly* the
right size, and avoiding a tokenizer dependency keeps this module usable
before any model is chosen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tabula_rag.models import Chunk

__all__ = ["chunk_document"]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class _Section:
    heading: str
    body: str


def _split_sections(text: str) -> list[_Section]:
    """Split a markdown document into (heading, body) sections.

    Text before the first heading is kept under an empty heading rather than
    dropped, so a document with no headings at all still chunks correctly.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [_Section("", text)]

    sections: list[_Section] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(_Section("", preamble))

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append(_Section(heading, body))
    return sections


def _approx_tokens(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _split_body(body: str, target_tokens: int, overlap_tokens: int) -> list[str]:
    """Split a section body into paragraph-aligned, token-budgeted pieces."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if not paragraphs:
        return []

    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        p_tokens = _approx_tokens(paragraph)
        if current and current_tokens + p_tokens > target_tokens:
            pieces.append("\n\n".join(current))
            # carry the tail of the previous piece forward as overlap context
            overlap: list[str] = []
            overlap_count = 0
            for prior in reversed(current):
                overlap.insert(0, prior)
                overlap_count += _approx_tokens(prior)
                if overlap_count >= overlap_tokens:
                    break
            current = overlap
            current_tokens = overlap_count
        current.append(paragraph)
        current_tokens += p_tokens

    if current:
        pieces.append("\n\n".join(current))
    return pieces


def chunk_document(
    document_id: str,
    text: str,
    *,
    title: str = "",
    target_tokens: int = 180,
    overlap_tokens: int = 30,
) -> list[Chunk]:
    """Chunk a markdown document into retrievable, cited units.

    Args:
        document_id: Stable identifier for the source document.
        text: Full document text, markdown headings used for section context.
        title: Human-readable document title, used in citation labels.
        target_tokens: Approximate chunk size, in whitespace-delimited tokens.
        overlap_tokens: Approximate carry-over between consecutive chunks in
            the same section, so a rule spanning a paragraph break is not
            split without any shared context.

    Returns:
        Chunks in document order, each with a stable ``chunk_id`` of the form
        ``{document_id}::{position:04d}``.
    """
    position = 0
    chunks: list[Chunk] = []
    for section in _split_sections(text):
        for piece in _split_body(section.body, target_tokens, overlap_tokens):
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}::{position:04d}",
                    document_id=document_id,
                    document_title=title,
                    section=section.heading,
                    position=position,
                    text=piece,
                )
            )
            position += 1
    return chunks
