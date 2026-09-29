"""Detect disagreeing facts across retrieved sources.

Two chunks can both be individually well-grounded and still contradict each
other — the base Hexfall rulebook states a 60-move turn limit, and the
errata sheet revises it to 80. A claim citing only the base rulebook would
pass the coverage check in ``verification.py`` while being *stale*. This
module is what catches that: it looks for a number near the same
query-relevant context in two different retrieved chunks and flags a
disagreement rather than silently letting the answer pick whichever chunk
the generator happened to draw from.

This is a narrower, more mechanical check than full entailment-based
contradiction detection — it looks for differing numbers near shared context
terms, not arbitrary logical conflicts — and that scope is deliberate. See
``docs/adr/004-scoped-conflict-detection.md`` for why a narrow, reliable
check was chosen over a broad, unreliable one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tabula_rag.models import RetrievedChunk, SourceConflict

__all__ = ["find_conflicts"]

_NUMBER_RE = re.compile(r"\b\d{1,4}\b")
_WINDOW = 6
"""Words of context kept on each side of a matched number, for the context key."""


@dataclass(frozen=True, slots=True)
class _NumberMention:
    value: str
    context_key: frozenset[str]
    source: str
    sentence: str


def _mentions(text: str, source_label: str) -> list[_NumberMention]:
    words = text.split()
    lower_words = [w.strip(".,;:()").lower() for w in words]
    mentions: list[_NumberMention] = []
    for i, word in enumerate(lower_words):
        if not _NUMBER_RE.fullmatch(word):
            continue
        start, end = max(0, i - _WINDOW), min(len(lower_words), i + _WINDOW + 1)
        context = frozenset(
            w for j, w in enumerate(lower_words[start:end]) if j + start != i and len(w) > 2
        )
        sentence_start = text.rfind(".", 0, sum(len(w) + 1 for w in words[:i])) + 1
        mentions.append(
            _NumberMention(
                value=word,
                context_key=context,
                source=source_label,
                sentence=text[sentence_start:].split(".")[0].strip()[:160],
            )
        )
    return mentions


def find_conflicts(
    retrieved: list[RetrievedChunk], *, min_shared_context: int = 2
) -> list[SourceConflict]:
    """Look for two retrieved chunks that state different numbers in similar context.

    Two number mentions are compared if they come from **different source
    documents** and share at least ``min_shared_context`` nearby content
    words — e.g. both mentions sit near "turn", "limit", "move", "draw". If
    their numeric values differ, that pair is reported as a conflict.

    Args:
        retrieved: The chunks retrieval surfaced for one query, across
            possibly multiple source documents.
        min_shared_context: How many nearby words must match before two
            mentions are considered "about the same fact." Higher values
            reduce false positives at the cost of missing genuine conflicts
            phrased very differently.

    Returns:
        One :class:`SourceConflict` per disagreeing pair found. An empty
        corpus, a single-source result set, or a result set with no shared
        context between any two numeric mentions all correctly return no
        conflicts — this function does not force a finding.
    """
    all_mentions: list[_NumberMention] = []
    for item in retrieved:
        label = item.chunk.citation_label
        all_mentions.extend(_mentions(item.chunk.text, label))

    conflicts: list[SourceConflict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for i, a in enumerate(all_mentions):
        for b in all_mentions[i + 1 :]:
            if a.source == b.source or a.value == b.value:
                continue
            shared = a.context_key & b.context_key
            if len(shared) < min_shared_context:
                continue
            key_a, key_b = sorted([f"{a.source}:{a.value}", f"{b.source}:{b.value}"])
            pair_key = (key_a, key_b)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            conflicts.append(
                SourceConflict(
                    claim_text=" / ".join(sorted(shared)),
                    value_a=a.value,
                    source_a=a.source,
                    value_b=b.value,
                    source_b=b.source,
                )
            )
    return conflicts
