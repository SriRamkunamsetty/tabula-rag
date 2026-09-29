"""Text helpers shared by indexing, retrieval and verification in contract mode.

Everything here is pure and deterministic. The grader compares answers after
uppercasing and removing whitespace and the characters ``- . · _``; verification
uses the same normalisation, so "is the answer really in this file" is judged the
way the grader will judge the answer itself.
"""

from __future__ import annotations

import re

__all__ = [
    "content_terms",
    "extract_identifiers",
    "grader_normalize",
    "normalized_contains",
    "strip_trailing_unit",
    "tokenize",
]

_MIDDLE_DOT = chr(0x00B7)
_DASHES = chr(0x2010) + "-" + chr(0x2015)  # hyphen ... horizontal bar, as a class range
_GRADER_STRIP = re.compile(rf"[\s\-._{_MIDDLE_DOT}{_DASHES}]+")
_WORD = re.compile(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*")
_SPLIT = re.compile(r"[-_.]")
_IDENT = re.compile(
    r"\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b"
)
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "but",
        "if",
        "then",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "by",
        "with",
        "from",
        "into",
        "not",
        "no",
        "do",
        "does",
        "did",
        "can",
        "may",
        "must",
        "will",
        "shall",
        "has",
        "have",
        "had",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
    ]
)
_UNIT = re.compile(
    r"^([+-]?\d+(?:[.,]\d+)*)\s*"
    r"(?:°\s*[cf]?|%|degrees?(?:\s*(?:c|f|celsius|fahrenheit))?|celsius|fahrenheit|"
    r"milliseconds?|seconds?|secs?|ms|s|volts?|v|watts?|w|amps?|a|[kmg]?hz|[kmgt]b|c|f)$",
    re.IGNORECASE,
)


def grader_normalize(text: str) -> str:
    """Normalise like the official grader: uppercase, drop whitespace and ``- . · _``."""
    return _GRADER_STRIP.sub("", text).upper()


def normalized_contains(haystack: str, needle: str) -> bool:
    """True if ``needle`` occurs in ``haystack`` after grader normalisation."""
    target = grader_normalize(needle)
    return bool(target) and target in grader_normalize(haystack)


def _stem(token: str) -> str:
    """Strip a plural ``s`` from purely alphabetic tokens, nothing cleverer."""
    if len(token) > 3 and token.isalpha() and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Identifier-aware tokeniser for lexical retrieval.

    ``TQ-40`` yields ``tq``, ``40`` *and* the compact ``tq40``, so it matches whether
    the query writes the identifier with or without its separator. Dotted numbers
    such as ``4.3.2`` yield their parts only (a compact ``432`` would be noise).
    """
    tokens: list[str] = []
    for match in _WORD.finditer(text):
        word = match.group(0).lower()
        parts = [p for p in _SPLIT.split(word) if p]
        tokens.extend(_stem(p) for p in parts)
        if len(parts) > 1 and any(not p.isdigit() for p in parts):
            tokens.append("".join(parts))
    return tokens


def content_terms(text: str) -> set[str]:
    """Distinct non-stopword tokens, used for relevance checks."""
    return {t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1}


def extract_identifiers(text: str, *, min_length: int = 4, limit: int = 20_000) -> set[str]:
    """Distinctive identifiers (ticket ids, part numbers, error codes), grader-normalised.

    An identifier is a token that mixes letters and digits (``ORR-1847``, ``E7731``,
    ``ORR-FAN-2214-B``). Pure numbers and pure words are not identifiers.
    """
    found: set[str] = set()
    for match in _IDENT.finditer(text):
        normalized = grader_normalize(match.group(0))
        if len(normalized) >= min_length:
            found.add(normalized)
            if len(found) >= limit:
                break
    return found


def strip_trailing_unit(answer: str) -> str:
    """Reduce ``94 °C`` / ``180 s`` style answers to the bare number.

    Only a *standalone* number with a unit is touched. Versions (``4.3.2``), quarters
    (``Q3 FY27``), revisions (``REV-C2``) and identifiers are returned unchanged.
    """
    match = _UNIT.match(answer.strip())
    return match.group(1) if match else answer.strip()
