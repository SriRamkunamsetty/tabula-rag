"""BM25Okapi lexical retrieval.

Implemented directly rather than pulled in as a dependency: BM25 is a small,
well-specified algorithm, and owning it means the tokenizer, IDF smoothing,
and scoring are all visible and unit-tested in this repository rather than
living in a black-box import. See ``tests/test_rag.py::TestBM25`` for the
correctness checks (IDF favours rare terms, exact term matches rank above
partial ones, an empty query returns no matches).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

__all__ = ["BM25Index"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-token tokenizer.

    Deliberately simple: BM25's value comes from term-frequency statistics,
    not from a sophisticated tokenizer. Rule text in this corpus is short,
    formal English, where a regex split loses little compared to a full NLP
    pipeline.
    """
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    """An in-memory BM25Okapi index over a fixed set of documents.

    Args:
        k1: Term-frequency saturation parameter. Higher values let repeated
            terms keep contributing to the score for longer.
        b: Length-normalisation parameter. 0 disables length normalisation
            entirely; 1 fully normalises by document length.
    """

    k1: float = 1.5
    b: float = 0.75
    _doc_ids: list[str] = field(default_factory=list, init=False)
    _doc_term_freqs: list[Counter[str]] = field(default_factory=list, init=False)
    _doc_lengths: list[int] = field(default_factory=list, init=False)
    _doc_freq: Counter[str] = field(default_factory=Counter, init=False)
    _avg_doc_length: float = field(default=0.0, init=False)
    _idf_cache: dict[str, float] = field(default_factory=dict, init=False)

    def __len__(self) -> int:
        """Number of documents currently indexed."""
        return len(self._doc_ids)

    def add(self, doc_id: str, text: str) -> None:
        """Add one document to the index. Rebuilds IDF and average length."""
        tokens = _tokenize(text)
        term_freqs = Counter(tokens)
        self._doc_ids.append(doc_id)
        self._doc_term_freqs.append(term_freqs)
        self._doc_lengths.append(len(tokens))
        for term in term_freqs:
            self._doc_freq[term] += 1
        self._idf_cache.clear()
        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        )

    def _idf(self, term: str) -> float:
        """Robertson-Spärck Jones IDF with the standard +1 smoothing.

        The smoothing keeps the score finite and non-negative even for a term
        that appears in every document, rather than letting it go to (or
        below) zero.
        """
        if term in self._idf_cache:
            return self._idf_cache[term]
        n = len(self._doc_ids)
        df = self._doc_freq.get(term, 0)
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        self._idf_cache[term] = idf
        return idf

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(doc_id, score)`` pairs, highest score first.

        An empty or entirely out-of-vocabulary query returns an empty list
        rather than an arbitrary ranking over all documents — no lexical
        signal means BM25 has nothing to contribute for this query, and hybrid
        fusion (see ``retrieval/fusion.py``) is built to handle that.
        """
        query_terms = _tokenize(query)
        if not query_terms or not self._doc_ids:
            return []

        scores = [0.0] * len(self._doc_ids)
        for term in set(query_terms):
            if self._doc_freq.get(term, 0) == 0:
                continue
            idf = self._idf(term)
            for i, term_freqs in enumerate(self._doc_term_freqs):
                freq = term_freqs.get(term, 0)
                if freq == 0:
                    continue
                length_norm = (
                    1.0
                    - self.b
                    + self.b
                    * (
                        self._doc_lengths[i] / self._avg_doc_length
                        if self._avg_doc_length
                        else 1.0
                    )
                )
                scores[i] += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * length_norm)

        ranked = sorted(
            (
                (doc_id, score)
                for doc_id, score in zip(self._doc_ids, scores, strict=True)
                if score > 0
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_k]
