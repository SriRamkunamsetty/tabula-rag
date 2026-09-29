"""Evaluation metrics for the RAG ablation ladder.

The brief asks for RAG that "improves accuracy and reduces hallucination."
Both halves of that claim need their own metric:

* **Answer correctness** — does the answer actually contain the expected
  content? Scored as term-recall against the expected answer's content
  words, the same lexical-overlap family used throughout this service
  (``verification.py``, ``conflicts.py``) rather than a second, inconsistent
  notion of "matches."
* **Faithfulness / unsupported sentence ratio** — already computed on every
  :class:`~tabula_rag.models.AnswerResult`; aggregated here across a dataset.
* **Abstention precision/recall** — a case marked ``expects_abstain=True`` in
  the dataset is a question the corpus genuinely cannot answer (or a
  deliberately out-of-scope one). Recall catches a system that answers
  anyway; precision catches one that abstains too readily.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tabula_rag.models import AnswerResult
from tabula_rag.verification import content_terms

__all__ = ["CaseOutcome", "EvalCounters", "score_case"]


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """Comparison of one query's result against its expected outcome."""

    case_id: str
    expects_abstain: bool
    abstained: bool
    correct: bool
    faithfulness: float
    grounded_ratio: float
    conflicts_found: int
    latency_ms: float
    total_tokens: int

    @property
    def abstention_matches_expectation(self) -> bool:
        """True when the system's abstain/answer decision matched the label."""
        return self.abstained == self.expects_abstain


def score_case(
    case_id: str, expected_answer: str | None, expects_abstain: bool, result: AnswerResult
) -> CaseOutcome:
    """Score one query result against its ground truth.

    Correctness is judged only when an answer was expected and one was
    given: an expected-abstain case is "correct" exactly when the system
    abstained (content correctness does not apply), and an
    expected-to-answer case that abstained is scored incorrect regardless of
    the (empty) content.
    """
    if expects_abstain:
        correct = result.abstained
    elif result.abstained:
        correct = False
    else:
        expected_terms = content_terms(expected_answer or "")
        answer_terms = content_terms(result.answer)
        recall = (
            len(expected_terms & answer_terms) / len(expected_terms) if expected_terms else 0.0
        )
        correct = recall >= 0.6

    return CaseOutcome(
        case_id=case_id,
        expects_abstain=expects_abstain,
        abstained=result.abstained,
        correct=correct,
        faithfulness=result.faithfulness,
        grounded_ratio=result.grounded_ratio,
        conflicts_found=len(result.conflicts),
        latency_ms=result.usage.latency_ms,
        total_tokens=result.usage.total_tokens,
    )


@dataclass
class EvalCounters:
    """Aggregates :class:`CaseOutcome` values into report-level metrics."""

    outcomes: list[CaseOutcome] = field(default_factory=list)

    def add(self, outcome: CaseOutcome) -> None:
        """Record one case outcome."""
        self.outcomes.append(outcome)

    @property
    def total(self) -> int:
        """Number of cases scored."""
        return len(self.outcomes)

    @property
    def accuracy(self) -> float:
        """Share of all cases scored correct."""
        return sum(1 for o in self.outcomes if o.correct) / self.total if self.total else 0.0

    @property
    def mean_faithfulness(self) -> float:
        """Mean faithfulness across cases that produced an answer.

        Faithfulness alone can be misleadingly high for a config that skips
        verification entirely (an ungrounded claim is never "refused," so it
        never lowers faithfulness) -- always read this alongside
        :attr:`mean_grounded_ratio`, which specifically will not be fooled.
        """
        answered = [o for o in self.outcomes if not o.abstained]
        return sum(o.faithfulness for o in answered) / len(answered) if answered else 1.0

    @property
    def mean_grounded_ratio(self) -> float:
        """Mean share of *actually verified* claims across answered cases."""
        answered = [o for o in self.outcomes if not o.abstained]
        return sum(o.grounded_ratio for o in answered) / len(answered) if answered else 0.0

    @property
    def abstention_recall(self) -> float:
        """Of cases that should abstain, the share that actually did."""
        should = [o for o in self.outcomes if o.expects_abstain]
        return sum(1 for o in should if o.abstained) / len(should) if should else 1.0

    @property
    def abstention_precision(self) -> float:
        """Of cases that did abstain, the share that should have."""
        did = [o for o in self.outcomes if o.abstained]
        return sum(1 for o in did if o.expects_abstain) / len(did) if did else 1.0

    @property
    def mean_latency_ms(self) -> float:
        """Mean end-to-end latency across all cases."""
        return sum(o.latency_ms for o in self.outcomes) / self.total if self.total else 0.0

    @property
    def total_tokens(self) -> int:
        """Sum of prompt + completion tokens across all cases."""
        return sum(o.total_tokens for o in self.outcomes)

    def to_dict(self) -> dict[str, float | int]:
        """Render the metrics as a JSON-safe dictionary."""
        return {
            "cases": self.total,
            "accuracy": round(self.accuracy, 4),
            "mean_faithfulness": round(self.mean_faithfulness, 4),
            "mean_grounded_ratio": round(self.mean_grounded_ratio, 4),
            "abstention_recall": round(self.abstention_recall, 4),
            "abstention_precision": round(self.abstention_precision, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "total_tokens": self.total_tokens,
        }
