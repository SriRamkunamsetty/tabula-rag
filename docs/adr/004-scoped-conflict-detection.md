# ADR 4 — Conflict detection is scoped to numeric/date disagreement, not general contradiction

## Status
Accepted

## Context
Two retrieved chunks can each individually pass claim verification and still
disagree with each other — the base Hexfall rulebook states a 60-move turn
limit; its errata sheet revises this to 80. A claim citing only the base
rulebook is well-grounded by every check in `verification.py` and is still
stale. Something needs to catch cross-source disagreement specifically,
since single-claim verification structurally cannot: it only ever checks a
claim against its own citation.

General contradiction detection ("do these two passages logically conflict")
is an open, unsolved NLP problem even for trained models. A narrower,
reliable check was preferred over a broad, unreliable one.

## Decision
`conflicts.py::find_conflicts` looks specifically for two numeric mentions,
from different source documents, that share enough nearby context words to
plausibly be "about the same fact," and reports them as a conflict when the
numbers differ. It does not attempt to detect conflicts that aren't numeric
or dated (e.g. two chunks disagreeing about which stone type can capture
which — a real possible conflict this method cannot see).

## Consequences
- **Positive.** High precision on the failure mode it targets. The Hexfall
  rulebook/errata turn-limit conflict is caught reliably and is exercised by
  `tests/test_rag.py::TestConflicts::test_turn_limit_conflict_is_detected`
  against the real corpus text, not a synthetic fixture.
- **Positive.** No false positives on unrelated numbers in nearby text —
  `test_unrelated_numbers_are_not_flagged` pins this directly, and the
  shared-context-word requirement (`min_shared_context`) is what prevents
  "25 cells" and "7 business days" from being flagged just because both are
  numbers.
- **Negative, scoped deliberately.** Non-numeric contradictions are invisible
  to this method entirely. A rulebook revision that changes a rule's *logic*
  without changing a number in it ("Towers may now also capture Runners
  diagonally") would not be caught. This is a real gap, not an oversight,
  and is the reason conflict detection is presented as a supplementary
  signal attached to every answer (`AnswerResult.conflicts`) rather than as
  a claim to have solved cross-source consistency checking in general.

## Alternatives considered
- **LLM-as-judge contradiction detection** (ask the model "do these two
  passages disagree?"). More general, but adds a model call per query,
  inherits whatever failure modes the judge model has, and — critically for
  a hackathon-scale corpus like this one — is harder to test deterministically
  than a rule-based check with unit tests that assert exact behaviour against
  real text.
- **Doing nothing and relying on retrieval ranking to surface the newer
  document first.** Rejected: ranking is not a substitute for detection. A
  caller reading only the answer text, not the full retrieved-chunk list,
  would never see the disagreement at all.
