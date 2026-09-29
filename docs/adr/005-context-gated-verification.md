# ADR 5 — Verification is gated on context availability, not on prompt wording

## Status
Accepted (supersedes an earlier version within the same development cycle —
kept as history for the same reason mini-challenge 2's ADR 3 keeps its own
bug: the failure it documents is exactly the class of mistake this project
is meant to catch, and catching it in the eval harness rather than in a
demo is the result the design is supposed to produce)

## Context
`PromptVersion` (`prompts/registry.py`) carries two flags: `requires_context`
(should retrieved chunks be shown to the model at all) and
`requires_citations` (does the system prompt instruct the model to cite
them). The first working version of `RAGService._verify_claims` gated claim
verification on `requires_citations`.

Running the ablation ladder this ADR exists because of
(`scripts/run_ablation.py`, `docs/rag_ablation.md`) surfaced the bug
immediately: `no-retrieval` (no context, no citation instruction) and
`naive` (context provided, but citations not instructed) produced identical
scores — 30% accuracy, 0% grounded — for reasons the write-up's own prose
could not honestly distinguish. Both rungs' claims arrived with an empty
`cited_chunk_ids` list, and the verification gate treated an empty citation
list identically regardless of whether a corpus had even been consulted.

The distinction that matters is not "did the prompt ask for a citation" —
it is "did anything exist to cite in the first place." Those are different
axes. `naive` had real, retrieved context available and chose not to cite
it: a genuine, checkable failure. `no-retrieval` had nothing in context at
all: demanding a citation there is not a stricter check, it is a category
error, equivalent to failing a student for not citing a textbook they were
never given.

## Decision
Verification now branches on `prompt.requires_context`:

- **Context was available** (`naive`, `cited`): an uncited claim is scored
  `ClaimStatus.UNCITED` — a real verification failure, counted against the
  claim's trustworthiness, and (past the `max_unsupported_ratio` threshold)
  grounds for withholding the whole answer.
- **No context was available** (`no-retrieval`): claims are scored
  `ClaimStatus.UNGROUNDED` instead — passed through as servable, since there
  was nothing to check them against, but permanently distinguishable from a
  verified claim. `AnswerResult.grounded_ratio` counts `SUPPORTED` only, so
  an `UNGROUNDED` answer never looks verified to a caller checking that
  field, even though it shipped.

`Claim.is_trustworthy` (governs both what ships in the answer text and the
abstention-ratio gate) now returns true for `SUPPORTED` and `UNGROUNDED`
alike; `grounded_ratio` stays strict and counts `SUPPORTED` only. The two
numbers together — `faithfulness` (nothing was refused) and `grounded_ratio`
(nothing was verified) — are what make "this answer shipped, but zero of it
was checked" visible at a glance instead of contradictory-looking.

## Consequences
- **Positive.** The ablation ladder now demonstrates what it claims to:
  `no-retrieval` and `naive` diverge on `abstain precision` (100% vs. 30% in
  `docs/rag_ablation.md`) for the correct, distinguishable reasons, instead
  of coincidentally landing on the same number for reasons the prose had to
  paper over.
- **Positive.** A real API caller who sends `use_retrieval: false` now gets
  a served answer (governed by the model's own declared `abstained` flag)
  with an explicit warning and an honest `grounded_ratio` of `0.0`, rather
  than every such call being silently forced to abstain by a citation check
  that never made sense for it.
- **Positive.** `tests/test_rag.py::test_no_retrieval_claims_are_ungrounded_not_supported`
  and its contrast case `test_naive_mode_uncited_claim_is_still_refused_unlike_no_retrieval`
  pin both halves of this distinction so a future refactor cannot silently
  re-merge them.
- **Negative.** `AnswerResult` now has two "how good is this" numbers
  (`faithfulness`, `grounded_ratio`) instead of one, which is more for a
  caller to understand. Judged worth it: collapsing them back into one
  number is exactly what caused the original bug, and the eval harness's
  markdown table (`eval/harness.py::render_markdown`) surfaces both side by
  side specifically so neither is missed.

## History
Kept deliberately rather than quietly rewritten: the first version of this
ADR would have described `requires_citations`-gated verification as the
design, because that is what shipped and passed every test that existed at
the time. The tests were not wrong; they simply never exercised the
no-retrieval rung against the naive rung in the same run and compared the
*reasons* behind matching scores. The ablation harness — built specifically
to demonstrate RAG's value per the mini-challenge brief — is what caught it,
which is the strongest argument available for actually running the
evaluation this project's own brief asks for, rather than treating it as a
deliverable to produce after the fact.
