# ADR 3 — Claim verification is lexical coverage, not a second model call

## Status
Accepted

## Context
A citation-enforced prompt makes the generator attach a chunk id to every
claim. That alone is not enough — a model can cite a real chunk and still
say something that chunk does not support (`FABRICATED_CLAIM` in
`tests/test_rag.py` is exactly this: a real citation, false content). Some
check has to verify the claim against the text of what it cites.

Two approaches were available: a second LLM call asking "does this passage
entail this claim?" (an NLI-style judge), or a cheap lexical-overlap score
computed with no model call at all.

## Decision
Verification (`verification.py::score_claim_coverage`) is lexical: strip
stopwords from both the claim and its cited chunk, and score what fraction
of the claim's remaining content terms also appear in the chunk. This is the
same family of metric as the "Unsupported Sentence Ratio" used in the RAG
evaluation literature, not a novel technique — the choice here is *using* it
as a live serving-time gate rather than only an offline evaluation metric.

## Consequences
- **Positive.** Free. No extra model call, no extra latency, no extra
  dollar cost per claim — verification runs on every claim, every request,
  with none of the cost-vs-thoroughness trade-off a second LLM call would
  force.
- **Positive.** Deterministic and testable with plain Python, no model
  needed. `tests/test_rag.py::TestVerification` asserts exact behaviour
  against real rulebook text without any scripted model in the loop.
- **Negative, named honestly.** Lexical overlap is a proxy for entailment,
  not entailment itself. It can be fooled by a claim that reuses a chunk's
  vocabulary while inverting its meaning ("a Runner may capture a Tower"
  built entirely from words in the sentence that states the opposite) — a
  trained NLI model would likely catch this; the coverage score alone would
  not. `claim_coverage_threshold` is tuned to reduce false negatives
  (correct claims wrongly refused) at the cost of this specific blind spot,
  which is the safer failure direction for a system whose bias should be
  toward under- rather than over-trusting.

## Mitigations already in place
- **Format-sensitive terms survive stopword filtering.** Numbers, named
  entities, and rule-specific vocabulary ("Surge", "Sunk", "orthogonally")
  are exactly the terms this method is strongest on, and they are also the
  terms most likely to be wrong in a fabricated claim.
- **`conflicts.py` catches a specific, different failure mode** — two
  well-grounded chunks disagreeing with each other — that lexical coverage
  by construction cannot see, since it checks one claim against its own
  citation and never compares across chunks.

## Upgrade path
If a corpus proves adversarial to lexical coverage in practice (measured via
the eval harness's `abstention_precision` dropping on a labelled set with
inverted-meaning distractor claims), the fix is local: `score_claim_coverage`
is the only function `pipeline.py` calls for this check, so an NLI-model
verification pass — served the same way as the LLM and reranker, behind a
`Protocol` — can be substituted or layered on top without touching the
pipeline's control flow.
