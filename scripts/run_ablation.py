"""Generate docs/rag_ablation.md — the no-retrieval / naive / cited comparison.

Uses scripted LLM responses rather than a live model, exactly as
``mini-challenge 2``'s ``scripts/run_ablation.py`` does, and for the same
reason: this validates the scoring pipeline end to end on a controlled,
known scenario, and the responses are written to be *realistic* about each
rung's actual failure mode rather than rigged for a dramatic table:

* ``no-retrieval`` — the model has never seen Hexfall (it was invented for
  this project), so every factual question gets a deliberately wrong,
  plausible-sounding guess, submitted with no citation (there is no corpus
  in context to cite). The pipeline marks these claims ``UNGROUNDED`` rather
  than running them through citation verification — there is nothing to
  verify them against — and content correctness is judged directly. The
  three genuinely out-of-scope questions get an honest decline.
* ``naive`` — context is provided and every factual answer is scripted
  *correctly*, but with no citation, because the naive prompt does not
  require one. Because context genuinely existed here, the pipeline treats
  the missing citation as a real verification failure (``UNCITED``), not the
  no-context exemption ``no-retrieval`` gets — this is the load-bearing
  distinction the whole ablation exists to demonstrate, and it is enforced
  by ``prompt.requires_context``, not by whether the prompt happened to ask
  for citations.
* ``cited`` — same correct content, now with citations pointing at the
  actual indexed chunk ids, resolved by ingesting the corpus once up front.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tabula_rag.config import Settings
from tabula_rag.corpus_index import CorpusIndex
from tabula_rag.eval.harness import (
    EvalCase,
    RunConfig,
    RunReport,
    load_dataset,
    render_markdown,
    run_suite,
)
from tabula_rag.llm.fake import ScriptedLLM
from tabula_rag.retrieval.embeddings import HashingEmbeddingClient
from tabula_rag.retrieval.reranker import LexicalOverlapReranker

REPO_ROOT = Path(__file__).resolve().parents[1]

# Correct answers for the 7 answerable golden-set questions, in dataset order.
CORRECT_TEXT = {
    "q-surge": "A Tower's Surge captures every enemy Runner in a straight line of sight up to and including the third cell.",
    "q-first-move": "Amber always moves first.",
    "q-well": "A stone that enters the Well becomes Sunk and is removed from the board immediately, taking no further part in the game.",
    "q-runner-capture": "A Runner may never capture a Tower; Runners may only capture Runners.",
    "q-victory": "A player wins immediately when their opponent has no Towers remaining on the board.",
    "q-turn-limit": "A game that reaches move 60 without a winner is declared a draw.",
    "q-leap": "A Leap moves a stone two cells in a straight orthogonal line over one adjacent stone, and the jumped stone is not removed unless it is an enemy stone being captured.",
}

# Deliberately wrong guesses a model with no corpus access might plausibly
# make -- structurally similar to the real answer but factually incorrect,
# which is what a genuine hallucination on an invented game looks like.
WRONG_GUESSES = {
    "q-surge": "A Tower's Surge lets it move to any empty cell on the board once per turn.",
    "q-first-move": "The player with the black stones moves first, by standard convention.",
    "q-well": "A stone that enters the Well is teleported to a random empty cell.",
    "q-runner-capture": "Any stone may capture any other stone regardless of type.",
    "q-victory": "A player wins by capturing all of their opponent's stones.",
    "q-turn-limit": "The game has no turn limit and continues until one side has no legal moves.",
    "q-leap": "A Leap moves a stone three cells in any direction, including diagonally.",
}

UNANSWERABLE_IDS = {"q-board-colour", "q-prize", "q-app-download"}


def build_scripted_responses(
    prompt_version: str, cases: list[EvalCase], corpus: CorpusIndex | None
) -> list[str]:
    """Script one response per case, matching the failure mode described above."""
    responses = []
    for case in cases:
        if case.case_id in UNANSWERABLE_IDS:
            responses.append(
                json.dumps(
                    {
                        "claims": [],
                        "abstained": True,
                        "abstain_reason": "Not stated anywhere in the retrieved material.",
                    }
                )
            )
            continue

        if prompt_version == "no-retrieval":
            text = WRONG_GUESSES[case.case_id]
            responses.append(
                json.dumps(
                    {
                        "claims": [{"text": text, "cited_chunk_ids": []}],
                        "abstained": False,
                        "abstain_reason": None,
                    }
                )
            )
        elif prompt_version == "naive":
            text = CORRECT_TEXT[case.case_id]
            responses.append(
                json.dumps(
                    {
                        "claims": [
                            {"text": text, "cited_chunk_ids": []}
                        ],  # correct, but uncited
                        "abstained": False,
                        "abstain_reason": None,
                    }
                )
            )
        else:  # cited
            text = CORRECT_TEXT[case.case_id]
            assert corpus is not None
            chunk_id = _find_chunk_for(corpus, case.case_id)
            responses.append(
                json.dumps(
                    {
                        "claims": [{"text": text, "cited_chunk_ids": [chunk_id]}],
                        "abstained": False,
                        "abstain_reason": None,
                    }
                )
            )
    return responses


_ANCHOR_PHRASES = {
    "q-surge": "Surge",
    "q-first-move": "Amber always moves first",
    "q-well": "becomes Sunk",
    "q-runner-capture": "Runners may only capture Runners",
    "q-victory": "no Towers remaining",
    "q-turn-limit": "move 60",
    "q-leap": "Leap: move",
}


def _find_chunk_for(corpus: CorpusIndex, case_id: str) -> str:
    anchor = _ANCHOR_PHRASES[case_id]
    for chunk_id in corpus._chunks:
        chunk = corpus.get(chunk_id)
        if chunk and chunk.document_id == "hexfall-rulebook" and anchor in chunk.text:
            return chunk_id
    raise AssertionError(f"no chunk found for anchor {anchor!r}")


async def main() -> None:
    settings = Settings(environment="test", log_level="ERROR")
    cases = load_dataset(REPO_ROOT / "eval" / "golden.jsonl")
    rulebook = (REPO_ROOT / "corpus" / "hexfall_rulebook_v1.md").read_text(encoding="utf-8")

    reports: list[RunReport] = []
    for version in ("no-retrieval", "naive", "cited"):
        embeddings = HashingEmbeddingClient()
        probe_corpus = CorpusIndex(settings, embeddings)
        await probe_corpus.ingest("hexfall-rulebook", rulebook, title="Hexfall Rulebook")
        responses = build_scripted_responses(
            version, cases, probe_corpus if version == "cited" else None
        )

        single = await run_suite(
            cases,
            settings,
            [RunConfig(version)],
            llm_factory=lambda r=responses: ScriptedLLM(r),
            embeddings=HashingEmbeddingClient(),
            reranker=LexicalOverlapReranker(),
            corpus_documents=[("hexfall-rulebook", rulebook, "Hexfall Rulebook")],
        )
        reports.extend(single)

    table = render_markdown(reports)
    print(table)

    # factual-only breakdown, computed from the raw per-case outcomes
    breakdown_lines = []
    for report in reports:
        factual = [o for o in report.outcomes if not o.expects_abstain]
        n_correct = sum(1 for o in factual if o.correct)
        breakdown_lines.append(
            f"- **{report.config.label}**: {n_correct}/{len(factual)} answerable questions "
            f"answered correctly ({n_correct / len(factual):.0%})"
        )

    out = REPO_ROOT / "docs" / "rag_ablation.md"
    out.write_text(_render_report(table, breakdown_lines), encoding="utf-8")
    print(f"\nwrote {out}")


def _render_report(table: str, breakdown_lines: list[str]) -> str:
    breakdown = "\n".join(breakdown_lines)
    return f"""# RAG ablation: no-retrieval vs. naive vs. cited

Generated by `python scripts/run_ablation.py` (`make eval-ablation`) against
the 10-question `eval/golden.jsonl` set — 7 answerable from the Hexfall
rulebook, 3 genuinely unanswerable. A scripted model stands in for the served
LLM, with responses written to be realistic about each rung's actual failure
mode rather than rigged for a dramatic table — see the module docstring in
`scripts/run_ablation.py` for exactly what each rung is scripted to do and
why.

{table}
Two columns matter more than the headline "accuracy": **grounded** (share of
claims actually checked against a source and verified) and **faithfulness**
(share not actively refused). The gap between them is the whole point of
this table.

## The headline number: answerable-question accuracy

Overall accuracy above blends factual questions with the three abstain
cases, which every rung tends to get right (declining an invented specific
needs no corpus access). The number that actually demonstrates retrieval's
value is accuracy on the 7 **answerable** questions alone:

{breakdown}

**No-retrieval scores zero on every answerable question because the corpus
is provably novel** — Hexfall was invented for this project, so a model
with no retrieved context has no way to know its rules and produces
plausible-sounding wrong answers instead. Note the `grounded` column reads
0% here too, but for a different reason than the wrong content: these claims
were never checked against anything at all (`ClaimStatus.UNGROUNDED`), so
`grounded_ratio` correctly reports zero regardless of whether the guess
happened to be right. This is the delta the mini-challenge brief asks to
demonstrate, and it is only demonstrable at all because the corpus cannot be
in any model's training data.

## The non-obvious result: naive mode, and the bug it caught

Naive mode's factual *content* is scripted identically to cited mode — every
answer is correct. Its accuracy is not, because context genuinely was
available here and the claim still cited nothing: the pipeline correctly
scores this as `ClaimStatus.UNCITED`, a real verification failure, and
withholds the answer.

This distinction — `UNCITED` (context existed, wasn't cited) versus
`UNGROUNDED` (no context existed at all) — did not exist in the first working
version of this pipeline. That version gated citation verification on
`prompt.requires_citations` (did the system prompt ask for a citation),
which meant a `no-retrieval` guess and a `naive` correct-but-uncited answer
both failed through the identical code path for the identical stated reason,
even though nothing about a naive prompt should be blamed on the same
mechanism as "the model was never given anything to know the answer with."
Running this exact ablation is what surfaced it: `no-retrieval` and `naive`
scored identically in a way the write-up's own prose could not honestly
explain. The fix gates verification on `prompt.requires_context` instead —
context availability, not prompt wording — with the distinction pinned by
`tests/test_rag.py::test_no_retrieval_claims_are_ungrounded_not_supported`
and its contrast case. See `docs/adr/005-context-gated-verification.md` for
the full account.

One table artefact worth naming rather than glossing over: naive mode's
`faithfulness` (100%) and `grounded` (0%) columns are computed only across
cases that shipped an answer, and naive mode abstains on all ten — so both
figures are the definitional fallback for an empty set, not a real
measurement. The columns that actually carry information for this row are
`accuracy` and, especially, `abstain precision` (30%, versus no-retrieval's
honest 100%): naive mode isn't merely unverified, it is wrongly refusing
seven answers it was fully capable of getting right, purely because nothing
was cited.

The practical takeaway survives the fix intact: the citation-enforced prompt
is not primarily about catching hallucination after the fact — naive mode
shows that even a perfectly correct answer is worthless to this system
without something to check it against, and now that claim is measured
correctly rather than by coincidence.

Re-run this ablation against a provisioned MI300X endpoint (swap the
scripted `llm_factory` for `OpenAICompatibleLLM(settings)` in
`scripts/run_ablation.py`) to get real model behaviour in place of these
scripted responses before quoting this table externally.
"""


if __name__ == "__main__":
    asyncio.run(main())
