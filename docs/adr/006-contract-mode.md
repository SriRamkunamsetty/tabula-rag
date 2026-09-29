# ADR 006: Contract mode, the shape the grader scores

## Status

Accepted. Added after the Mini-Challenge 3 specification was published; the service in the rest
of this repository was designed before it, from the one-line brief.

## Context

The published specification differs from the service's shape in ways that matter more than any
retrieval-quality difference:

| | Service (ADR 1-5) | What is graded |
|---|---|---|
| answer | a cited sentence | the **value only**, compared after normalisation |
| citations | chunk ids | the **exact set of files** the answer needed |
| unanswerable | prose that abstains | `answer: ""` and `citations: []` (prose is scored wrong) |
| corpus | markdown | pdf, docx, xlsx, csv, txt/log, py, and images whose text is the only answer |
| hostile files | not considered | empty dir, unknown type, unreadable, encrypted: indexing must continue |
| process model | long-running API | a **new process per question**; `--index` once, charged to start-up |
| runtime | three separate model servers | one container on the mandated ROCm base, GPU required |

Citations are scored as an exact set with a *necessity* rule ("cite a file only if removing it
would make the answer impossible"), and a correct answer with a wrong citation set scores zero.

## Decision

Keep the service; add `tabula_rag.contract`, a second, thinner entry point that reuses the ideas
and changes their shape. The pieces:

1. **Every file becomes text, or a recorded reason it did not.** Parsers never raise. An
   encrypted PDF is skipped *even if it opens with an empty password*: the specification says
   anything inside an encrypted file is unanswerable, and its own sample question 10 is one whose
   answer sits inside such a file. Reading it because we can would return exactly the value that
   must not be returned.
2. **Images are read once, at index time**, by the same vision-language model that answers
   questions. This is what makes the two image-only sample questions answerable, and it keeps
   image reading out of the 30 second per-question budget.
3. **The index is persisted** (chunks, texts, identifiers and the BM25 postings) so a fresh process
   loads it instead of rebuilding it. Measured on a 16 MB / 64k-chunk corpus, loading dropped from
   7.1 s (re-tokenising) to 0.9 s.
4. **Retrieval is lexical and identifier-aware**, plus one second round for chains. The corpus is
   full of part numbers, ticket ids and error codes, where BM25 is strong; `TQ-40` also indexes as
   `tq40`. Ids found in the best excerpts that the question did not contain are searched for
   directly, which is how "the log shows an incident, which release fixed it" reaches the bug
   database the question never names. File names are indexed too, so "the asset label" finds
   `asset_label.jpg`.
5. **Superseded revisions lose.** Family and revision are derived from the file name
   (`..._r1_WITHDRAWN.pdf` vs `..._r2.pdf`) and from the content; older or withdrawn documents are
   demoted in retrieval and labelled in the prompt.
6. **The model proposes, plain code decides.** After the model answers, deterministic checks
   apply: the answer must literally occur (after the grader's own normalisation) in a cited file
   and near terms from the question, otherwise the result is the required empty refusal; citations
   are cut to the necessary set: superseded files dropped, files that add nothing dropped, genuine
   chain links kept (a file that supplied an identifier the question did not contain, used to look
   the value up elsewhere).

## Alternatives considered

* **Dense retrieval and a reranker, as in the service.** Better on paraphrase, but it needs two
  more model servers inside one container's VRAM budget and start-up time, and the graded corpus
  is dominated by identifier lookups. Deferred, not rejected: it is the first thing to add if a
  real run shows paraphrase misses.
* **An agent loop (model chooses follow-up searches).** More general for multi-hop questions, but
  each round costs seconds of a 30 second budget and adds failure modes. The deterministic
  identifier expansion covers the chain shape the specification shows.
* **Letting the model's citations stand.** Models over-cite (every file that mentions the topic),
  which the exact-set scoring punishes. Pruning is deterministic so it can be tested.
* **Stripping units with a model.** A regex on standalone number-plus-unit answers is enough
  (`94 °C` becomes `94`) and cannot touch versions, quarters or identifiers.

## Consequences

* The contract path is exercised end to end by 100+ tests and by
  `scripts/contract_selfcheck.py`, which replays the grader's invocation model with real
  subprocesses and real HTTP, but against a **scripted** model. That validates parsing, retrieval,
  verification, timing and the file contract. It does **not** measure how well a real model reads
  images or follows the citation rule; that needs `--live` on real hardware.
* Scanned PDFs (pages that are images) yield no text and are not OCR'd; the specification lists
  PDFs as datasheets, so this is a known, cheap-to-close gap rather than a design choice.
* Grounding is lexical, so an answer that is *computed* from the corpus (a sum, a difference) is
  refused. The specification's answers are printed values, so this is the intended trade.
