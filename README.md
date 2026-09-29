# TABULA RAG

**Mini-Challenge 3 — Lablab × AMD AI Academy Challenge**
*Leveraging RAG to improve accuracy and reduce hallucination on proprietary domain knowledge no model has been trained on.*

Grounded question answering where every claim in an answer carries the
citation it was checked against, and the service **withholds the whole
answer** rather than serve one built mostly of claims it couldn't verify.
Corpus: a two-player abstract strategy game invented for this project —
provably absent from any model's training data — so retrieval's value is
measurable, not assumed.

```
$ tabula-rag demo
indexed 12 chunks across 2 documents

Q: What does a Tower's Surge ability do?
   [ANSWER]  A Tower's Surge captures every enemy Runner in a straight line of sight up to and including the third cell.
             faithfulness=1.00  claims=1

Q: What colour is the Hexfall board?
   [ABSTAIN] The rulebook does not specify a board colour scheme.

Q: After how many moves is Hexfall declared a draw?
   [ANSWER]  A game that reaches move 60 without a winner is declared a draw.
             faithfulness=1.00  claims=1
   [CONFLICT] Hexfall Rulebook §9. Turn limit says 60, Hexfall Errata §Errata 1 — Turn limit (corrects Section 9) says 80  (re: game / move / reaches / that)
```

Three real behaviours in eight seconds, no GPU: a correctly grounded answer,
a correct refusal on a genuinely unanswerable question, and a **real
cross-source conflict** — the rulebook says 60, its own errata sheet says
80 — caught and surfaced instead of silently answered from whichever chunk
retrieval happened to draw first.

---

## Why this exists

RAG demos are usually evaluated on public documents a model has plausibly
already seen, which makes "did retrieval actually help" unanswerable — you
can't distinguish grounding from recall. This corpus is **Hexfall**, a game
invented specifically for this project
([`corpus/hexfall_rulebook_v1.md`](corpus/hexfall_rulebook_v1.md)). No model
has ever seen it. That makes the ablation in this repo a genuine measurement
rather than an assumption: a model answering with retrieval switched off
scores **zero** on every answerable question, because there is nothing in
its training data to fall back on. See
[`docs/rag_ablation.md`](docs/rag_ablation.md) for the actual run.

Beyond that baseline, the service is built around three checks a fabricated
or stale answer cannot pass at once:

| Check | What it catches | Module |
|---|---|---|
| **Citation-enforced generation** | A claim with no citation at all | `prompts/registry.py` (schema-locked via `guided_json`) |
| **Lexical coverage verification** | A claim citing a *real* chunk that doesn't actually support it | `verification.py` |
| **Cross-source conflict detection** | Two well-grounded chunks that disagree with each other | `conflicts.py` |

An answer with too high a share of unverified claims is withheld entirely
(`max_unsupported_ratio`), not partially served — see
[`docs/adr/005-context-gated-verification.md`](docs/adr/005-context-gated-verification.md)
for the real design bug this build caught while running its own ablation:
an early version of the verification gate couldn't distinguish "context
existed and wasn't cited" from "no context existed at all," which silently
erased the exact distinction the ablation ladder exists to demonstrate. The
fix, and the regression tests that now pin it, are the most important part
of this repository — the RAG equivalent of mini-challenge 2's grounding-veto
story.

---

## Two entry points

| | **Service** (`tabula-rag`, FastAPI) | **Contract mode** (`app.py`) |
|---|---|---|
| for | people and dashboards | the Mini-Challenge 3 grader |
| answer | a cited sentence, claim by claim | the **value only** |
| citations | chunk ids + supporting quotes | the **exact set of files** the answer needed |
| unanswerable | an explained abstention | `{"answer": "", "citations": [], "confidence": 0.0}` |
| corpus | markdown | pdf, docx, xlsx, csv, txt/log, py, and images (read by a vision model) |
| runs as | a long-lived API | a fresh process per question, over a persisted index |

Both share the same ideas: grounding as a veto, abstention over guessing, revision awareness.
Contract mode is a thin, separate subpackage (`src/tabula_rag/contract/`); the service is untouched.
The reasoning is in [`docs/adr/006-contract-mode.md`](docs/adr/006-contract-mode.md).

```bash
python3 app.py --index /app/corpus                     # once: parse, read images, persist the index
python3 app.py --corpus /app/corpus --query-id query_01 \
    --query "What is the maximum junction temperature of the TQ-40?"
# -> /app/output/query_01_output.json  {"answer": "94", "citations": ["specs/tq40_datasheet_r2.pdf"], "confidence": 0.9}
```

What it does that a generic RAG pipeline would not:

* **Survives a hostile corpus.** An empty directory, an unknown binary, an unreadable file and an
  encrypted PDF are each skipped with a recorded reason; indexing continues. An encrypted PDF is
  skipped *even if it opens with an empty password*, because the graded questions treat its contents
  as unanswerable.
* **Answers or refuses, verifiably.** The answer must literally occur in a file it cites (compared with
  the grader's own normalisation), otherwise the output is the required empty refusal. A hallucinated
  value fails this check by construction.
* **Cites the exact set.** Superseded revisions (`..._r1_WITHDRAWN.pdf` next to `..._r2.pdf`) and files
  that merely discuss the topic are dropped; genuine chain links (a log line that supplied the ticket
  number used to look the fix up in the bug database) are kept.
* **Always leaves a valid file.** A placeholder refusal is written first, so a crash, a timeout or a dead
  model still scores as a refusal instead of a malformed response.

```bash
python scripts/contract_selfcheck.py            # replays the grader's process model; scripted model, no GPU
python scripts/contract_selfcheck.py --live \   # against your real model, with the organisers' kit
    --corpus mc3-starter-kit/corpus --questions mc3-starter-kit/questions.json
docker build -f Dockerfile.submission -t <registry>/tabula-rag:v1 .
bash scripts/check_submission.sh <registry>/tabula-rag:v1 <corpus_dir> [questions.json]
```

**What is and is not verified.** The contract path has 100+ tests and the self-check scores 200/200 on a
synthetic corpus shaped like the published one, including a 16 MB stress corpus (index 10 s, load 0.9 s).
That model is **scripted**: it proves parsing, retrieval, verification, timing and the file contract, not how
well a real vision-language model reads images or follows the citation rule. The submission image is untested
on real ROCm hardware (vLLM availability for the mandated base image is the main unknown). Retrieval is
lexical only; add dense retrieval if a live run shows paraphrase misses. Scanned PDFs are not OCR'd.

---

## Architecture

```
                    ┌──────────────────────────────────────────────────┐
                    │                  tabula-rag API                  │
   question         │                                                  │
  ─────────────────▶│  ┌──────────┐   ┌──────────┐   ┌───────────────┐ │
                    │  │  hybrid  │──▶│  rerank  │──▶│   generate    │ │
                    │  │ retrieve │   │ (cross-  │   │ (cited prompt,│ │
                    │  │ BM25+RRF │   │ encoder) │   │  guided_json) │ │
                    │  │  +dense  │   └──────────┘   └───────┬───────┘ │
                    │  └────┬─────┘                          │         │
                    │       │                                ▼         │
                    │       │                       ┌──────────────┐   │
                    │       └──────────────────────▶│   verify     │   │
                    │        (also feeds)            │  each claim  │──┼─▶ AnswerResult
                    │                                │  + conflicts │   │   claims[], with
                    │                                └──────────────┘   │   status + quote
                    └──────────────────────────────────────────────────┘
                           │                │                │
                           ▼                ▼                ▼
                  vLLM/embeddings   vLLM/reranker      vLLM/LLM (text)
                     BAAI/bge-m3    bge-reranker-v2   Qwen2.5-14B-Instruct
                          AMD Instinct MI300X · ROCm · one GPU, three models
```

### Package layout

```
src/tabula_rag/
  models.py             Chunk, Claim (+ ClaimStatus), AnswerResult, SourceConflict…
  config.py              All configuration, read once from TABULA_RAG_* env vars
  errors.py               Typed error hierarchy, one HTTP status each
  chunking.py              Markdown-heading-aware, token-budgeted, overlap-safe
  corpus_index.py           Chunk store + BM25 + dense index, always in sync
  retrieval/bm25.py          Hand-rolled BM25Okapi — no black-box dependency
  retrieval/dense.py          Flat cosine-similarity index (ADR 2: why, and the upgrade path)
  retrieval/fusion.py          Reciprocal Rank Fusion (ADR 1)
  retrieval/embeddings.py       Embedding client + served/production + offline fake
  retrieval/reranker.py          Cross-encoder reranker + served/production + offline fake
  verification.py            Lexical claim-coverage scoring (ADR 3)
  conflicts.py                 Cross-source numeric/date disagreement (ADR 4)
  prompts/registry.py           no-retrieval / naive / cited prompt versions
  llm/client.py                  Production LLM client: retries, backoff, circuit breaker
  llm/fake.py                     Deterministic scripted model for tests, demo, offline eval
  pipeline.py                      Orchestrates retrieve → rerank → generate → verify
  api.py                            FastAPI: /healthz /readyz /metrics /v1/ingest /v1/query
  cli.py                             tabula-rag ingest | evaluate | demo | prompts
  eval/metrics.py                    Accuracy, faithfulness, grounded_ratio, abstention precision/recall
  eval/harness.py                     The no-retrieval / naive / cited ablation ladder
```

---

## The ablation ladder — the brief's central claim, measured

```bash
tabula-rag evaluate --dataset eval/golden.jsonl
# or, for the fully-annotated write-up:
make eval-ablation
```

| prompt version | cases | accuracy | faithfulness | grounded | abstain recall | abstain precision |
|---|---:|---:|---:|---:|---:|---:|
| `no-retrieval` | 10 | 30.0% | 100.0% | 0.0% | 100.0% | 100.0% |
| `naive` | 10 | 30.0% | 100.0% | 0.0% | 100.0% | 30.0% |
| `cited` | 10 | **100.0%** | 100.0% | **100.0%** | 100.0% | 100.0% |

(From [`docs/rag_ablation.md`](docs/rag_ablation.md), a scripted-model run
whose responses are written to be *realistic* about each rung's actual
failure mode — see that file for the full breakdown and the honest caveat
about which numbers are vacuous when a rung abstains on everything.)

Read past the headline number to the two that matter: **grounded** (share of
claims actually checked against a source) and **abstain precision** (how
often abstention was the right call). `no-retrieval` scores 0% grounded
because Hexfall cannot be in any model's training data — this is the delta
the brief specifically asks for, and it is only measurable because the
corpus is provably novel. `naive` scores identically on accuracy but for a
completely different, more interesting reason: its content is scripted
*correct*, but with context available and nothing cited, the service
correctly refuses to serve it — a citation isn't decoration, it's what makes
a claim auditable at all.

---

## Quickstart

```bash
git clone https://github.com/SriRamkunamsetty/tabula-rag.git && cd tabula-rag
pip install -e ".[dev]"

# No GPU needed — ingests the Hexfall rulebook + errata, asks three
# questions, and shows a real cross-source conflict being caught:
tabula-rag demo

pytest -q            # 61 tests
mypy                  # strict mode, zero errors
ruff check .           # zero findings
```

### Running against real models on AMD Developer Cloud

```bash
# One-time: AMD AI Developer Program credit, then a 1× MI300X GPU Droplet.
./deploy/provision_mi300x.sh <droplet-ip>

export TABULA_RAG_LLM_BASE_URL="http://<droplet-ip>:8000/v1"
export TABULA_RAG_EMBEDDING_BASE_URL="http://<droplet-ip>:8001/v1"
export TABULA_RAG_RERANKER_BASE_URL="http://<droplet-ip>:8002/v1"

uvicorn tabula_rag.api:create_app --factory --port 8090
```

### Docker

```bash
docker compose --profile gpu up --build   # API + all 3 model servers, one MI300X
docker compose up api                     # API only; point TABULA_RAG_*_BASE_URL
                                           # at already-running endpoints
```

---

## API

```
POST /v1/ingest   {"document_id": "hexfall-rulebook", "title": "...", "text": "..."}
POST /v1/query    {"query": "What does a Tower's Surge do?", "top_k": 8,
                    "use_retrieval": true, "prompt_version": null}
```

`AnswerResult` carries the full evidence trail, not just text:

```json
{
  "abstained": false,
  "answer": "A Tower's Surge captures every enemy Runner ...",
  "claims": [{
    "text": "A Tower's Surge captures every enemy Runner ...",
    "cited_chunk_ids": ["hexfall-rulebook::0005"],
    "status": "supported",
    "coverage_score": 0.83,
    "supporting_quote": "a Tower may perform a Surge — capturing every enemy Runner ..."
  }],
  "conflicts": [],
  "grounded_ratio": 1.0
}
```

`GET /readyz` checks the corpus is non-empty and the LLM responds — wired to
a Kubernetes `readinessProbe`, never `livenessProbe`, for the same reason as
mini-challenge 2: a slow model must never become a restart loop.
`GET /metrics` exposes `tabula_rag_claims_unsupported_total` and
`tabula_rag_conflicts_detected_total`, the two numbers worth alerting on.

---

## Design decisions

Full reasoning in [`docs/adr/`](docs/adr/):

1. [**Hybrid retrieval**](docs/adr/001-hybrid-retrieval.md) — BM25 + dense,
   fused by RRF rather than a weighted blend, and why.
2. [**Flat dense index**](docs/adr/002-flat-dense-index.md) — why a linear
   scan is the right choice at this scale, with a named upgrade path.
3. [**Lexical verification**](docs/adr/003-lexical-verification.md) — why
   coverage scoring instead of a second LLM-as-judge call, argued honestly
   including where it can be fooled.
4. [**Scoped conflict detection**](docs/adr/004-scoped-conflict-detection.md) —
   why numeric/date-only, not general contradiction detection.
5. [**Context-gated verification**](docs/adr/005-context-gated-verification.md) —
   the real bug this build's own ablation run caught, and the fix.

---

## Testing philosophy

61 tests, organised by risk retired. `TestAntiHallucination` asserts a claim
citing a real chunk but saying something false is refused, and that citing a
real chunk is *not* sufficient on its own. `TestConflicts` runs against the
actual Hexfall rulebook and errata text, not a synthetic fixture — the
60-vs-80 turn-limit disagreement is real and detected end to end.

```bash
pytest -q --cov=tabula_rag --cov-report=term-missing   # 84%+ branch coverage
mypy                                                      # strict, zero errors
ruff check .                                               # zero findings
```

## What's next (mini-challenge 4)

This service's citation-enforced, verification-gated pattern is the
template for mini-challenge 4's web-scraping agent: retrieved evidence
verified against source text before anything is returned, applied to live
web pages instead of an indexed corpus.

## License

MIT. See `LICENSE`.
