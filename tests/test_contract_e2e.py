"""End-to-end tests for contract mode against the synthetic sample corpus."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from contract_corpus import (
    IMAGE_TRANSCRIPTS,
    QUESTIONS,
    ScriptedModel,
    build_corpus,
    good_replies,
    image_key,
)
from tabula_rag.contract import cli
from tabula_rag.contract.answer import QueryResult, answer_query
from tabula_rag.contract.config import ContractSettings
from tabula_rag.contract.index import ContractIndex
from tabula_rag.contract.llm import OpenAIModelClient, extract_json_object
from tabula_rag.contract.runner import (
    index_corpus,
    load_or_build,
    output_path,
    run_query,
)
from tabula_rag.contract.serve import gpu_memory_fraction
from tabula_rag.contract.text import grader_normalize


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return build_corpus(tmp_path / "corpus")


@pytest.fixture
def settings(tmp_path: Path) -> ContractSettings:
    return ContractSettings(
        index_dir=tmp_path / "index", output_dir=tmp_path / "out", index_llm_wait_s=0.0
    )


def competent(corpus: Path, **overrides: dict[str, Any]) -> ScriptedModel:
    return ScriptedModel(
        replies={**good_replies(), **overrides},
        transcripts={
            image_key(corpus / path): text for path, text in IMAGE_TRANSCRIPTS.items()
        },
    )


async def ask(index: ContractIndex, model: ScriptedModel, question: str) -> QueryResult:
    return await answer_query(
        index, model, question, top_k=10, llm_timeout_s=5, deadline=time.monotonic() + 20
    )


def graded(result: QueryResult, answer: str, citations: list[str]) -> bool:
    """The published scoring rule: normalised exact answer AND exact citation set."""
    return grader_normalize(result.answer) == grader_normalize(answer) and set(
        result.citations
    ) == set(citations)


# --------------------------------------------------------------------------- the sample questions
class TestSampleQuestions:
    async def test_a_competent_model_scores_full_marks(
        self, corpus: Path, settings: ContractSettings
    ) -> None:
        model = competent(corpus)
        index = await index_corpus(corpus, model, settings)
        failures = [
            (q, r.answer, r.citations)
            for q, a, c in QUESTIONS
            if not graded(r := await ask(index, model, q), a, c)
        ]
        assert failures == []

    async def test_both_image_only_answers_need_the_vision_pass(
        self, corpus: Path, settings: ContractSettings
    ) -> None:
        blind = competent(corpus)
        blind.ready = False  # the model server never comes up
        index = await index_corpus(corpus, blind, settings)
        assert not index.docs["specs/backplane_pinout.png"].indexed
        assert "unavailable" in (index.docs["support/asset_label.jpg"].reason or "")
        for question, _, _ in [QUESTIONS[6], QUESTIONS[7]]:
            assert (
                await ask(index, blind, question)
            ).answer == ""  # graceful refusal, no crash

    async def test_text_files_are_still_indexed_when_the_model_is_down(
        self, corpus: Path, settings: ContractSettings
    ) -> None:
        down = competent(corpus)
        down.ready = False
        index = await index_corpus(corpus, down, settings)
        assert index.docs["support/bug_database.csv"].indexed
        assert down.transcribe_calls == 0
        assert ContractIndex.load(settings.index_dir).docs["support/bug_database.csv"].indexed

    async def test_images_are_read_once_and_land_in_the_persisted_index(
        self, corpus: Path, settings: ContractSettings
    ) -> None:
        model = competent(corpus)
        await index_corpus(corpus, model, settings)
        assert model.transcribe_calls == 2
        saved = ContractIndex.load(settings.index_dir)
        assert "THERM_ALERT#" in saved.docs["specs/backplane_pinout.png"].text

    async def test_the_prompt_labels_the_withdrawn_revision_and_omits_the_encrypted_file(
        self, corpus: Path, settings: ContractSettings
    ) -> None:
        model = competent(corpus)
        index = await index_corpus(corpus, model, settings)
        await ask(index, model, QUESTIONS[0][0])
        prompt = model.prompts[-1]
        assert "[FILE specs/tq40_datasheet_r2.pdf | current]" in prompt
        assert "superseded by specs/tq40_datasheet_r2.pdf" in prompt
        assert "supplier_agreement_ENCRYPTED" not in prompt and "412" not in prompt

    async def test_the_second_retrieval_round_pulls_in_the_file_the_question_never_names(
        self, corpus: Path, settings: ContractSettings
    ) -> None:
        model = competent(corpus)
        index = await index_corpus(corpus, model, settings)
        await ask(index, model, QUESTIONS[8][0])  # log incident -> ticket -> bug database
        assert "[FILE support/bug_database.csv" in model.prompts[-1]


# --------------------------------------------------------------------------- adversarial models
class TestAdversarialModels:
    async def test_a_hallucinated_price_is_refused(
        self, corpus: Path, settings: ContractSettings
    ) -> None:
        liar = competent(
            corpus,
            **{
                "unit price": {
                    "reasoning": "recalled",
                    "answerable": True,
                    "answer": "USD 412.50",
                    "sources": ["vendor/internal_audit.txt"],
                }
            },
        )
        index = await index_corpus(corpus, liar, settings)
        result = await ask(index, liar, QUESTIONS[9][0])
        assert (result.answer, result.citations) == ("", [])

    async def test_an_over_citing_model_is_cut_back_to_the_exact_set(
        self, corpus: Path, settings: ContractSettings
    ) -> None:
        greedy = competent(
            corpus,
            **{
                "maximum junction temperature": {
                    "reasoning": "cite everything",
                    "answerable": True,
                    "answer": "94 C",
                    "sources": [
                        "specs/tq40_datasheet_r2.pdf",
                        "specs/tq40_datasheet_r1_WITHDRAWN.pdf",
                        "planning/roadmap_fy27.docx",
                        "vendor/internal_audit.txt",
                    ],
                }
            },
        )
        index = await index_corpus(corpus, greedy, settings)
        result = await ask(index, greedy, QUESTIONS[0][0])
        assert graded(result, "94", ["specs/tq40_datasheet_r2.pdf"])

    async def test_a_model_that_cites_the_wrong_file_is_corrected(
        self, corpus: Path, settings: ContractSettings
    ) -> None:
        sloppy = competent(
            corpus,
            **{
                "error code is logged": {
                    "reasoning": "x",
                    "answerable": True,
                    "answer": "E7731",
                    "sources": ["support/bug_database.csv"],
                }
            },
        )
        index = await index_corpus(corpus, sloppy, settings)
        result = await ask(index, sloppy, QUESTIONS[4][0])
        assert (
            result.answer == "E7731"
            and result.citations[0] == "logs/prod_inference_2026-09-02.log"
        )

    async def test_a_model_that_says_unanswerable_yields_an_empty_answer(
        self, corpus: Path, settings: ContractSettings
    ) -> None:
        shy = competent(
            corpus,
            **{
                "board revision": {
                    "reasoning": "?",
                    "answerable": False,
                    "answer": "REV-C2",
                    "sources": ["support/asset_label.jpg"],
                }
            },
        )
        index = await index_corpus(corpus, shy, settings)
        assert (await ask(index, shy, QUESTIONS[7][0])).answer == ""

    async def test_a_model_that_returns_nothing_yields_an_empty_answer(
        self, corpus: Path, settings: ContractSettings
    ) -> None:
        index = await index_corpus(corpus, competent(corpus), settings)

        class Silent(ScriptedModel):
            async def chat_json(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
                return None

        assert (await ask(index, Silent(), QUESTIONS[0][0])).answer == ""

    async def test_an_answer_never_ships_without_a_citation(self) -> None:
        assert QueryResult(answer="94", citations=[], confidence=0.9).to_output() == {
            "answer": "",
            "citations": [],
            "confidence": 0.0,
        }

    async def test_a_question_with_no_lexical_overlap_is_refused_without_calling_the_model(
        self, corpus: Path, settings: ContractSettings
    ) -> None:
        model = competent(corpus)
        index = await index_corpus(corpus, model, settings)
        result = await ask(index, model, "zzzz qqqq xxxx")
        assert result.answer == "" and model.prompts == []

    async def test_the_model_is_not_called_when_the_deadline_has_already_passed(
        self, corpus: Path, settings: ContractSettings
    ) -> None:
        model = competent(corpus)
        index = await index_corpus(corpus, model, settings)
        result = await answer_query(
            index,
            model,
            QUESTIONS[0][0],
            top_k=10,
            llm_timeout_s=5,
            deadline=time.monotonic() - 1,
        )
        assert result.answer == "" and model.prompts == []


# --------------------------------------------------------------------------- hostile corpus
class TestHostileCorpus:
    async def test_indexing_survives_all_four_traps_and_keeps_going(
        self, corpus: Path, settings: ContractSettings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (corpus / "zzz_last.txt").write_text(
            "the last file, indexed after every trap", encoding="utf-8"
        )
        locked = corpus / "support" / "locked.txt"
        locked.write_text("classified", encoding="utf-8")
        real_open = Path.open

        def deny(self: Path, *args: Any, **kwargs: Any) -> Any:
            if self == locked:
                raise PermissionError(13, "denied")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", deny)
        index = await index_corpus(corpus, competent(corpus), settings)
        assert "permission" in (index.docs["support/locked.txt"].reason or "")
        assert index.docs["vendor/supplier_agreement_ENCRYPTED.pdf"].reason == "encrypted PDF"
        assert index.docs["vendor/telemetry_capture.dat"].reason == "unknown file type"
        assert index.docs[
            "zzz_last.txt"
        ].indexed  # a trap earlier in the walk did not stop later files
        assert "archive" not in " ".join(index.docs)  # the empty directory contributes nothing

    @pytest.mark.skipif(
        os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
        reason="POSIX non-root",
    )
    async def test_a_real_unreadable_file_does_not_stop_indexing(
        self, tmp_path: Path, settings: ContractSettings
    ) -> None:
        root = build_corpus(tmp_path / "hostile", include_unreadable=True)
        try:
            index = await index_corpus(root, competent(root), settings)
            assert not index.docs["support/restricted_notes.txt"].indexed
            assert all("399" not in c.text for c in index.chunks)
        finally:
            (root / "support" / "restricted_notes.txt").chmod(0o600)

    async def test_an_empty_corpus_and_a_missing_corpus_both_work(
        self, tmp_path: Path, settings: ContractSettings
    ) -> None:
        (tmp_path / "empty").mkdir()
        for root in (tmp_path / "empty", tmp_path / "missing"):
            index = await index_corpus(root, ScriptedModel(), settings, save=False)
            assert len(index.chunks) == 0
            assert (await ask(index, ScriptedModel(), "anything at all")).answer == ""


# --------------------------------------------------------------------------- persistence & runner
class TestRunner:
    def test_output_path_uses_the_query_id_and_cannot_escape(
        self, settings: ContractSettings
    ) -> None:
        assert output_path(settings, "query_01").name == "query_01_output.json"
        assert output_path(settings, "../../etc/x").parent == settings.output_dir

    async def test_a_saved_matching_index_is_reused_as_is(
        self, corpus: Path, settings: ContractSettings
    ) -> None:
        await index_corpus(corpus, competent(corpus), settings)
        loaded = load_or_build(corpus, settings)
        assert loaded.docs[
            "specs/backplane_pinout.png"
        ].indexed  # image text came from disk, no model needed

    async def test_a_stale_index_is_rebuilt_but_keeps_the_image_transcripts(
        self, corpus: Path, settings: ContractSettings
    ) -> None:
        await index_corpus(corpus, competent(corpus), settings)
        (corpus / "engineering" / "new_note.txt").write_text(
            "added after indexing", encoding="utf-8"
        )
        rebuilt = load_or_build(corpus, settings)
        assert rebuilt.docs["engineering/new_note.txt"].indexed
        assert "THERM_ALERT#" in rebuilt.docs["specs/backplane_pinout.png"].text

    def test_a_missing_index_falls_back_to_a_text_only_build(
        self, corpus: Path, settings: ContractSettings
    ) -> None:
        rebuilt = load_or_build(corpus, settings)
        assert rebuilt.docs["support/bug_database.csv"].indexed
        assert not rebuilt.docs["support/asset_label.jpg"].indexed

    def test_run_query_writes_a_valid_answer_file(
        self, corpus: Path, settings: ContractSettings
    ) -> None:
        asyncio.run(index_corpus(corpus, competent(corpus), settings))
        run_query(
            corpus, "query_05", QUESTIONS[4][0], settings, competent(corpus), watchdog=False
        )
        written = json.loads(output_path(settings, "query_05").read_text(encoding="utf-8"))
        assert set(written) == {"answer", "citations", "confidence"}
        assert written["answer"] == "E7731"
        assert written["citations"] == ["logs/prod_inference_2026-09-02.log"]
        assert 0.0 < written["confidence"] <= 1.0

    def test_run_query_still_writes_a_refusal_when_everything_explodes(
        self, corpus: Path, settings: ContractSettings
    ) -> None:
        class Exploding(ScriptedModel):
            async def chat_json(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
                raise RuntimeError("boom")

        run_query(corpus, "query_x", QUESTIONS[0][0], settings, Exploding(), watchdog=False)
        assert json.loads(output_path(settings, "query_x").read_text(encoding="utf-8")) == {
            "answer": "",
            "citations": [],
            "confidence": 0.0,
        }

    def test_run_query_with_an_unusable_corpus_still_writes_a_refusal(
        self, tmp_path: Path, settings: ContractSettings
    ) -> None:
        run_query(
            tmp_path / "nowhere",
            "query_y",
            "anything",
            settings,
            ScriptedModel(),
            watchdog=False,
        )
        assert (
            json.loads(output_path(settings, "query_y").read_text(encoding="utf-8"))["answer"]
            == ""
        )


# --------------------------------------------------------------------------- the CLI contract
class TestCli:
    def test_index_then_query_produce_the_files_the_harness_reads(
        self,
        corpus: Path,
        settings: ContractSettings,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv(
            "TABULA_RAG_LLM_BASE_URL", "http://127.0.0.1:9/v1"
        )  # nothing listens here
        monkeypatch.setenv("TABULA_RAG_INDEX_DIR", str(settings.index_dir))
        monkeypatch.setenv("TABULA_RAG_INDEX_LLM_WAIT_S", "0")
        assert cli.main(["--index", str(corpus)]) == 0
        assert (
            ContractIndex.load(settings.index_dir)
            .docs["logs/prod_inference_2026-09-02.log"]
            .indexed
        )

        code = cli.main(
            [
                "--corpus",
                str(corpus),
                "--query-id",
                "query_01",
                "--query",
                QUESTIONS[0][0],
                "--output-dir",
                str(settings.output_dir),
                "--surprise-flag",
                "1",
            ]
        )
        assert code == 0
        written = json.loads(
            (settings.output_dir / "query_01_output.json").read_text(encoding="utf-8")
        )
        assert set(written) == {"answer", "citations", "confidence"}
        assert written["answer"] == ""  # no model is reachable: a valid refusal, not a crash
        assert "ignoring unrecognized" in capsys.readouterr().err

    def test_running_with_no_arguments_does_not_crash(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main([]) == 0
        assert "usage" in capsys.readouterr().err


# --------------------------------------------------------------------------- the HTTP client
def _client(handler: Any, settings: ContractSettings | None = None) -> OpenAIModelClient:
    transport = httpx.MockTransport(handler)
    cfg = settings or ContractSettings()
    return OpenAIModelClient(
        cfg, httpx.AsyncClient(transport=transport, base_url="http://llm/v1")
    )


class TestHttpClient:
    async def test_chat_json_parses_the_reply_and_requests_structured_output(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "m1"}]})
            seen.update(json.loads(request.content))
            reply = {"answerable": True, "answer": "94", "sources": [], "reasoning": "r"}
            return httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps(reply)}}]}
            )

        client = _client(handler)
        out = await client.chat_json(
            "sys", "user", {"type": "object"}, max_tokens=50, timeout_s=5
        )
        assert out is not None and out["answer"] == "94"
        assert seen["model"] == "m1" and seen["response_format"]["type"] == "json_schema"
        assert seen["temperature"] == 0.0

    async def test_it_falls_back_to_plain_json_when_the_server_rejects_structured_output(
        self,
    ) -> None:
        calls: list[bool] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "m1"}]})
            body = json.loads(request.content)
            calls.append("response_format" in body)
            if "response_format" in body:
                return httpx.Response(400, text="unsupported")
            content = 'Sure! ```json\n{"answerable": true, "answer": "x", "sources": [], "reasoning": ""}\n```'
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        out = await _client(handler).chat_json("s", "u", {}, max_tokens=10, timeout_s=5)
        assert out is not None and out["answer"] == "x"
        assert calls == [True, False]

    async def test_server_errors_and_unreachable_servers_yield_none(self) -> None:
        def failing(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "m1"}]})
            return httpx.Response(500, text="oops")

        def unreachable(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        assert (
            await _client(failing).chat_json("s", "u", {}, max_tokens=10, timeout_s=1) is None
        )
        assert (
            await _client(unreachable).chat_json("s", "u", {}, max_tokens=10, timeout_s=1)
            is None
        )
        assert await _client(unreachable).transcribe(b"x", timeout_s=1) is None

    async def test_transcribe_sends_the_image_as_a_data_uri(self, corpus: Path) -> None:
        sent: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "vlm"}]})
            sent.update(json.loads(request.content))
            return httpx.Response(
                200, json={"choices": [{"message": {"content": " Pin B14: THERM_ALERT# "}}]}
            )

        text = await _client(handler).transcribe(
            (corpus / "specs/backplane_pinout.png").read_bytes(), timeout_s=5
        )
        assert text == "Pin B14: THERM_ALERT#"
        parts = sent["messages"][0]["content"]
        assert parts[0]["type"] == "text" and parts[1]["type"] == "image_url"
        uri = parts[1]["image_url"]["url"]
        assert uri.startswith("data:image/png;base64,")
        assert base64.b64decode(uri.split(",", 1)[1]).startswith(b"\x89PNG")

    async def test_a_corrupt_image_is_skipped_without_calling_the_model(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "vlm"}]})
            raise AssertionError("the model must not be called for an undecodable image")

        assert await _client(handler).transcribe(b"not an image", timeout_s=1) is None

    async def test_wait_ready_polls_until_the_server_comes_up(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] < 3:
                return httpx.Response(503)
            return httpx.Response(200, json={"data": [{"id": "m"}]})

        async def instant(_seconds: float) -> None:
            return None

        monkeypatch.setattr("tabula_rag.contract.llm.asyncio.sleep", instant)
        assert await _client(handler).wait_ready(30) is True
        assert attempts["n"] == 3

    async def test_wait_ready_gives_up_when_the_budget_is_spent(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

        assert await _client(handler).wait_ready(0.0) is False

    def test_json_extraction_tolerates_fences_reasoning_and_chatter(self) -> None:
        assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
        assert extract_json_object('<think>hmm {"a": 2}</think>{"a": 1}') == {"a": 1}
        assert extract_json_object('Here you go: {"a": {"b": 2}} done') == {"a": {"b": 2}}
        assert extract_json_object("no json here") is None
        assert extract_json_object("[1, 2]") is None


class TestServe:
    def test_the_vram_budget_becomes_a_fraction_of_the_whole_card(self) -> None:
        assert gpu_memory_fraction(36, 192) == pytest.approx(
            0.188, abs=1e-3
        )  # a big accelerator
        assert gpu_memory_fraction(36, 48) == pytest.approx(0.75, abs=1e-3)
        assert gpu_memory_fraction(36, 24) == 0.92  # capped: never claim the whole card
        assert gpu_memory_fraction(36, None) == 0.75
