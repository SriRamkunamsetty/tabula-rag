"""Unit tests for contract mode: text helpers, parsers, revisions, index, verification."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from contract_corpus import build_corpus, make_pdf
from tabula_rag.contract.answer import clean_answer, resolve_paths, verify_and_cite
from tabula_rag.contract.corpus import (
    DocRecord,
    corpus_fingerprint,
    family_and_revision,
    mark_superseded,
    walk_corpus,
)
from tabula_rag.contract.index import ContractIndex, Hit, chunk_segments
from tabula_rag.contract.parsers import Segment, detect_kind, parse_file
from tabula_rag.contract.runner import build_text_index
from tabula_rag.contract.text import (
    content_terms,
    extract_identifiers,
    grader_normalize,
    normalized_contains,
    strip_trailing_unit,
    tokenize,
)


# --------------------------------------------------------------------------- text helpers
class TestText:
    def test_grader_normalisation_matches_the_published_rule(self) -> None:
        for variant in ["7ABC123", "7abc123", "7-ABC-123", "7 ABC 123", "7.ABC_123"]:
            assert grader_normalize(variant) == "7ABC123"
        assert grader_normalize("Q3 FY27") == "Q3FY27"
        assert grader_normalize("A·B") == "AB"

    def test_normalized_contains_is_separator_insensitive(self) -> None:
        assert normalized_contains("Board Rev: REV-C2", "rev c2")
        assert not normalized_contains("Board Rev: REV-C3", "REV-C2")
        assert not normalized_contains("anything", "")

    def test_tokenizer_matches_identifiers_with_or_without_separators(self) -> None:
        assert {"tq", "40", "tq40"} <= set(tokenize("TQ-40"))
        assert "tq40" in tokenize("TQ40")
        assert {"orr", "1847", "orr1847"} <= set(tokenize("ORR-1847"))

    def test_dotted_versions_do_not_produce_a_compact_noise_token(self) -> None:
        assert tokenize("4.3.2") == ["4", "3", "2"]

    def test_plural_s_is_stripped_only_from_plain_words(self) -> None:
        assert "part" in tokenize("parts")
        assert "class" in tokenize("class")  # 'ss' words are left alone
        assert "e7731" in tokenize("E7731")

    def test_content_terms_drop_stopwords(self) -> None:
        assert "what" not in content_terms("What is the maximum junction temperature?")
        assert {"maximum", "junction", "temperature"} <= content_terms(
            "What is the maximum junction temperature?"
        )

    def test_identifiers_mix_letters_and_digits(self) -> None:
        found = extract_identifiers(
            "Fixed ORR-1847 and E7731 on 2026-09-02, version 4.3.2, part ORR-FAN-2214-B"
        )
        assert {"ORR1847", "E7731", "ORRFAN2214B"} <= found
        assert not any(identifier in found for identifier in ("20260902", "432"))

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("94 C", "94"), ("94°C", "94"), ("94 °C", "94"), ("180 s", "180"), ("12 V", "12"), ("75%", "75"),
            ("4.3.2", "4.3.2"), ("Q3 FY27", "Q3 FY27"), ("REV-C2", "REV-C2"), ("ORR-FAN-2214-B", "ORR-FAN-2214-B"),
            ("10,000", "10,000"), ("B14", "B14"), ("E7731", "E7731"),
        ],
    )  # fmt: skip
    def test_trailing_units_are_stripped_from_bare_numbers_only(
        self, raw: str, expected: str
    ) -> None:
        assert strip_trailing_unit(raw) == expected


# --------------------------------------------------------------------------- parsers
class TestParsers:
    def test_plain_text_and_code_are_read(self, tmp_path: Path) -> None:
        (tmp_path / "a.log").write_text("line one\nline two\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("DEFAULT_BATCH_TIMEOUT_S = 180\n", encoding="utf-8")
        assert "line two" in parse_file(tmp_path / "a.log").segments[0].text
        assert "180" in parse_file(tmp_path / "b.py").segments[0].text

    def test_utf16_text_is_decoded(self, tmp_path: Path) -> None:
        (tmp_path / "u.txt").write_bytes("café total: 42".encode("utf-16"))
        assert "total: 42" in parse_file(tmp_path / "u.txt").segments[0].text

    def test_csv_rows_carry_their_header_names(self, tmp_path: Path) -> None:
        (tmp_path / "t.csv").write_text(
            "ticket,fixed_in\nORR-1,4.3.2\nORR-2,4.4.0\n", encoding="utf-8"
        )
        texts = [s.text for s in parse_file(tmp_path / "t.csv").segments]
        assert any("ticket: ORR-1" in t and "fixed_in: 4.3.2" in t for t in texts)

    def test_csv_with_a_semicolon_delimiter(self, tmp_path: Path) -> None:
        (tmp_path / "s.csv").write_text("id;value\nA1;77\nB2;88\n", encoding="utf-8")
        assert any("value: 88" in s.text for s in parse_file(tmp_path / "s.csv").segments)

    def test_xlsx_reads_every_sheet(self, tmp_path: Path) -> None:
        root = build_corpus(tmp_path / "c")
        texts = " ".join(
            s.text for s in parse_file(root / "support" / "rma_parts.xlsx").segments
        )
        assert "part_number: ORR-FAN-2214-B" in texts
        assert "unit_cost_usd: 38.5" in texts  # second sheet
        assert "lead_time_days: 14" in texts  # integral float rendered without ".0"

    def test_docx_tables_become_labelled_rows(self, tmp_path: Path) -> None:
        root = build_corpus(tmp_path / "c")
        texts = [s.text for s in parse_file(root / "planning" / "roadmap_fy27.docx").segments]
        assert any("Product: TQ-60" in t and "Quarter: Q3 FY27" in t for t in texts)
        assert any(t.startswith("# FY27 Roadmap") for t in texts)  # heading kept

    def test_pdf_pages_are_extracted(self, tmp_path: Path) -> None:
        (tmp_path / "d.pdf").write_bytes(make_pdf([["page one text"], ["page two text"]]))
        result = parse_file(tmp_path / "d.pdf")
        assert [s.locator for s in result.segments] == ["page 1", "page 2"]
        assert "page two text" in result.segments[1].text

    def test_withdrawn_marker_in_content_is_detected(self, tmp_path: Path) -> None:
        (tmp_path / "w.pdf").write_bytes(
            make_pdf([["WITHDRAWN - superseded by revision 2", "old data"]])
        )
        assert parse_file(tmp_path / "w.pdf").withdrawn is True
        (tmp_path / "n.pdf").write_bytes(make_pdf([["Current data"]]))
        assert parse_file(tmp_path / "n.pdf").withdrawn is False

    def test_an_encrypted_pdf_is_skipped_even_though_it_could_be_opened(
        self, tmp_path: Path
    ) -> None:
        root = build_corpus(tmp_path / "c")
        result = parse_file(root / "vendor" / "supplier_agreement_ENCRYPTED.pdf")
        assert result.segments == []
        assert result.skipped == "encrypted PDF"

    def test_an_unknown_binary_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        (tmp_path / "x.dat").write_bytes(os.urandom(512))
        assert parse_file(tmp_path / "x.dat").skipped == "unknown file type"

    def test_a_corrupt_pdf_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4\nthis is not a pdf")
        result = parse_file(tmp_path / "bad.pdf")
        assert result.segments == [] and result.skipped

    def test_a_corrupt_office_file_is_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "bad.docx").write_bytes(b"not a zip")
        (tmp_path / "bad.xlsx").write_bytes(b"not a zip")
        assert parse_file(tmp_path / "bad.docx").skipped
        assert parse_file(tmp_path / "bad.xlsx").skipped

    def test_an_empty_file_is_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "e.txt").write_text("   \n", encoding="utf-8")
        assert parse_file(tmp_path / "e.txt").skipped == "empty file"

    def test_permission_errors_are_caught(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "secret.txt"
        target.write_text("classified", encoding="utf-8")
        real_open = Path.open

        def deny(self: Path, *args: object, **kwargs: object) -> object:
            if self == target:
                raise PermissionError(13, "Permission denied")
            return real_open(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "open", deny)
        result = parse_file(target)
        assert result.segments == [] and "permission" in (result.skipped or "")

    @pytest.mark.skipif(
        os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
        reason="needs POSIX non-root",
    )
    def test_a_real_chmod_000_file_is_skipped(self, tmp_path: Path) -> None:
        target = tmp_path / "locked.txt"
        target.write_text("classified", encoding="utf-8")
        target.chmod(0o000)
        try:
            assert "permission" in (parse_file(target).skipped or "")
        finally:
            target.chmod(0o600)

    def test_kind_detection_by_extension_and_by_magic_bytes(self, tmp_path: Path) -> None:
        (tmp_path / "noext").write_bytes(b"%PDF-1.7 rest")
        (tmp_path / "img").write_bytes(b"\x89PNG\r\n\x1a\n....")
        (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02\x03")
        assert detect_kind(tmp_path / "noext") == "pdf"
        assert detect_kind(tmp_path / "img") == "image"
        assert detect_kind(tmp_path / "blob.bin") is None
        assert detect_kind(tmp_path / "PHOTO.JPG") == "image"

    def test_parse_file_skips_images_for_the_vision_pass(self, tmp_path: Path) -> None:
        (tmp_path / "p.png").write_bytes(b"\x89PNG")
        assert "vision" in (parse_file(tmp_path / "p.png").skipped or "")


# --------------------------------------------------------------------------- corpus / revisions
class TestCorpus:
    def test_revision_and_withdrawn_markers_come_out_of_the_file_name(self) -> None:
        family1, rev1, wd1 = family_and_revision("specs/tq40_datasheet_r1_WITHDRAWN.pdf")
        family2, rev2, wd2 = family_and_revision("specs/tq40_datasheet_r2.pdf")
        assert family1 == family2 == "specs/tq40_datasheet"
        assert wd1 is True and wd2 is False
        assert rev2 > rev1

    @pytest.mark.parametrize("name", ["plain.txt", "notes_final.txt", "report_2026.txt"])
    def test_names_without_a_revision_are_their_own_family(self, name: str) -> None:
        family, revision, _ = family_and_revision(name)
        assert family and revision == (0,)

    def test_older_revision_and_withdrawn_documents_are_marked_superseded(self) -> None:
        def record(path: str, withdrawn: bool = False) -> DocRecord:
            family, revision, name_withdrawn = family_and_revision(path)
            return DocRecord(
                path,
                "pdf",
                indexed=True,
                family=family,
                revision=revision,
                withdrawn=withdrawn or name_withdrawn,
            )

        old, new, other = record("s/a_r1.pdf"), record("s/a_r2.pdf"), record("s/b.pdf")
        mark_superseded([old, new, other])
        assert old.superseded and old.superseded_by == "s/a_r2.pdf"
        assert not new.superseded and not other.superseded

    def test_a_lone_withdrawn_document_is_still_marked_but_usable(self) -> None:
        family, revision, _ = family_and_revision("only_WITHDRAWN.pdf")
        only = DocRecord(
            "only_WITHDRAWN.pdf",
            "pdf",
            indexed=True,
            family=family,
            revision=revision,
            withdrawn=True,
        )
        mark_superseded([only])
        assert only.superseded and only.superseded_by is None

    def test_walk_handles_an_empty_directory_and_a_missing_one(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        assert list(walk_corpus(tmp_path)) == []
        assert list(walk_corpus(tmp_path / "does-not-exist")) == []

    def test_walk_order_is_stable_and_skips_noise_directories(self, tmp_path: Path) -> None:
        for name in ["b.txt", "a.txt", "sub/c.txt", "__pycache__/x.pyc", ".git/config"]:
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x", encoding="utf-8")
        assert [p.relative_to(tmp_path).as_posix() for p in walk_corpus(tmp_path)] == [
            "a.txt",
            "b.txt",
            "sub/c.txt",
        ]

    def test_fingerprint_changes_when_files_change(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("one", encoding="utf-8")
        first = corpus_fingerprint(tmp_path)
        (tmp_path / "b.txt").write_text("two", encoding="utf-8")
        assert corpus_fingerprint(tmp_path) != first


# --------------------------------------------------------------------------- index / retrieval
class TestIndex:
    def test_a_large_log_is_split_into_bounded_chunks(self) -> None:
        text = "\n".join(
            f"2026-09-02 12:00:{i % 60:02d} INFO event {i} processed ok" for i in range(400)
        )
        chunks = chunk_segments("logs/big.log", [Segment(text, "big.log")], target_tokens=60)
        assert len(chunks) > 20
        assert (
            max(len(c.text.split()) for c in chunks) <= 60 + 12
        )  # budget plus one overlap line
        assert [c.chunk_id for c in chunks] == list(range(len(chunks)))

    def test_a_single_huge_line_is_windowed_not_dropped(self) -> None:
        words = " ".join(f"w{i}" for i in range(1000))
        chunks = chunk_segments("x.txt", [Segment(words, "")], target_tokens=100)
        assert len(chunks) == 10 and "w999" in chunks[-1].text

    def test_overlap_never_emits_a_duplicate_chunk(self) -> None:
        lines = "\n".join(["alpha " * 50, "beta " * 50, "gamma " * 50])
        chunks = chunk_segments("x.txt", [Segment(lines, "")], target_tokens=50)
        assert len({c.text for c in chunks}) == len(chunks)

    def test_short_rows_stay_one_chunk_each(self) -> None:
        segments = [Segment(f"t | ticket: ORR-{i}", f"row {i}") for i in range(5)]
        assert len(chunk_segments("t.csv", segments)) == 5

    def _index(self, tmp_path: Path) -> ContractIndex:
        return build_text_index(build_corpus(tmp_path / "c"))[0]

    def test_identifier_queries_hit_the_right_file(self, tmp_path: Path) -> None:
        index = self._index(tmp_path)
        assert index.search("ORR-1847")[0].chunk.path == "support/bug_database.csv"
        assert index.search("orr1847 firmware")[0].chunk.path == "support/bug_database.csv"
        assert index.search("E7731")[0].chunk.path == "logs/prod_inference_2026-09-02.log"

    def test_file_names_are_searchable(self, tmp_path: Path) -> None:
        index = self._index(tmp_path)
        # the label image is not indexed in a text-only build, so use a text file's name
        assert (
            index.search("ingest service timeout")[0].chunk.path
            == "engineering/ingest_service.py"
        )

    def test_the_current_revision_outranks_the_withdrawn_one(self, tmp_path: Path) -> None:
        index = self._index(tmp_path)
        paths = [h.chunk.path for h in index.search("maximum junction temperature TQ-40")]
        assert paths.index("specs/tq40_datasheet_r2.pdf") < paths.index(
            "specs/tq40_datasheet_r1_WITHDRAWN.pdf"
        )

    def test_at_most_per_file_chunks_come_from_one_file(self, tmp_path: Path) -> None:
        index = self._index(tmp_path)
        hits = index.search("batch complete latency", top_k=12, per_file=2)
        counts: dict[str, int] = {}
        for hit in hits:
            counts[hit.chunk.path] = counts.get(hit.chunk.path, 0) + 1
        assert max(counts.values()) <= 2

    def test_a_query_with_no_matching_term_returns_nothing(self, tmp_path: Path) -> None:
        assert self._index(tmp_path).search("zzzz qqqq") == []

    def test_an_empty_index_is_searchable(self) -> None:
        index = ContractIndex()
        index.finalize()
        assert index.search("anything") == []

    def test_save_and_load_round_trip_preserves_search_results(self, tmp_path: Path) -> None:
        index = self._index(tmp_path)
        index.save(tmp_path / "saved")
        loaded = ContractIndex.load(tmp_path / "saved")
        assert loaded.fingerprint == index.fingerprint
        assert [(h.chunk.path, round(h.score, 6)) for h in loaded.search("ORR-1847")] == [
            (h.chunk.path, round(h.score, 6)) for h in index.search("ORR-1847")
        ]
        assert loaded.docs["specs/tq40_datasheet_r1_WITHDRAWN.pdf"].superseded
        assert loaded.identifier_files == index.identifier_files

    @pytest.mark.parametrize("damage", ["missing", "garbage", "wrong-size"])
    def test_a_damaged_lexical_file_falls_back_to_a_rebuild(
        self, tmp_path: Path, damage: str
    ) -> None:
        import pickle

        index = self._index(tmp_path)
        index.save(tmp_path / "saved")
        target = tmp_path / "saved" / "lexical.pkl"
        if damage == "missing":
            target.unlink()
        elif damage == "garbage":
            target.write_bytes(b"not a pickle")
        else:  # a valid pickle that belongs to a different (smaller) index
            with target.open("wb") as handle:
                pickle.dump(ContractIndex().__dict__["_lexical"].state(), handle)
        loaded = ContractIndex.load(tmp_path / "saved")
        assert [h.chunk.path for h in loaded.search("ORR-1847")] == [
            h.chunk.path for h in index.search("ORR-1847")
        ]

    def test_the_persisted_lexical_index_is_used_rather_than_rebuilt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        index = self._index(tmp_path)
        index.save(tmp_path / "saved")

        def forbidden(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("the lexical index should have been loaded from disk")

        monkeypatch.setattr("tabula_rag.contract.index._rebuild_lexical", forbidden)
        assert ContractIndex.load(tmp_path / "saved").search("E7731")

    def test_saving_over_an_existing_index_replaces_it(self, tmp_path: Path) -> None:
        index = self._index(tmp_path)
        index.save(tmp_path / "saved")
        index.save(tmp_path / "saved")
        assert len(ContractIndex.load(tmp_path / "saved").chunks) == len(index.chunks)

    def test_loading_a_missing_index_raises(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            ContractIndex.load(tmp_path / "nope")

    def test_encrypted_and_unknown_files_never_reach_the_index(self, tmp_path: Path) -> None:
        index = self._index(tmp_path)
        assert not index.docs["vendor/supplier_agreement_ENCRYPTED.pdf"].indexed
        assert not index.docs["vendor/telemetry_capture.dat"].indexed
        assert all("412" not in c.text for c in index.chunks)


# --------------------------------------------------------------------------- verification
class TestVerification:
    @pytest.fixture
    def index(self, tmp_path: Path) -> ContractIndex:
        return build_text_index(build_corpus(tmp_path / "c"))[0]

    def hits(self, index: ContractIndex, question: str) -> list[Hit]:
        return index.search(question, top_k=10)

    def test_clean_answer_reduces_prose_to_a_value(self) -> None:
        assert clean_answer('"94 C".') == "94"
        assert clean_answer("Answer: E7731") == "E7731"
        assert clean_answer("REV-C2\nbecause the label says so") == "REV-C2"
        assert clean_answer("   ") == ""

    def test_paths_are_resolved_from_sloppy_model_output(self, index: ContractIndex) -> None:
        cited = [
            "/app/corpus/specs/tq40_datasheet_r2.pdf",
            "SUPPORT/BUG_DATABASE.CSV",
            "`logs/nope.log`",
            "ingest_service.py",
            "specs/tq40_datasheet_r2.pdf",
        ]
        assert resolve_paths(index, cited) == [
            "specs/tq40_datasheet_r2.pdf",
            "support/bug_database.csv",
            "engineering/ingest_service.py",
        ]

    def test_a_grounded_answer_is_accepted(self, index: ContractIndex) -> None:
        q = "What is the maximum junction temperature of the TQ-40?"
        verdict = verify_and_cite(
            index, q, self.hits(index, q), "94", ["specs/tq40_datasheet_r2.pdf"]
        )
        assert (verdict.answer, verdict.citations) == ("94", ["specs/tq40_datasheet_r2.pdf"])
        assert verdict.confidence >= 0.9

    def test_an_invented_value_is_refused(self, index: ContractIndex) -> None:
        q = "What is the maximum junction temperature of the TQ-40?"
        verdict = verify_and_cite(
            index, q, self.hits(index, q), "112", ["specs/tq40_datasheet_r2.pdf"]
        )
        assert verdict.answer == "" and verdict.citations == []

    def test_a_value_that_only_exists_in_a_skipped_file_is_refused(
        self, index: ContractIndex
    ) -> None:
        q = "What is the unit price of the TQ-40 at 10,000 unit volume?"
        verdict = verify_and_cite(
            index, q, self.hits(index, q), "412.50", ["vendor/internal_audit.txt"]
        )
        assert verdict.answer == ""

    def test_a_topic_only_file_is_dropped_from_the_citations(
        self, index: ContractIndex
    ) -> None:
        q = "Which firmware version fixed ticket ORR-1847?"
        verdict = verify_and_cite(
            index, q, self.hits(index, q), "4.3.2",
            ["support/bug_database.csv", "engineering/meridian_release_notes.txt"])  # fmt: skip
        assert verdict.citations == ["support/bug_database.csv"]

    def test_a_file_that_shares_only_an_identifier_from_the_question_is_not_a_chain_link(
        self, index: ContractIndex
    ) -> None:
        q = "Which firmware version fixed ticket ORR-1847?"  # ORR-1847 is *in the question*
        verdict = verify_and_cite(
            index, q, self.hits(index, q), "4.3.2",
            ["support/bug_database.csv", "logs/prod_inference_2026-09-02.log"])  # fmt: skip
        assert verdict.citations == ["support/bug_database.csv"]

    def test_a_genuine_chain_keeps_both_files(self, index: ContractIndex) -> None:
        q = "The production log shows a thermal throttle incident. Which firmware release fixed the underlying defect?"
        verdict = verify_and_cite(
            index, q, self.hits(index, q), "4.3.2",
            ["logs/prod_inference_2026-09-02.log", "support/bug_database.csv"])  # fmt: skip
        assert set(verdict.citations) == {
            "logs/prod_inference_2026-09-02.log",
            "support/bug_database.csv",
        }

    def test_a_superseded_revision_is_dropped_when_the_current_one_carries_the_value(
        self, index: ContractIndex, tmp_path: Path
    ) -> None:
        # make the withdrawn revision *also* contain the value, so both are value-bearing
        record = index.docs["specs/tq40_datasheet_r1_WITHDRAWN.pdf"]
        record.text += "\nMaximum junction temperature: 94 deg C"
        q = "What is the maximum junction temperature of the TQ-40?"
        verdict = verify_and_cite(
            index, q, self.hits(index, q), "94",
            ["specs/tq40_datasheet_r1_WITHDRAWN.pdf", "specs/tq40_datasheet_r2.pdf"])  # fmt: skip
        assert verdict.citations == ["specs/tq40_datasheet_r2.pdf"]

    def test_when_the_model_cites_nothing_the_best_supported_file_is_used(
        self, index: ContractIndex
    ) -> None:
        q = "What error code is logged when the thermal throttle engages?"
        verdict = verify_and_cite(index, q, self.hits(index, q), "E7731", [])
        assert verdict.citations == ["logs/prod_inference_2026-09-02.log"]

    def test_when_the_model_cites_the_wrong_file_the_right_one_is_found(
        self, index: ContractIndex
    ) -> None:
        q = "What error code is logged when the thermal throttle engages?"
        verdict = verify_and_cite(
            index, q, self.hits(index, q), "E7731", ["engineering/ingest_service.py"]
        )
        assert verdict.citations[0] == "logs/prod_inference_2026-09-02.log"

    def test_an_empty_answer_stays_empty(self, index: ContractIndex) -> None:
        assert verify_and_cite(index, "q", [], "", []).answer == ""

    def test_the_answer_may_come_from_a_chunk_that_was_not_shown(
        self, index: ContractIndex
    ) -> None:
        q = "What is the default batch timeout, in seconds, in the ingest service?"
        assert verify_and_cite(index, q, self.hits(index, q)[:1], "180", []).answer == "180"
