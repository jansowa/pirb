"""Tests for the manifest and the resume decision behind a splade build."""

import json
import os

import pytest

from search.splade_artifacts import (
    REBUILD,
    RESUME,
    REUSE,
    PassagesManifest,
    PassagesWriter,
    ResumeMismatch,
    decide_phase_a,
    prepare_resume,
    skip_documents,
)
from utils.artifacts import atomic_write_json, cached_line_count, model_fingerprint, remove_stale_dirs

from data import IndexInput


def _corpus(tmp_path, documents=10):
    path = tmp_path / "corpus.jsonl"
    with open(path, "w", encoding="utf-8") as corpus_file:
        for i in range(documents):
            corpus_file.write(json.dumps({"id": f"doc_{i}", "contents": f"text {i}"}) + "\n")
    return str(path)


def _expected(corpus_path, fingerprint="sha256:aaa"):
    stat = os.stat(corpus_path)
    return PassagesManifest(
        task_id="task", model_path="/models/checkpoint", model_fingerprint=fingerprint,
        max_seq_length=512, quantization_factor=100, corpus_path=corpus_path,
        corpus_size_bytes=stat.st_size, corpus_mtime=stat.st_mtime,
    )


def _write_passages(path, documents, ids_from=0):
    with open(path, "a", encoding="utf-8") as passages_file:
        for i in range(ids_from, ids_from + documents):
            passages_file.write(json.dumps({"id": f"doc_{i}", "content": "", "vector": {"a": 1}}) + "\n")


def _stored(tmp_path, expected, documents, complete):
    passages = str(tmp_path / "passages.jsonl")
    _write_passages(passages, documents)
    manifest = PassagesManifest(**expected.to_dict())
    manifest.written_doc_count = documents
    manifest.written_bytes = os.path.getsize(passages)
    manifest.last_doc_id = f"doc_{documents - 1}"
    manifest.complete = complete
    manifest_path = str(tmp_path / "passages_manifest.json")
    atomic_write_json(manifest_path, manifest.to_dict())
    return manifest_path, passages, manifest


def test_a_complete_manifest_is_reused(tmp_path):
    corpus = _corpus(tmp_path)
    expected = _expected(corpus)
    manifest_path, passages, _ = _stored(tmp_path, expected, 10, complete=True)

    decision = decide_phase_a(manifest_path, passages, expected)

    assert decision.action == REUSE
    assert decision.manifest.written_doc_count == 10


def test_a_partial_manifest_is_resumed(tmp_path):
    corpus = _corpus(tmp_path)
    expected = _expected(corpus)
    manifest_path, passages, _ = _stored(tmp_path, expected, 4, complete=False)

    decision = decide_phase_a(manifest_path, passages, expected)

    assert decision.action == RESUME
    assert decision.skip_docs == 4


def test_a_changed_model_fingerprint_is_rejected(tmp_path):
    corpus = _corpus(tmp_path)
    manifest_path, passages, _ = _stored(tmp_path, _expected(corpus), 4, complete=False)

    decision = decide_phase_a(manifest_path, passages, _expected(corpus, fingerprint="sha256:bbb"))

    assert decision.action == REBUILD
    assert "fingerprint" in decision.reason


def test_an_unfingerprintable_model_is_rejected(tmp_path):
    corpus = _corpus(tmp_path)
    manifest_path, passages, _ = _stored(tmp_path, _expected(corpus), 4, complete=True)

    decision = decide_phase_a(manifest_path, passages, _expected(corpus, fingerprint=None))

    assert decision.action == REBUILD


def test_a_changed_corpus_mtime_is_rejected(tmp_path):
    corpus = _corpus(tmp_path)
    expected = _expected(corpus)
    manifest_path, passages, _ = _stored(tmp_path, expected, 10, complete=True)
    os.utime(corpus, (expected.corpus_mtime + 120, expected.corpus_mtime + 120))

    decision = decide_phase_a(manifest_path, passages, _expected(corpus))

    assert decision.action == REBUILD
    assert "mtime" in decision.reason


def test_a_legacy_file_is_rejected_without_the_opt_in(tmp_path):
    corpus = _corpus(tmp_path)
    passages = str(tmp_path / "passages.jsonl")
    _write_passages(passages, 10)

    decision = decide_phase_a(str(tmp_path / "missing.json"), passages, _expected(corpus))

    assert decision.action == REBUILD
    assert "no manifest" in decision.reason


def test_a_legacy_file_is_trusted_when_it_is_complete(tmp_path):
    corpus = _corpus(tmp_path)
    passages = str(tmp_path / "passages.jsonl")
    _write_passages(passages, 10)

    decision = decide_phase_a(str(tmp_path / "missing.json"), passages, _expected(corpus), trust_legacy=True)

    assert decision.action == REUSE
    assert decision.manifest.complete is True
    assert decision.manifest.written_doc_count == 10
    assert decision.manifest.last_doc_id == "doc_9"


def test_a_short_legacy_file_is_still_rejected(tmp_path):
    corpus = _corpus(tmp_path)
    passages = str(tmp_path / "passages.jsonl")
    _write_passages(passages, 6)

    decision = decide_phase_a(str(tmp_path / "missing.json"), passages, _expected(corpus), trust_legacy=True)

    assert decision.action == REBUILD
    assert "6 of 10" in decision.reason


def test_a_torn_tail_is_truncated_before_resuming(tmp_path):
    corpus = _corpus(tmp_path)
    manifest_path, passages, manifest = _stored(tmp_path, _expected(corpus), 4, complete=False)
    with open(passages, "ab") as passages_file:
        passages_file.write(b'{"id": "doc_4", "content": "", "vec')

    ok, reason, resumed = prepare_resume(passages, manifest)

    assert ok is True
    assert resumed.written_doc_count == 4
    assert os.path.getsize(passages) == manifest.written_bytes
    assert reason.endswith("'doc_3'")


def test_an_unreadable_last_line_rolls_the_count_back(tmp_path):
    corpus = _corpus(tmp_path)
    manifest_path, passages, manifest = _stored(tmp_path, _expected(corpus), 4, complete=False)
    with open(passages, "a", encoding="utf-8") as passages_file:
        passages_file.write("not json at all\n")
    manifest.written_doc_count = 5
    manifest.written_bytes = os.path.getsize(passages)
    manifest.last_doc_id = "doc_4"

    ok, _reason, resumed = prepare_resume(passages, manifest)

    assert ok is True
    assert resumed.written_doc_count == 4
    assert resumed.last_doc_id == "doc_3"


def test_a_mismatched_last_document_refuses_to_resume(tmp_path):
    corpus = _corpus(tmp_path)
    _manifest_path, passages, manifest = _stored(tmp_path, _expected(corpus), 4, complete=False)
    manifest.last_doc_id = "doc_999"

    ok, reason, _resumed = prepare_resume(passages, manifest)

    assert ok is False
    assert "does not match the manifest" in reason


def test_skipping_documents_verifies_the_position(tmp_path):
    docs = [IndexInput(f"doc_{i}", f"text {i}") for i in range(10)]

    remaining = skip_documents(iter(docs), 4, "doc_3")

    assert [doc.id for doc in remaining] == [f"doc_{i}" for i in range(4, 10)]


def test_skipping_documents_detects_a_shifted_corpus(tmp_path):
    docs = [IndexInput(f"other_{i}", f"text {i}") for i in range(10)]

    with pytest.raises(ResumeMismatch, match="doc_3"):
        skip_documents(iter(docs), 4, "doc_3")


def test_skipping_documents_detects_a_shortened_corpus(tmp_path):
    docs = [IndexInput(f"doc_{i}", f"text {i}") for i in range(2)]

    with pytest.raises(ResumeMismatch, match="fewer"):
        skip_documents(iter(docs), 4, "doc_3")


def test_the_manifest_never_claims_more_than_is_on_disk(tmp_path):
    passages = str(tmp_path / "passages.jsonl")
    manifest_path = str(tmp_path / "passages_manifest.json")
    manifest = PassagesManifest(task_id="task", corpus_path=str(_corpus(tmp_path)))

    with PassagesWriter(passages, manifest_path, manifest, flush_batches=2) as writer:
        for batch in range(5):
            writer.file.write(json.dumps({"id": f"doc_{batch}", "vector": {}}) + "\n")
            writer.note_batch(1, f"doc_{batch}")
            stored = json.loads(open(manifest_path, encoding="utf-8").read())
            assert stored["written_bytes"] <= os.path.getsize(passages)
        final = writer.finalize()

    assert final.complete is True
    assert final.written_doc_count == 5
    assert final.corpus_doc_count == 5
    assert json.loads(open(manifest_path, encoding="utf-8").read())["complete"] is True


def test_the_corpus_line_count_is_cached_next_to_it(tmp_path):
    corpus = _corpus(tmp_path, documents=7)

    assert cached_line_count(corpus, compute=False) is None
    assert cached_line_count(corpus) == 7
    assert cached_line_count(corpus, compute=False) == 7

    with open(corpus, "a", encoding="utf-8") as corpus_file:
        corpus_file.write(json.dumps({"id": "doc_7", "contents": "text"}) + "\n")

    assert cached_line_count(corpus, compute=False) is None


def test_the_model_fingerprint_follows_the_weight_files(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")

    first = model_fingerprint(str(model_dir))
    assert first is not None and first.startswith("sha256:")
    assert model_fingerprint(str(model_dir)) == first

    (model_dir / "model.safetensors").write_bytes(b"different weights")
    assert model_fingerprint(str(model_dir)) != first
    assert model_fingerprint(str(tmp_path / "absent")) is None


def test_stale_build_directories_are_removed(tmp_path):
    index_dir = str(tmp_path / "lucene_index")
    stale = tmp_path / "lucene_index.tmp-123-456"
    fresh = tmp_path / "lucene_index.old-999-999"
    for directory in (stale, fresh):
        directory.mkdir()
    os.utime(stale, (0, 0))

    assert remove_stale_dirs(index_dir) == 1
    assert not stale.exists()
    assert fresh.exists()
