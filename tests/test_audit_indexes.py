"""Tests for scripts/audit_indexes.py."""

import importlib.util
import json
import os
from types import SimpleNamespace

import pytest

from search.splade_artifacts import PASSAGES_MANIFEST_NAME, PassagesManifest
from utils.artifacts import atomic_write_json, write_index_manifest

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "audit_indexes.py")


@pytest.fixture
def audit_module():
    spec = importlib.util.spec_from_file_location("audit_indexes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cache(tmp_path, audit_module, monkeypatch):
    """A cache with one healthy task and one whose index was truncated."""
    counts = {}
    monkeypatch.setattr(audit_module, "lucene_num_docs", lambda index_dir: counts.get(index_dir))
    monkeypatch.setattr(audit_module, "lucene_available", lambda: True)

    def task(task_id, documents, index_docs, manifest=True, index_manifest=True):
        corpus = tmp_path / "data" / task_id / "passages" / "passages.jsonl"
        corpus.parent.mkdir(parents=True)
        with open(corpus, "w", encoding="utf-8") as corpus_file:
            for i in range(documents):
                corpus_file.write(json.dumps({"id": f"doc_{i}", "contents": "text"}) + "\n")
        model_dir = tmp_path / "cache" / task_id / "some_model"
        docs_dir = model_dir / "docs"
        docs_dir.mkdir(parents=True)
        passages = docs_dir / "passages.jsonl"
        with open(passages, "w", encoding="utf-8") as passages_file:
            for i in range(documents):
                passages_file.write(json.dumps({"id": f"doc_{i}", "vector": {"a": 1}}) + "\n")
        if manifest:
            stored = PassagesManifest(
                task_id=task_id, model_path="/models/final", corpus_path=str(corpus),
                corpus_doc_count=documents, written_doc_count=documents,
                written_bytes=os.path.getsize(passages), last_doc_id=f"doc_{documents - 1}", complete=True,
            )
            atomic_write_json(str(model_dir / PASSAGES_MANIFEST_NAME), stored.to_dict())
        index_dir = model_dir / "lucene_index"
        index_dir.mkdir()
        (index_dir / "segments_1").write_text("segments", encoding="utf-8")
        if index_manifest:
            write_index_manifest(str(index_dir), documents)
        counts[str(index_dir)] = index_docs
        return model_dir, index_dir, passages

    return SimpleNamespace(tmp_path=tmp_path, make_task=task, module=audit_module)


def _run(module, tmp_path, *args):
    return module.main([
        "--cache-dir", str(tmp_path / "cache"),
        "--data-dir", str(tmp_path / "data"),
        *args,
    ])


def test_a_dry_run_reports_without_touching_anything(cache, capsys):
    _healthy, index_dir, passages = cache.make_task("good", documents=10, index_docs=10)
    _model, broken_index, _passages = cache.make_task("bad", documents=10, index_docs=6)

    assert _run(cache.module, cache.tmp_path, "--json", str(cache.tmp_path / "report.json")) == 0

    output = capsys.readouterr().out
    assert "truncated" in output
    assert "1 broken index(es)" in output
    assert "would be freed" in output
    assert os.path.isdir(broken_index), "a dry run must not delete anything"
    assert os.path.isdir(index_dir)
    assert os.path.exists(passages)
    report = json.loads((cache.tmp_path / "report.json").read_text(encoding="utf-8"))
    verdicts = {entry["task_id"]: entry["verdict"] for entry in report["indexes"]}
    assert verdicts == {"good": "complete", "bad": "truncated"}


def test_fix_deletes_broken_indexes_and_keeps_the_encoded_corpus(cache):
    _model, broken_index, passages = cache.make_task("bad", documents=10, index_docs=6)

    _run(cache.module, cache.tmp_path, "--fix")

    assert not os.path.exists(broken_index)
    assert os.path.exists(passages), "passages.jsonl is worth GPU hours and must survive --fix"


def test_fix_adds_a_manifest_to_a_verified_legacy_index(cache):
    from utils.artifacts import read_index_manifest

    _model, index_dir, _passages = cache.make_task("legacy", documents=10, index_docs=10, index_manifest=False)

    _run(cache.module, cache.tmp_path, "--fix")

    assert read_index_manifest(str(index_dir))["doc_count"] == 10


def test_an_index_without_segments_is_reported_and_removed(cache):
    _model, index_dir, _passages = cache.make_task("nosegments", documents=10, index_docs=10)
    os.remove(os.path.join(index_dir, "segments_1"))

    _run(cache.module, cache.tmp_path, "--fix")

    assert not os.path.exists(index_dir)


def test_a_legacy_encoded_corpus_is_reported(cache, capsys):
    cache.make_task("legacy", documents=10, index_docs=10, manifest=False)

    _run(cache.module, cache.tmp_path)

    assert "no-manifest" in capsys.readouterr().out


def test_fix_refuses_to_run_where_no_index_can_be_read(cache, monkeypatch):
    """Without pyserini every index looks broken; deleting them all would be wrong."""
    _model, index_dir, _passages = cache.make_task("good", documents=10, index_docs=10)
    monkeypatch.setattr(cache.module, "lucene_available", lambda: False)

    with pytest.raises(SystemExit, match="pyserini is not installed"):
        _run(cache.module, cache.tmp_path, "--fix")

    assert os.path.isdir(index_dir)


def test_purging_passages_can_be_filtered_by_model_path(cache):
    _model, _index, kept = cache.make_task("keep", documents=10, index_docs=10)
    manifest_path = os.path.join(os.path.dirname(os.path.dirname(kept)), PASSAGES_MANIFEST_NAME)
    stored = json.loads(open(manifest_path, encoding="utf-8").read())
    stored["model_path"] = "/models/checkpoint-100"
    atomic_write_json(manifest_path, stored)
    _model2, _index2, purged = cache.make_task("purge", documents=10, index_docs=10)

    _run(cache.module, cache.tmp_path, "--fix", "--purge-passages-matching", "final")

    assert os.path.exists(kept), "a corpus from another checkpoint must not be purged"
    assert not os.path.exists(purged)
