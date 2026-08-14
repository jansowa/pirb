"""Tests for telling a finished Lucene index from a half-written one."""

import os

import pytest

import _stubs

_stubs.install()

import backend.sparse as sparse_module  # noqa: E402
from backend.base import SparseBackend  # noqa: E402
from utils.artifacts import read_index_manifest, write_index_manifest  # noqa: E402


def _index_dir(tmp_path, segments=True):
    index_dir = tmp_path / "lucene_index"
    index_dir.mkdir()
    if segments:
        (index_dir / "segments_1").write_text("segments", encoding="utf-8")
    return str(index_dir)


def _backend(index_dir, num_docs, expected=None, monkeypatch=None):
    monkeypatch.setattr(sparse_module, "lucene_num_docs", lambda path: num_docs)
    provider = (lambda: expected) if expected is not None else None
    return sparse_module.LuceneBackend(index_dir, encoder_provider=lambda: None,
                                      expected_doc_count_provider=provider)


def test_a_missing_directory_is_not_complete(tmp_path, monkeypatch):
    backend = _backend(str(tmp_path / "absent"), num_docs=10, monkeypatch=monkeypatch)

    assert backend.exists() is False
    assert backend.is_complete() is False


def test_a_directory_without_segments_is_not_complete(tmp_path, monkeypatch):
    index_dir = _index_dir(tmp_path, segments=False)
    backend = _backend(index_dir, num_docs=10, monkeypatch=monkeypatch)

    assert backend.exists() is True, "the directory is there, which is what used to be checked"
    assert backend.is_complete() is False


def test_a_matching_document_count_is_complete(tmp_path, monkeypatch):
    index_dir = _index_dir(tmp_path)
    write_index_manifest(index_dir, 10)
    backend = _backend(index_dir, num_docs=10, monkeypatch=monkeypatch)

    assert backend.is_complete() is True


def test_an_index_truncated_after_committing_segments_is_not_complete(tmp_path, monkeypatch):
    index_dir = _index_dir(tmp_path)
    write_index_manifest(index_dir, 10)
    backend = _backend(index_dir, num_docs=7, monkeypatch=monkeypatch)

    assert backend.is_complete() is False, "an index short of documents opens fine but scores worse"


def test_an_unreadable_index_is_not_complete(tmp_path, monkeypatch):
    index_dir = _index_dir(tmp_path)
    write_index_manifest(index_dir, 10)
    backend = _backend(index_dir, num_docs=None, monkeypatch=monkeypatch)

    assert backend.is_complete() is False


def test_a_verified_legacy_index_gets_a_manifest(tmp_path, monkeypatch):
    index_dir = _index_dir(tmp_path)
    backend = _backend(index_dir, num_docs=10, expected=10, monkeypatch=monkeypatch)

    assert backend.is_complete() is True
    assert read_index_manifest(index_dir)["doc_count"] == 10


def test_a_short_legacy_index_is_rejected(tmp_path, monkeypatch):
    index_dir = _index_dir(tmp_path)
    backend = _backend(index_dir, num_docs=7, expected=10, monkeypatch=monkeypatch)

    assert backend.is_complete() is False
    assert read_index_manifest(index_dir) is None


def test_a_legacy_index_without_an_expected_count_is_kept(tmp_path, monkeypatch):
    """Nothing to compare against: keep the index rather than invalidate the cache."""
    index_dir = _index_dir(tmp_path)
    backend = _backend(index_dir, num_docs=10, monkeypatch=monkeypatch)

    assert backend.is_complete() is True
    assert read_index_manifest(index_dir)["doc_count"] == 10


def test_a_failing_expected_count_provider_does_not_break_the_check(tmp_path, monkeypatch):
    index_dir = _index_dir(tmp_path)

    def explode():
        raise OSError("corpus unavailable")

    monkeypatch.setattr(sparse_module, "lucene_num_docs", lambda path: 10)
    backend = sparse_module.LuceneBackend(index_dir, encoder_provider=lambda: None,
                                          expected_doc_count_provider=explode)

    assert backend.is_complete() is True


def test_other_sparse_backends_keep_the_old_behaviour(tmp_path):
    """BM25 shares SparseBackend and must not start validating anything."""

    class Simple(SparseBackend):
        def build(self, corpus_path, target_dir=None):
            pass

        def search(self, queries, encoded_queries, ids, top_k):
            return {}

        def exists(self):
            return os.path.isdir(self.index_dir)

    backend = Simple(str(tmp_path / "bm25_index"))
    assert backend.is_complete() is False

    os.makedirs(backend.index_dir)
    assert backend.is_complete() is True
