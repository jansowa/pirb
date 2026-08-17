"""Tests for backgrounded Lucene index builds and the evaluation pipeline."""

import os
import threading
from types import SimpleNamespace
import time

import pytest

import _stubs

# The scheduling and configuration logic under test never reaches pyserini, so
# stubs stand in for it when it is not installed.
_stubs.install()

import search.splade as splade_module
from data import IndexInput
from search.splade import SpladeIndex
from utils.artifacts import read_index_manifest


@pytest.fixture(autouse=True)
def _reset_shared_pool():
    splade_module.shutdown_build_executor()
    splade_module._ACTIVE_BUILDS.clear()
    yield
    splade_module.shutdown_build_executor()
    splade_module._ACTIVE_BUILDS.clear()


def _index(tmp_path, name="model", **config):
    config.setdefault("name", name)
    index = SpladeIndex(config, data_dir=str(tmp_path / name))
    index.encoder = SimpleNamespace()
    index._get_encoder = lambda: index.encoder
    index._write_batch = lambda batch, output: None
    # These tests care about scheduling, not about Lucene: an index is "complete"
    # when the build wrote its manifest. Validating the count is covered by
    # tests/test_lucene_backend_complete.py.
    index.backend.num_docs = lambda index_dir=None: (
        read_index_manifest(index_dir or index.backend.index_dir) or {}).get("doc_count")
    return index


def _build(index, backend_build, documents=2):
    def build(docs_path, target_dir=None):
        directory = target_dir or index.index_dir
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "segments_1"), "w", encoding="utf-8") as segments:
            segments.write("segments")
        return backend_build(docs_path)

    index.backend.build = build
    index.build([IndexInput(str(i), f"text {i}") for i in range(documents)])


def test_deferred_build_returns_before_the_index_is_ready(tmp_path):
    index = _index(tmp_path)
    released = threading.Event()
    finished = threading.Event()

    def slow_build(docs_path):
        released.wait(5)
        finished.set()

    assert index.supports_deferred_build() is True
    _build(index, slow_build)
    assert not finished.is_set()
    released.set()
    index.wait_for_build()
    assert finished.is_set()


def test_inline_build_keeps_previous_behaviour(tmp_path):
    index = _index(tmp_path, index_build_workers=0)
    calls = []

    assert index.supports_deferred_build() is False
    _build(index, lambda docs_path: calls.append(docs_path))
    assert len(calls) == 1
    assert index._pending_build is None


def test_search_waits_for_a_pending_build(tmp_path):
    index = _index(tmp_path)
    order = []
    released = threading.Event()

    def slow_build(docs_path):
        released.wait(5)
        order.append("build")

    _build(index, slow_build)

    def search_now():
        index.backend.search = lambda *a, **kw: order.append("search") or {}
        index.encoder.encode_queries = lambda texts, batch_size, query_ids: {}
        index.search([IndexInput("q1", "query")], k=10, verbose=False)

    searcher = threading.Thread(target=search_now)
    searcher.start()
    time.sleep(0.2)
    assert order == []
    released.set()
    searcher.join(5)

    assert order == ["build", "search"]


def test_build_failure_propagates_to_the_waiting_thread(tmp_path):
    index = _index(tmp_path)

    def failing_build(docs_path):
        raise RuntimeError("lucene exploded")

    _build(index, failing_build)

    with pytest.raises(RuntimeError, match="lucene exploded"):
        index.wait_for_build()


def test_build_failure_reaches_search(tmp_path):
    index = _index(tmp_path)
    _build(index, lambda docs_path: 1 / 0)

    with pytest.raises(ZeroDivisionError):
        index.search([IndexInput("q1", "query")], k=10, verbose=False)


def test_exists_waits_for_a_pending_build(tmp_path):
    index = _index(tmp_path)
    completed = []
    released = threading.Event()

    def slow_build(docs_path):
        released.wait(5)
        completed.append(True)

    _build(index, slow_build)
    released.set()

    assert index.exists() is True
    assert completed == [True]


def test_concurrent_builds_are_capped_by_the_worker_limit(tmp_path):
    running = []
    peak = []
    lock = threading.Lock()
    released = threading.Event()

    def slow_build(docs_path):
        with lock:
            running.append(1)
            peak.append(len(running))
        released.wait(5)
        with lock:
            running.pop()

    indexes = [_index(tmp_path, name=f"model{i}") for i in range(4)]
    for index in indexes:
        _build(index, slow_build)
    time.sleep(0.2)
    released.set()
    for index in indexes:
        index.wait_for_build()

    assert max(peak) == 1


def test_a_shared_index_directory_is_not_removed_under_a_running_build(tmp_path):
    order = []
    released = threading.Event()

    def slow_build(docs_path):
        released.wait(5)
        order.append("first-build-done")

    first = _index(tmp_path, name="shared")
    _build(first, slow_build)
    # The index is staged aside and renamed into place only once it is finished,
    # so nothing at the final path can be mistaken for a usable index yet.
    assert not os.path.isdir(first.index_dir)
    assert [name for name in os.listdir(first.base_dir) if ".tmp-" in name]

    # A second index over the same cache directory, as tasks sharing a cache do.
    second = _index(tmp_path, name="shared")
    assert second.index_dir == first.index_dir

    def rebuild():
        order.append("second-build-start")
        _build(second, lambda docs_path: order.append("second-build-done"))

    thread = threading.Thread(target=rebuild)
    thread.start()
    time.sleep(0.2)
    released.set()
    thread.join(5)
    # The second build is backgrounded too, so wait for it before ordering.
    second.wait_for_build()

    assert order.index("first-build-done") < order.index("second-build-done")


def test_close_waits_for_a_pending_build(tmp_path):
    index = _index(tmp_path)
    finished = []
    released = threading.Event()

    def slow_build(docs_path):
        released.wait(5)
        finished.append(True)

    _build(index, slow_build)
    released.set()
    index.close()

    assert finished == [True]


def test_worker_limit_is_configurable(tmp_path):
    assert _index(tmp_path, index_build_workers=3).index_build_workers == 3
    assert _index(tmp_path, name="m2", index_build_workers=0).index_build_workers == 0
    assert _index(tmp_path, name="m3").index_build_workers == 1
