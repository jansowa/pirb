"""Tests for the task loop that overlaps index builds with encoding."""

from types import SimpleNamespace

import pytest

import _stubs

# The scheduling and configuration logic under test never reaches pyserini, so
# stubs stand in for it when it is not installed.
_stubs.install()

import run_benchmark
from data import IndexInput
from run_benchmark import RetrievalEvaluator, evaluate_encoder


class FakeIndex:
    """Records the phase order of a build that defers its second half."""

    def __init__(self, task_id, events, deferred=True, exists=False):
        self.task_id = task_id
        self.events = events
        self.deferred = deferred
        self._exists = exists
        self._pending = False

    def supports_deferred_build(self) -> bool:
        return self.deferred

    def exists(self) -> bool:
        self._settle()
        return self._exists

    def results_exist(self, k, cache_prefix) -> bool:
        return False

    def build(self, docs):
        list(docs)
        self.events.append(f"encode:{self.task_id}")
        self._exists = True
        if self.deferred:
            self._pending = True
        else:
            self.events.append(f"lucene:{self.task_id}")

    def _settle(self):
        if self._pending:
            self._pending = False
            self.events.append(f"lucene:{self.task_id}")

    def search(self, queries, k, cache_prefix=None, overwrite=False):
        self._settle()
        self.events.append(f"search:{self.task_id}")
        return {}

    def index_path(self):
        return f"/tmp/{self.task_id}"

    def name(self):
        return f"fake_{self.task_id}"

    def accumulate_stats(self, stats):
        pass


class FakeTask:
    def __init__(self, task_id):
        self.task_id = task_id
        self.group_id = "group"
        self.skip_self = False

    def set_limit_queries(self, limit):
        pass

    def passages(self, data_dir):
        return iter([])

    def get_chunk_document_mapping(self, data_dir):
        return None

    def queries(self, data_dir):
        return iter([IndexInput(f"{self.task_id}-q0", "query", [f"{self.task_id}-doc0"], [1])])


def _args():
    return SimpleNamespace(
        recall_k=100,
        ndcg_k=10,
        query_limit=None,
        overwrite=False,
        data_dir="data",
        benchmark_label=None,
    )


def _run(monkeypatch, deferred, task_count=4):
    events = []
    tasks = [FakeTask(f"task{i}") for i in range(task_count)]
    indexes = {task.task_id: FakeIndex(task.task_id, events, deferred=deferred) for task in tasks}
    monkeypatch.setattr(
        run_benchmark.AutoIndex,
        "from_config",
        staticmethod(lambda encoder, task, args: indexes[task.task_id]),
    )
    monkeypatch.setattr(RetrievalEvaluator, "_log_score", lambda *a, **kw: None)
    evaluator = RetrievalEvaluator(_args())
    index_paths = []
    results, ndcg_tasks = evaluate_encoder({"name": "fake"}, tasks, evaluator, _args(), index_paths)
    return events, results, index_paths


def test_next_task_is_encoded_while_the_index_build_is_pending(monkeypatch):
    events, _, _ = _run(monkeypatch, deferred=True)

    # task1 is encoded before task0's index build is collected.
    assert events.index("encode:task1") < events.index("lucene:task0")
    # No search ever runs before its own index build finished.
    for task_id in ("task0", "task1", "task2", "task3"):
        assert events.index(f"lucene:{task_id}") < events.index(f"search:{task_id}")


def test_sequential_indexes_keep_the_previous_order(monkeypatch):
    events, _, _ = _run(monkeypatch, deferred=False)

    assert events == [
        item
        for task_id in ("task0", "task1", "task2", "task3")
        for item in (f"encode:{task_id}", f"lucene:{task_id}", f"search:{task_id}")
    ]


def test_every_task_is_evaluated_exactly_once(monkeypatch):
    events, results, index_paths = _run(monkeypatch, deferred=True)

    assert sorted(results) == ["task0", "task1", "task2", "task3"]
    assert len(index_paths) == 4
    assert events.count("encode:task0") == 1
    assert [event for event in events if event.startswith("search:")] == [
        "search:task0",
        "search:task1",
        "search:task2",
        "search:task3",
    ]


def test_an_already_built_index_is_not_rebuilt(monkeypatch):
    events = []
    tasks = [FakeTask("task0")]
    index = FakeIndex("task0", events, deferred=True, exists=True)
    monkeypatch.setattr(
        run_benchmark.AutoIndex,
        "from_config",
        staticmethod(lambda encoder, task, args: index),
    )
    monkeypatch.setattr(RetrievalEvaluator, "_log_score", lambda *a, **kw: None)

    evaluate_encoder({"name": "fake"}, tasks, RetrievalEvaluator(_args()), _args(), [])

    assert events == ["search:task0"]


def test_overwrite_forces_a_rebuild(monkeypatch):
    events = []
    tasks = [FakeTask("task0")]
    index = FakeIndex("task0", events, deferred=True, exists=True)
    monkeypatch.setattr(
        run_benchmark.AutoIndex,
        "from_config",
        staticmethod(lambda encoder, task, args: index),
    )
    monkeypatch.setattr(RetrievalEvaluator, "_log_score", lambda *a, **kw: None)
    args = _args()
    args.overwrite = True

    evaluate_encoder({"name": "fake"}, tasks, RetrievalEvaluator(args), args, [])

    assert events == ["encode:task0", "lucene:task0", "search:task0"]
