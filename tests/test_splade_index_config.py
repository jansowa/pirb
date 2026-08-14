"""Tests for SpladeIndex configuration knobs."""

import pytest

import _stubs

# The scheduling and configuration logic under test never reaches pyserini, so
# stubs stand in for it when it is not installed.
_stubs.install()

from search.splade import SpladeIndex


def _index(tmp_path, **config):
    config.setdefault("name", "some/model")
    return SpladeIndex(config, data_dir=str(tmp_path))


def test_batch_sizes_default_to_thirty_two(tmp_path):
    index = _index(tmp_path)

    assert index.batch_size == 32
    assert index.index_batch_size == 32


def test_batch_size_is_read_from_config(tmp_path):
    index = _index(tmp_path, batch_size=64)

    assert index.batch_size == 64
    assert index.index_batch_size == 64


def test_index_batch_size_overrides_batch_size(tmp_path):
    index = _index(tmp_path, batch_size=64, index_batch_size=16)

    assert index.batch_size == 64
    assert index.index_batch_size == 16


def test_threads_are_read_from_config(tmp_path):
    index = _index(tmp_path, threads=3)

    assert index.threads == 3
    assert index.backend.threads == 3


def test_backend_threads_default_to_eight(tmp_path):
    index = _index(tmp_path)

    assert index.backend.threads == 8


def test_build_batches_use_the_index_batch_size(tmp_path, monkeypatch):
    from data import IndexInput

    # Built inline so that the stubbed backend cannot outlive the monkeypatch.
    index = _index(tmp_path, batch_size=64, index_batch_size=4, index_build_workers=0)
    index.encoder = object()
    monkeypatch.setattr(index, "_get_encoder", lambda: index.encoder)
    monkeypatch.setattr(index.backend, "build", lambda docs_path, target_dir=None: None)
    sizes = []
    monkeypatch.setattr(index, "_write_batch", lambda batch, output: sizes.append(len(batch)))

    index.build([IndexInput(str(i), f"text {i}") for i in range(10)])

    assert sizes == [4, 4, 2]
