"""Tests for the asymmetric query encoder and the small-weight threshold of SpladeIndex."""

import numpy as np
import torch

import _stubs

_stubs.install()

from search.splade import SpladeEncoder, SpladeIndex


def _encoder(**config):
    """A SpladeEncoder without a model: only the vector -> weights code is exercised."""
    enc = SpladeEncoder.__new__(SpladeEncoder)
    enc.quantization_factor = 100
    enc.quantize_queries = bool(config.get("quantize_queries", False))
    enc.min_weight = float(config.get("min_weight", 0.0))
    enc.vocab = {0: "a", 1: "b", 2: "c", 3: "d"}
    enc.vocab_size = 4
    return enc


def test_documents_keep_every_rounded_weight_without_a_threshold():
    enc = _encoder()
    assert enc.encode_document_vector(torch.tensor([0.004, 0.006, 0.5, 0.0])) == {"b": 1, "c": 50}


def test_the_threshold_drops_small_document_weights():
    enc = _encoder(min_weight=0.02)
    assert enc.encode_document_vector(torch.tensor([0.004, 0.019, 0.02, 0.5])) == {"c": 2, "d": 50}


def test_the_threshold_applies_to_quantized_and_float_queries():
    vec = np.array([0.004, 0.019, 0.02, 0.5], dtype=np.float32)
    assert _encoder(min_weight=0.02, quantize_queries=True)._output_to_weight_dicts(vec) == [{"d": 50, "c": 2}]
    assert list(_encoder(min_weight=0.02)._output_to_weight_dicts(vec)[0]) == ["d", "c"]
    assert list(_encoder()._output_to_weight_dicts(vec)[0]) == ["d", "c", "b", "a"]


def _index(tmp_path, **config):
    config.setdefault("name", "big/model")
    return SpladeIndex(config, data_dir=str(tmp_path))


def test_without_a_query_model_both_sides_share_one_encoder(tmp_path, monkeypatch):
    index = _index(tmp_path)
    monkeypatch.setattr("search.splade.SpladeEncoder", lambda config, bt, qf: ("enc", config["name"]))
    assert index._get_query_encoder() is index._get_encoder()
    assert index.name() == "splade_big_model"


def test_a_query_model_gets_its_own_encoder_and_the_same_index(tmp_path, monkeypatch):
    index = _index(tmp_path, query_model="small/student")
    monkeypatch.setattr("search.splade.SpladeEncoder", lambda config, bt, qf: ("enc", config["name"]))
    assert index._get_encoder() == ("enc", "big/model")
    assert index._get_query_encoder() == ("enc", "small/student")
    assert index.index_dir == _index(tmp_path).index_dir
    assert index.name() == "splade_big_model_q_small_student"
    # the impact searcher is handed the query encoder
    assert index.backend.encoder_provider() == ("enc", "small/student")
