"""Equivalence tests for the vectorized splade document serialization."""

import io
import json

import numpy as np
import pytest
import torch

pytest.importorskip("pyserini", reason="splade index requires pyserini")

from data import IndexInput
from search.splade import SpladeEncoder, SpladeIndex


VOCAB_SIZE = 512


def _encoder(quantization_factor: int = 100) -> SpladeEncoder:
    """A SpladeEncoder with just the fields the serialization touches."""
    encoder = SpladeEncoder.__new__(SpladeEncoder)
    encoder.quantization_factor = quantization_factor
    encoder.vocab = {index: f"tok{index}" for index in range(VOCAB_SIZE)}
    return encoder


def _random_batch(rows: int, dtype: torch.dtype, seed: int, sparsity: float = 0.9) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand((rows, VOCAB_SIZE), generator=generator) * 3.0
    mask = torch.rand((rows, VOCAB_SIZE), generator=generator) < sparsity
    values[mask] = 0.0
    return values.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows", [1, 2, 32, 33])
def test_vectorized_matches_legacy_on_random_batches(dtype, rows):
    encoder = _encoder()
    batch = _random_batch(rows, dtype, seed=rows * 17 + (1 if dtype == torch.float16 else 2))

    vectorized = encoder.encode_document_vectors(batch)
    legacy = [encoder.encode_document_vector_legacy(row) for row in batch]

    assert vectorized == legacy
    assert [json.dumps(v, ensure_ascii=False) for v in vectorized] == [
        json.dumps(v, ensure_ascii=False) for v in legacy
    ]


@pytest.mark.parametrize("quantization_factor", [1, 10, 100, 1000])
def test_vectorized_matches_legacy_for_quantization_factors(quantization_factor):
    encoder = _encoder(quantization_factor)
    batch = _random_batch(16, torch.float16, seed=quantization_factor)

    assert encoder.encode_document_vectors(batch) == [
        encoder.encode_document_vector_legacy(row) for row in batch
    ]


def test_vectorized_matches_legacy_on_edge_values():
    encoder = _encoder()
    # Values that round to zero, exact halves, an all-zero row, and negatives
    # (which relu never produces but which the legacy filter tolerated).
    rows = [
        [0.0] * VOCAB_SIZE,
        [0.0001] * VOCAB_SIZE,
        [0.005] * VOCAB_SIZE,
        [0.015] * VOCAB_SIZE,
        [-1.0] * VOCAB_SIZE,
    ]
    for index in range(VOCAB_SIZE):
        rows[3][index] = index / 100.0
    batch = torch.tensor(rows, dtype=torch.float32)

    assert encoder.encode_document_vectors(batch) == [
        encoder.encode_document_vector_legacy(row) for row in batch
    ]


def test_single_vector_api_is_preserved():
    encoder = _encoder()
    vector = _random_batch(1, torch.float32, seed=5)[0]

    assert encoder.encode_document_vector(vector) == encoder.encode_document_vector_legacy(vector)


def test_token_order_is_ascending_by_token_id():
    encoder = _encoder()
    batch = _random_batch(4, torch.float32, seed=11)

    for weights in encoder.encode_document_vectors(batch):
        token_ids = [int(token.removeprefix("tok")) for token in weights]
        assert token_ids == sorted(token_ids)


def test_write_batch_output_is_unchanged(monkeypatch):
    import search.splade as splade_module

    encoder = _encoder()
    index = SpladeIndex.__new__(SpladeIndex)
    index.encoder = encoder
    documents = [IndexInput(f"doc{i}", f"text {i}") for i in range(8)]
    batch = _random_batch(len(documents), torch.float16, seed=23)
    monkeypatch.setattr(encoder, "encode_batch", lambda texts: batch)

    def write(legacy: bool) -> str:
        monkeypatch.setattr(splade_module, "LEGACY_WRITE_BATCH", legacy)
        buffer = io.StringIO()
        index._write_batch(documents, buffer)
        return buffer.getvalue()

    output = write(False)
    assert output == write(True)
    records = [json.loads(line) for line in output.splitlines()]
    assert [record["id"] for record in records] == [doc.id for doc in documents]
    assert all(record["content"] == "" for record in records)


def test_write_batch_handles_single_document(monkeypatch):
    import search.splade as splade_module

    encoder = _encoder()
    index = SpladeIndex.__new__(SpladeIndex)
    index.encoder = encoder
    documents = [IndexInput("doc0", "text")]
    # encode_batch squeezes a one-element batch down to a single vector.
    vector = _random_batch(1, torch.float16, seed=29)[0]
    monkeypatch.setattr(encoder, "encode_batch", lambda texts: vector)
    monkeypatch.setattr(splade_module, "LEGACY_WRITE_BATCH", False)

    buffer = io.StringIO()
    index._write_batch(documents, buffer)
    record = json.loads(buffer.getvalue())

    assert record["id"] == "doc0"
    assert record["vector"] == encoder.encode_document_vector_legacy(vector)


def test_batch_transfer_happens_once(monkeypatch):
    encoder = _encoder()
    batch = _random_batch(32, torch.float32, seed=31)
    transfers = {"count": 0}

    class CountingTensor:
        def __init__(self, tensor):
            self._tensor = tensor

        def cpu(self):
            transfers["count"] += 1
            return self

        def numpy(self):
            return self._tensor.numpy()

        @property
        def ndim(self):
            return self._tensor.ndim

    encoder.encode_document_vectors(CountingTensor(batch))

    assert transfers["count"] == 1


def test_quantization_stays_in_half_precision():
    """Upcasting to float32 before rounding would change borderline values."""
    encoder = _encoder()
    values = torch.tensor([[0.125, 0.135, 0.145, 0.155]], dtype=torch.float16)
    padded = torch.zeros((1, VOCAB_SIZE), dtype=torch.float16)
    padded[0, : values.shape[1]] = values[0]

    weights = encoder.encode_document_vectors(padded)[0]
    expected = np.rint(padded.numpy()[0] * 100).astype(int)

    assert [weights[f"tok{i}"] for i in range(values.shape[1])] == [
        int(expected[i]) for i in range(values.shape[1])
    ]
