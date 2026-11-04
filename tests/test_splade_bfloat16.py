"""Exercise real tensor conversion without loading a model or using a GPU."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from search.splade import SpladeEncoder, _numpy_compatible


def encoder():
    instance = object.__new__(SpladeEncoder)
    instance.vocab = {0: 'zero', 1: 'one', 2: 'two'}
    instance.vocab_size = 3
    instance.quantization_factor = 100
    instance.quantize_queries = True
    return instance


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.bfloat16])
def test_document_and_query_quantization_supports_bfloat16(dtype):
    model = encoder()
    vector = torch.tensor([0, 0.25, 1.5, 2], dtype=dtype)
    assert model.encode_document_vector(vector) == {'one': 25, 'two': 150}
    model.encode_batch = lambda texts: vector.unsqueeze(0).repeat(len(texts), 1)
    assert model.encode_queries(['query'], 1, ['qid']) == {'qid': {'two': 150, 'one': 25}}


def test_fp16_conversion_preserves_original_dtype_and_rounding():
    vector = torch.tensor([0.0049, 0.0051, 1.234], dtype=torch.float16)
    output = _numpy_compatible(vector)
    assert output.dtype == np.float16
    np.testing.assert_array_equal(np.rint(output * 100), np.rint(vector.numpy() * 100))
