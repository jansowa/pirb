import atexit
import json
import logging
import os.path
import shutil
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Iterable, Dict, Optional, TextIO
import numpy as np
import torch
from pyserini.encode import QueryEncoder
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, BatchEncoding
from backend import IndexBackend, SparseBackend
from data import IndexInput
from search.base import SearchIndex

# Temporary A/B switch used by the perf harness in perf_scratch/. Read in this
# single place so it can be deleted together with the harness.
LEGACY_WRITE_BATCH = os.environ.get("PIRB_LEGACY_WRITE_BATCH", "0") == "1"

# Encoding a corpus needs the GPU and building a Lucene index needs the CPU, so
# the two phases of different tasks can run at the same time. The pool is shared
# by every index in the process: it caps how many Lucene indexes are built
# simultaneously, no matter how many indexes are in flight.
_BUILD_LOCK = threading.Lock()
_BUILD_EXECUTOR: Optional[ThreadPoolExecutor] = None
_ACTIVE_BUILDS: Dict[str, Future] = {}


def _get_build_executor(max_workers: int) -> ThreadPoolExecutor:
    global _BUILD_EXECUTOR
    with _BUILD_LOCK:
        if _BUILD_EXECUTOR is None:
            _BUILD_EXECUTOR = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="lucene-build")
            # A shared pool cannot be closed by whichever index happens to be
            # collected first, so it is drained once at interpreter shutdown.
            atexit.register(shutdown_build_executor)
        return _BUILD_EXECUTOR


def shutdown_build_executor() -> None:
    """Wait for pending index builds and drop the shared pool."""
    global _BUILD_EXECUTOR
    with _BUILD_LOCK:
        executor, _BUILD_EXECUTOR = _BUILD_EXECUTOR, None
    if executor is not None:
        executor.shutdown(wait=True)


class SpladeEncoder(QueryEncoder):

    def __init__(self, config: Dict, use_bettertransformer: bool, quantization_factor: int = 100):
        self.quantization_factor = quantization_factor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.use_bettertransformer = use_bettertransformer
        self.model_name_or_path = config["name"]
        self.maxlen = config.get("max_seq_length", 512)
        self.model_kwargs = config.get("model_kwargs", {})
        self.tokenizer, self.model = self._init_tokenizer_and_model()
        self.vocab = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.queries_cache = {}

    def _init_tokenizer_and_model(self):
        torch_dtype = torch.float32
        if self.config.get("fp16", False):
            torch_dtype = torch.float16
        elif self.config.get("bf16", False):
            torch_dtype = torch.bfloat16
        model_kwargs = {"torch_dtype": torch_dtype}
        model_kwargs.update(self.model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        model = AutoModelForMaskedLM.from_pretrained(self.model_name_or_path, **model_kwargs).to(self.device)
        tokenizer.model_max_length = self.maxlen
        if self.use_bettertransformer:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        model.eval()
        return tokenizer, model

    def encode_batch(self, batch: List[str]):
        batch: BatchEncoding = self.tokenizer(
            batch, padding="longest", truncation=True, add_special_tokens=True,
            return_tensors="pt", max_length=self.tokenizer.model_max_length
        ).to(self.device)
        output = self.model(**batch)
        logits, attention_mask = output["logits"].detach(), batch["attention_mask"].detach()
        attention_mask = attention_mask.unsqueeze(-1)
        _relu = torch.relu(logits)
        splade_vectors = torch.max(torch.log(torch.add(_relu, 1)) * attention_mask, dim=1)
        input_ids = batch["input_ids"]
        input_ids.detach()
        del batch, logits, attention_mask, input_ids, _relu
        result = splade_vectors[0].detach().squeeze()
        splade_vectors[1].detach()
        del splade_vectors
        return result

    def encode_document_vectors(self, splade_vectors: torch.Tensor) -> List[Dict[str, int]]:
        """Quantize a batch of splade vectors into token -> weight dictionaries.

        The whole batch is copied off the device once, and quantization runs on
        the non-zero entries of the batch in one go - splade vectors are sparse,
        so quantizing the full vocabulary would cost far more than it saves. The
        multiplication deliberately stays in the tensor dtype so half-precision
        vectors round exactly as they did when every document was converted on
        its own.
        """
        data = splade_vectors.cpu().numpy()
        if data.ndim == 1:
            data = data.reshape(1, -1)
        rows, cols = np.nonzero(data)
        values = np.rint(data[rows, cols] * self.quantization_factor).astype(int)
        # np.nonzero returns indices in row-major order, so each document is a
        # contiguous slice.
        bounds = np.searchsorted(rows, np.arange(data.shape[0] + 1))
        vocab = self.vocab
        return [
            {
                vocab[int(token_id)]: int(value)
                for token_id, value in zip(cols[start:end], values[start:end])
                if value > 0
            }
            for start, end in zip(bounds[:-1], bounds[1:])
        ]

    def encode_document_vector(self, splade_vector: torch.Tensor) -> Dict[str, int]:
        return self.encode_document_vectors(splade_vector)[0]

    def encode_document_vector_legacy(self, splade_vector: torch.Tensor):
        """Pre-vectorization implementation, kept for the perf harness only."""
        data = splade_vector.cpu().numpy()
        idx = np.nonzero(data)
        data = data[idx]
        data = np.rint(data * self.quantization_factor).astype(int)
        dict_splade = dict()
        for id_token, value_token in zip(idx[0], data):
            if value_token > 0:
                real_token = self.vocab[id_token]
                dict_splade[real_token] = int(value_token)
        return dict_splade

    def encode_queries(self, queries: List[str], batch_size: int, query_ids: List[str] = None):
        if query_ids is None:
            query_ids = queries
        cached = {}
        results = {}
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            ids_batch = query_ids[i:i + batch_size]
            encoded = self.encode_batch(batch).cpu().numpy()
            outputs = self._output_to_weight_dicts(encoded)
            assert len(batch) == len(outputs)
            cached.update({text: output for text, output in zip(batch, outputs)})
            results.update({qid: output for qid, output in zip(ids_batch, outputs)})
        self.queries_cache = cached
        return results

    def _output_to_weight_dicts(self, batch_aggregated_logits: torch.Tensor, max_clauses: int = 1024):
        to_return = []
        if len(batch_aggregated_logits.shape) == 1:
            batch_aggregated_logits = [batch_aggregated_logits]
        for aggregated_logits in batch_aggregated_logits:
            col = np.nonzero(aggregated_logits)[0]
            weights = aggregated_logits[col]
            outputs = [(self.vocab[k], float(v)) for k, v in zip(list(col), list(weights))]
            outputs = sorted(outputs, key=lambda v: -v[1])[:max_clauses]
            d = {k: v for k, v in outputs}
            to_return.append(d)
        return to_return

    def encode(self, text, max_length=512, **kwargs):
        return self.queries_cache[text]


class SpladeIndex(SearchIndex):

    def __init__(self, config: Dict, data_dir: str, use_bettertransformer: bool = False):
        # Queries are a handful of tokens while passages are encoded at
        # max_seq_length, so the corpus batch size can be tuned separately.
        self.batch_size = int(config.get("batch_size", 32))
        self.index_batch_size = int(config.get("index_batch_size", self.batch_size))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.data_dir = data_dir
        self.model_name_or_path = config["name"]
        self.index_name = self.model_name_or_path.replace("/", "_").replace(".", "_")
        self.base_dir = os.path.join(self.data_dir, self.index_name)
        self.index_dir = os.path.join(self.base_dir, f"{self._get_backend_name(config)}_index")
        self.use_bettertransformer = use_bettertransformer
        self.quantization_factor = config.get("quantization_factor", 100)
        self.threads = config.get("threads", 8)
        # Number of Lucene indexes that may be built in the background at once.
        # 0 keeps the index build inline, exactly as before.
        self.index_build_workers = max(0, int(config.get("index_build_workers", 1)))
        self.backend = self._create_backend(config)
        self.encoder: Optional[SpladeEncoder] = None
        self.searcher = None
        self._pending_build: Optional[Future] = None

    def _get_backend_name(self, config: Dict):
        backend = config.get("backend", "lucene")
        # noinspection PyTypeChecker
        return backend if isinstance(backend, str) else backend["backend_type"]

    def _create_backend(self, config: Dict) -> SparseBackend:
        backend = config.get("backend", "lucene")
        if isinstance(backend, str):
            backend = {"backend_type": backend, "index_dir": self.index_dir}
        elif isinstance(backend, dict):
            backend["index_dir"] = self.index_dir
        else:
            raise AssertionError("'backend' arg accepts backend name or backend config as dict")
        # Without this the thread count from the model config never reached the
        # backend, which silently kept its own default.
        backend.setdefault("threads", self.threads)
        if backend["backend_type"] == "lucene":
            # noinspection PyTypeChecker
            backend["encoder_provider"] = self._get_encoder
        return IndexBackend.from_config(backend)

    def _get_encoder(self):
        if self.encoder is None:
            self.encoder = SpladeEncoder(self.config, self.use_bettertransformer, self.quantization_factor)
        return self.encoder

    def supports_deferred_build(self) -> bool:
        """Whether build() returns before the index is queryable."""
        return self.index_build_workers > 0

    def wait_for_build(self):
        """Block until a backgrounded index build finishes, re-raising its error."""
        pending, self._pending_build = self._pending_build, None
        if pending is not None:
            pending.result()

    def exists(self) -> bool:
        # An index that is still being written is not usable yet.
        self.wait_for_build()
        return self.backend.exists()

    def build(self, docs: Iterable[IndexInput]):
        # Another index object may be writing into the same directory when
        # several tasks share an index cache; never remove it underneath them.
        with _BUILD_LOCK:
            concurrent_build = _ACTIVE_BUILDS.get(self.index_dir)
        if concurrent_build is not None:
            concurrent_build.result()
        if self.exists(): shutil.rmtree(self.index_dir)
        if self.encoder is None: self._get_encoder()
        docs_dir = os.path.join(self.base_dir, "docs")
        docs_path = os.path.join(docs_dir, "passages.jsonl")
        os.makedirs(docs_dir, exist_ok=True)
        logging.info("Building splade vectors index %s", self.index_dir)
        with open(docs_path, "w", encoding='utf-8') as docs_file:
            with torch.no_grad():
                batch = []
                for doc in docs:
                    batch.append(doc)
                    if len(batch) >= self.index_batch_size:
                        self._write_batch(batch, docs_file)
                        batch = []
                if len(batch) > 0:
                    self._write_batch(batch, docs_file)
        os.makedirs(self.index_dir, exist_ok=True)
        if self.index_build_workers <= 0:
            self.backend.build(docs_path)
            return
        self._pending_build = self._submit_build(docs_path)

    def _submit_build(self, docs_path: str) -> Future:
        index_dir = self.index_dir
        executor = _get_build_executor(self.index_build_workers)
        future = executor.submit(self.backend.build, docs_path)
        with _BUILD_LOCK:
            _ACTIVE_BUILDS[index_dir] = future

        def release(_future: Future):
            with _BUILD_LOCK:
                if _ACTIVE_BUILDS.get(index_dir) is _future:
                    del _ACTIVE_BUILDS[index_dir]

        future.add_done_callback(release)
        return future

    def _write_batch(self, batch: List[IndexInput], output: TextIO):
        texts = [val.text for val in batch]
        splade_batch = self.encoder.encode_batch(texts)
        if len(texts) == 1:
            splade_batch = splade_batch.unsqueeze(0)
        if LEGACY_WRITE_BATCH:
            splade_dicts = [self.encoder.encode_document_vector_legacy(vector) for vector in splade_batch]
        else:
            splade_dicts = self.encoder.encode_document_vectors(splade_batch)
        for doc, splade_dict in zip(batch, splade_dicts):
            dict_ = dict(id=doc.id, content="", vector=splade_dict)
            json_dict = json.dumps(dict_, ensure_ascii=False)
            output.write(json_dict + "\n")

    def search(self, queries: List[IndexInput], k: int, batch_size=1024, verbose=True, cache_prefix=None, overwrite=False) -> Dict:
        results = self.load_results_from_cache(k, cache_prefix) if not overwrite else None
        if results is not None: return results
        # No query may hit a half-written index.
        self.wait_for_build()
        if self.encoder is None: self._get_encoder()
        results = {}
        for i in tqdm(range(0, len(queries), batch_size), desc="Searching in splade index", disable=not verbose):
            batch: List[IndexInput] = queries[i:i + batch_size]
            texts = [val.text for val in batch]
            ids = [val.id for val in batch]
            with torch.no_grad():
                encoded_texts = self.encoder.encode_queries(texts, batch_size=self.batch_size, query_ids=ids)
            batch_results = self.backend.search(texts, encoded_texts, ids, top_k=k)
            results.update(batch_results)
        self.save_results_to_cache(k, cache_prefix, results)
        return results

    def name(self):
        return f"splade_{self.index_name}"

    def __len__(self):
        return 0

    def model_dict(self) -> Dict:
        return self.config

    def index_path(self) -> str:
        return self.index_dir

    def close(self):
        # Never drop the backend while a build thread is still using it.
        if hasattr(self, "_pending_build"):
            self.wait_for_build()
        if getattr(self, "backend", None) is not None:
            self.backend.close()

    def __del__(self):
        self.close()
