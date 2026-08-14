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
from search.splade_artifacts import (
    REBUILD,
    RESUME,
    REUSE,
    PassagesManifest,
    PassagesWriter,
    PhaseADecision,
    ResumeMismatch,
    SampledDocument,
    decide_phase_a,
    passages_manifest_path,
    read_passages_manifest,
    prepare_resume,
    skip_documents,
)
from utils.artifacts import (
    atomic_write_json,
    cached_line_count,
    corpus_signature,
    model_fingerprint,
    remove_stale_dirs,
    temporary_dir_name,
    trust_legacy_artifacts,
    write_index_manifest,
)

# Encoding a corpus needs the GPU and building a Lucene index needs the CPU, so
# the two phases of different tasks can run at the same time. The pool is shared
# by every index in the process: it caps how many Lucene indexes are built
# simultaneously, no matter how many indexes are in flight.
_BUILD_LOCK = threading.Lock()
_BUILD_EXECUTOR: Optional[ThreadPoolExecutor] = None
_ACTIVE_BUILDS: Dict[str, Future] = {}


def _top_token_overlap(stored: Dict, fresh: Dict, top_k: int) -> float:
    """Share of the strongest tokens of two splade vectors that agree."""
    def strongest(vector: Dict):
        return {token for token, _weight in sorted(vector.items(), key=lambda item: -item[1])[:top_k]}

    left, right = strongest(stored), strongest(fresh)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


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

    def encode_document_vector(self, splade_vector: torch.Tensor):
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

    def __init__(self, config: Dict, data_dir: str, use_bettertransformer: bool = False,
                 task_id: Optional[str] = None, corpus_path: Optional[str] = None):
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
        self.docs_dir = os.path.join(self.base_dir, "docs")
        self.docs_path = os.path.join(self.docs_dir, "passages.jsonl")
        # Kept outside docs/: LuceneBackend.build() hands Anserini the whole
        # directory as -input, so any *.json left there ends up in the index.
        self.manifest_path = passages_manifest_path(self.base_dir)
        self.task_id = task_id
        self.corpus_path = corpus_path
        self.use_bettertransformer = use_bettertransformer
        self.quantization_factor = config.get("quantization_factor", 100)
        self.max_seq_length = config.get("max_seq_length", 512)
        # How often the encoding manifest is rewritten, in batches. More often
        # means less re-encoding after a kill and more small atomic writes.
        self.manifest_flush_batches = max(1, int(config.get("manifest_flush_batches", 100)))
        # Documents re-encoded to identify the model behind an encoded corpus that
        # has no manifest. A few dozen cost seconds and save hours.
        self.legacy_sample_size = max(2, int(config.get("legacy_sample_size", 64)))
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
            # noinspection PyTypeChecker
            backend["expected_doc_count_provider"] = self._expected_doc_count
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
        # An index that is still being written, or was left half-written by a
        # killed job, is not usable yet.
        self.wait_for_build()
        return self.backend.is_complete()

    def build(self, docs: Iterable[IndexInput]):
        # Another index object may be writing into the same directory when
        # several tasks share an index cache; never remove it underneath them.
        with _BUILD_LOCK:
            concurrent_build = _ACTIVE_BUILDS.get(self.index_dir)
        if concurrent_build is not None:
            concurrent_build.result()
        self.wait_for_build()
        os.makedirs(self.docs_dir, exist_ok=True)
        remove_stale_dirs(self.index_dir)
        logging.info("Building splade vectors index %s", self.index_dir)
        decision = self._decide_phase_a()
        logging.info(
            "Splade encoding for %s: %s (%s), %d document(s) already encoded",
            self.index_dir, decision.action.upper(), decision.reason, decision.skip_docs,
        )
        if decision.action == REUSE:
            doc_count = decision.manifest.written_doc_count
            self._store_manifest(decision.manifest)
        else:
            doc_count = self._encode_corpus(docs, decision)
        if self.index_build_workers <= 0:
            self._build_index(self.docs_path, doc_count)
            return
        self._pending_build = self._submit_build(self.docs_path, doc_count)

    def _current_manifest(self) -> PassagesManifest:
        """The manifest that describes what this run would encode."""
        corpus_size, corpus_mtime = corpus_signature(self.corpus_path)
        return PassagesManifest(
            task_id=self.task_id,
            model_path=self.model_name_or_path,
            model_fingerprint=model_fingerprint(self.model_name_or_path),
            max_seq_length=self.max_seq_length,
            quantization_factor=self.quantization_factor,
            corpus_path=self.corpus_path,
            corpus_size_bytes=corpus_size,
            corpus_mtime=corpus_mtime,
            # Never counted here: phase A streams the whole corpus anyway, so
            # the count is recorded once it finishes.
            corpus_doc_count=cached_line_count(self.corpus_path, compute=False) if self.corpus_path else None,
        )

    def _decide_phase_a(self) -> PhaseADecision:
        expected = self._current_manifest()
        if self.corpus_path is None:
            # Without the corpus behind the stream nothing can be verified.
            return PhaseADecision(REBUILD, "the corpus path of this task is unknown", expected)
        decision = decide_phase_a(self.manifest_path, self.docs_path, expected,
                                  trust_legacy=trust_legacy_artifacts(), sample_size=self.legacy_sample_size)
        if not decision.samples:
            return decision
        ok, detail = self._sample_matches_current_model(decision.samples)
        logging.info("Verified %d sampled document(s) of %s: %s", len(decision.samples), self.docs_path, detail)
        if ok:
            return decision
        return PhaseADecision(REBUILD, f"the encoded corpus was written by another model: {detail}", expected)

    def _sample_matches_current_model(self, samples: List[SampledDocument], top_k: int = 20,
                                      mean_overlap: float = 0.9, min_overlap: float = 0.6):
        """Re-encode a few documents and compare them with what is on disk.

        The only thing an encoded corpus without a manifest cannot tell us is
        which model produced it. Encoding a handful of its documents answers that
        in seconds, which is what makes an interrupted msmarco-pl run resumable
        instead of an eight hour re-run.

        Vectors are compared by their strongest tokens rather than exactly:
        batches pad to their longest member, so the same model does not
        necessarily reproduce identical weights document by document.
        """
        if self.encoder is None: self._get_encoder()
        overlaps = []
        with torch.no_grad():
            for start in range(0, len(samples), self.index_batch_size):
                batch = samples[start:start + self.index_batch_size]
                encoded = self.encoder.encode_batch([sample.text for sample in batch])
                if len(batch) == 1:
                    encoded = encoded.unsqueeze(0)
                for sample, vector in zip(batch, encoded):
                    fresh = self.encoder.encode_document_vector(vector)
                    overlaps.append(_top_token_overlap(sample.vector, fresh, top_k))
        if not overlaps:
            return False, "nothing could be re-encoded"
        mean = sum(overlaps) / len(overlaps)
        worst = min(overlaps)
        detail = f"mean top-{top_k} overlap {mean:.2f}, worst {worst:.2f}"
        return mean >= mean_overlap and worst >= min_overlap, detail

    def _expected_doc_count(self) -> Optional[int]:
        """How many documents a complete index of this task should hold.

        Only used to validate indexes built before manifests existed, so reading
        the corpus once (cached next to it) is acceptable here.
        """
        manifest = read_passages_manifest(self.manifest_path)
        if manifest is not None and manifest.complete:
            return manifest.written_doc_count
        if not self.corpus_path:
            return None
        return cached_line_count(self.corpus_path)

    def _store_manifest(self, manifest: PassagesManifest):
        atomic_write_json(self.manifest_path, manifest.to_dict())

    def _encode_corpus(self, docs: Iterable[IndexInput], decision: PhaseADecision) -> int:
        """Run phase A, appending to a partial file when resuming."""
        manifest, append, skipped = decision.manifest, False, 0
        if decision.action == RESUME:
            ok, reason, manifest = prepare_resume(self.docs_path, decision.manifest)
            if ok:
                try:
                    docs = skip_documents(docs, manifest.written_doc_count, manifest.last_doc_id)
                    append, skipped = True, manifest.written_doc_count
                    logging.info("Skipped %d already encoded document(s): %s", skipped, reason)
                except ResumeMismatch as error:
                    logging.warning("Cannot resume the splade encoding, rebuilding: %s", error)
            else:
                logging.warning("Cannot resume the splade encoding, rebuilding: %s", reason)
        if not append:
            manifest = self._current_manifest()
        if self.encoder is None: self._get_encoder()
        with PassagesWriter(self.docs_path, self.manifest_path, manifest,
                            flush_batches=self.manifest_flush_batches, append=append) as writer:
            with torch.no_grad():
                batch = []
                for doc in docs:
                    batch.append(doc)
                    if len(batch) >= self.index_batch_size:
                        self._write_batch(batch, writer.file)
                        writer.note_batch(len(batch), batch[-1].id)
                        batch = []
                if len(batch) > 0:
                    self._write_batch(batch, writer.file)
                    writer.note_batch(len(batch), batch[-1].id)
            final = writer.finalize()
        return final.written_doc_count

    def _build_index(self, docs_path: str, doc_count: Optional[int]):
        """Build the Lucene index aside and swap it in only once it is complete.

        The old index stays queryable until the new one is ready, and a killed
        job leaves a `.tmp-*` directory that nothing mistakes for the index.
        """
        target_dir = temporary_dir_name(self.index_dir)
        moved_aside = None
        try:
            os.makedirs(target_dir, exist_ok=True)
            self.backend.build(docs_path, target_dir=target_dir)
            write_index_manifest(target_dir, doc_count)
            if os.path.exists(self.index_dir):
                moved_aside = temporary_dir_name(self.index_dir, kind="old")
                shutil.rmtree(moved_aside, ignore_errors=True)
                os.rename(self.index_dir, moved_aside)
            os.rename(target_dir, self.index_dir)
        except BaseException:
            shutil.rmtree(target_dir, ignore_errors=True)
            if moved_aside is not None and not os.path.exists(self.index_dir):
                os.rename(moved_aside, self.index_dir)
                moved_aside = None
            raise
        finally:
            if moved_aside is not None:
                shutil.rmtree(moved_aside, ignore_errors=True)

    def _submit_build(self, docs_path: str, doc_count: Optional[int] = None) -> Future:
        index_dir = self.index_dir
        executor = _get_build_executor(self.index_build_workers)
        future = executor.submit(self._build_index, docs_path, doc_count)
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
        for doc, splade_vector in zip(batch, splade_batch):
            splade_dict = self.encoder.encode_document_vector(splade_vector)
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
