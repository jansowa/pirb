import logging
import os
from typing import List, Dict, Callable, Optional
from pyserini.search import LuceneImpactSearcher
from pyserini.encode import QueryEncoder
from backend.base import SparseBackend
from data import IndexResult
from utils.artifacts import has_lucene_segments, read_index_manifest, write_index_manifest
from utils.lucene import lucene_num_docs


class LuceneBackend(SparseBackend):

    def __init__(self, index_dir: str, encoder_provider: Callable[[], QueryEncoder], threads: int = 8,
                 expected_doc_count_provider: Optional[Callable[[], Optional[int]]] = None):
        super().__init__(index_dir)
        self.threads = threads
        self.encoder_provider = encoder_provider
        # Only consulted for indexes predating the manifest, so that they are not
        # invalidated wholesale; it may read the corpus, hence the indirection.
        self.expected_doc_count_provider = expected_doc_count_provider
        self.encoder = None
        self.searcher = None

    def build(self, corpus_path: str, target_dir: Optional[str] = None):
        from jnius import autoclass
        docs_dir = os.path.dirname(corpus_path)
        args = [
            "-collection", "JsonVectorCollection",
            "-input", os.path.abspath(docs_dir),
            # Written aside and renamed into place by the caller, so that a
            # killed build never leaves something that looks like an index.
            "-index", os.path.abspath(target_dir or self.index_dir),
            "-threads", str(self.threads),
            "-generator", "DefaultLuceneDocumentGenerator",
            "-impact",
            "-pretokenized",
            "-storeDocvectors"
        ]
        JIndexCollection = autoclass('io.anserini.index.IndexCollection')
        JIndexCollection.main(args)

    def search(self, q: List[str], eq: Dict, ids: List[str], top_k: int) -> Dict[str, List[IndexResult]]:
        if self.searcher is None:
            self.open()
        batch_results = self.searcher.batch_search(q, ids, k=top_k, threads=self.threads)
        for qid, relevant in batch_results.items():
            batch_results[qid] = [IndexResult(val.docid, val.score) for val in relevant]
        return batch_results

    def open(self):
        self.encoder = self.encoder_provider()
        self.searcher = LuceneImpactSearcher(self.index_dir, self.encoder)

    def exists(self) -> bool:
        return os.path.exists(self.index_dir)

    def is_complete(self) -> bool:
        """Whether the index holds every document it was supposed to hold.

        A marker file alone would not catch an index killed after committing only
        some of its segments: that one opens cleanly and quietly returns worse
        results, so the document count is compared as well.
        """
        if not os.path.isdir(self.index_dir) or not has_lucene_segments(self.index_dir):
            return False
        manifest = read_index_manifest(self.index_dir)
        expected = manifest.get("doc_count") if manifest else self._expected_doc_count()
        num_docs = self.num_docs()
        if num_docs is None:
            logging.info("Index %s cannot be opened, treating it as incomplete", self.index_dir)
            return False
        if expected is not None and num_docs != expected:
            logging.info("Index %s holds %d of %d documents", self.index_dir, num_docs, expected)
            return False
        if manifest is None:
            # An index from before manifests existed, verified just now.
            logging.info("Adding a manifest to the existing index %s (%d documents)", self.index_dir, num_docs)
            try:
                write_index_manifest(self.index_dir, num_docs)
            except OSError as error:
                logging.debug("Could not write the manifest of %s: %s", self.index_dir, error)
        return True

    def _expected_doc_count(self) -> Optional[int]:
        if self.expected_doc_count_provider is None:
            return None
        try:
            return self.expected_doc_count_provider()
        except Exception as error:
            logging.debug("Could not determine the expected document count: %s", error)
            return None

    def num_docs(self, index_dir: Optional[str] = None) -> Optional[int]:
        """Documents in an index on disk, or None if it cannot be read.

        Takes a directory so that a build staged aside can be inspected before it
        is renamed into place.
        """
        return lucene_num_docs(index_dir or self.index_dir)
