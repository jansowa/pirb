"""A splade index whose encoder and Lucene backend are cheap fakes.

Shared by the in-process tests and by the worker that gets SIGKILLed, so both
observe exactly the same build. The fake encoder records how many documents it
encoded - the only reliable way to prove that a resumed run did not redo work.
"""

import json
import os
from typing import List, Optional

ENCODER_LOG = "encoder_calls.log"
INDEX_DOC_COUNT = "doc_count.txt"


class FakeVectors(list):
    """Stands in for the tensor batch returned by SpladeEncoder.encode_batch."""

    def unsqueeze(self, _dim):
        return self


class FakeEncoder:
    """Counts encoded documents and can kill the process mid-corpus.

    ``model_tag`` stands in for the identity of a checkpoint: vectors are keyed by
    it, so encoding the same text with a different tag produces a disjoint vector,
    exactly as a different model would.
    """

    def __init__(self, log_path: str, kill_after: Optional[int] = None, model_tag: str = "m1"):
        self.log_path = log_path
        self.kill_after = kill_after
        self.model_tag = model_tag
        self.encoded = 0

    def encode_batch(self, texts: List[str]) -> FakeVectors:
        self.encoded += len(texts)
        with open(self.log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"{len(texts)}\n")
            log_file.flush()
            os.fsync(log_file.fileno())
        if self.kill_after is not None and self.encoded >= self.kill_after:
            os.kill(os.getpid(), 9)
        return FakeVectors(f"vector-{text}" for text in texts)

    def encode_document_vector(self, vector) -> dict:
        text = str(vector).removeprefix("vector-")
        tokens = {f"{self.model_tag}_{word}": len(word) + 1 for word in text.split()}
        return tokens or {f"{self.model_tag}_empty": 1}


def encoded_documents(log_path: str) -> int:
    """How many documents the fake encoder processed, across all runs."""
    if not os.path.exists(log_path):
        return 0
    with open(log_path, "r", encoding="utf-8") as log_file:
        return sum(int(line) for line in log_file if line.strip())


def write_corpus(path: str, count: int, prefix: str = "doc") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as corpus_file:
        for i in range(count):
            corpus_file.write(json.dumps({"id": f"{prefix}_{i}", "contents": f"passage number {i}"}) + "\n")
    return path


def read_passages(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as passages_file:
        return [json.loads(line) for line in passages_file if line.strip()]


def fake_backend_build(index, kill_in_phase_b: bool = False, drop_docs: int = 0):
    """A backend.build that writes a Lucene-looking directory, not a real index.

    ``drop_docs`` simulates an index committed with fewer documents than the
    corpus holds; ``kill_in_phase_b`` kills the process once the directory looks
    half-written, exactly where Anserini would be interrupted.
    """

    def build(corpus_path: str, target_dir: Optional[str] = None):
        directory = target_dir or index.index_dir
        os.makedirs(directory, exist_ok=True)
        documents = len(read_passages(corpus_path)) - drop_docs
        with open(os.path.join(directory, "segments_1"), "w", encoding="utf-8") as segments:
            segments.write("fake lucene segments\n")
        if kill_in_phase_b:
            os.kill(os.getpid(), 9)
        with open(os.path.join(directory, INDEX_DOC_COUNT), "w", encoding="utf-8") as counter:
            counter.write(str(documents))

    return build


def fake_num_docs(index_dir: str) -> Optional[int]:
    """What a Lucene reader would report for a directory built by the fake."""
    counter_path = os.path.join(index_dir, INDEX_DOC_COUNT)
    if not os.path.exists(counter_path):
        return None
    with open(counter_path, "r", encoding="utf-8") as counter:
        return int(counter.read().strip())
