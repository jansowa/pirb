"""Manifests and resume decisions for the two phases of a splade index build.

Encoding the corpus into ``docs/passages.jsonl`` costs GPU hours; turning that
file into a Lucene index costs CPU minutes. A job killed mid-benchmark must
therefore be able to (a) reuse a finished ``passages.jsonl`` outright and
(b) resume a partial one instead of encoding it again.

This module holds everything that decision needs and deliberately imports
neither pyserini nor torch, so it can be tested without a GPU or a JVM.
"""

import json
import logging
import os
import re
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Tuple

from utils.artifacts import (
    atomic_write_json,
    read_json_or_none,
    store_line_count,
)

_ID_PATTERN = re.compile(rb'"id"\s*:\s*"([^"]*)"')

PASSAGES_MANIFEST_SCHEMA = 1
# Anserini indexes every file in the directory passed as -input, so the manifest
# lives next to docs/ rather than inside it.
PASSAGES_MANIFEST_NAME = "passages_manifest.json"

REUSE = "reuse"
RESUME = "resume"
REBUILD = "rebuild"

# How much of the file's tail is inspected when validating a resume point.
TAIL_BLOCK_BYTES = 4 * 1024 * 1024


class ResumeMismatch(Exception):
    """The corpus stream does not line up with what the manifest recorded."""


@dataclass
class PassagesManifest:
    """What is known about the encoded corpus in ``docs/passages.jsonl``."""

    task_id: Optional[str] = None
    model_path: Optional[str] = None
    model_fingerprint: Optional[str] = None
    max_seq_length: Optional[int] = None
    quantization_factor: Optional[int] = None
    corpus_path: Optional[str] = None
    corpus_size_bytes: Optional[int] = None
    corpus_mtime: Optional[float] = None
    corpus_doc_count: Optional[int] = None
    written_doc_count: int = 0
    written_bytes: int = 0
    last_doc_id: Optional[str] = None
    complete: bool = False
    schema_version: int = PASSAGES_MANIFEST_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "PassagesManifest":
        known = {key: value.get(key) for key in cls.__dataclass_fields__ if key in value}
        manifest = cls(**known)
        manifest.written_doc_count = int(manifest.written_doc_count or 0)
        manifest.written_bytes = int(manifest.written_bytes or 0)
        manifest.complete = bool(manifest.complete)
        return manifest

    def describes_same_run(self, other: "PassagesManifest") -> Optional[str]:
        """Reason why ``other`` was produced by a different configuration."""
        if self.model_fingerprint is None or other.model_fingerprint is None:
            return "the model directory cannot be fingerprinted"
        checks = (
            ("model fingerprint", self.model_fingerprint, other.model_fingerprint),
            ("max_seq_length", self.max_seq_length, other.max_seq_length),
            ("quantization_factor", self.quantization_factor, other.quantization_factor),
            ("corpus size", self.corpus_size_bytes, other.corpus_size_bytes),
            ("corpus mtime", self.corpus_mtime, other.corpus_mtime),
        )
        for name, expected, actual in checks:
            if expected != actual:
                return f"{name} changed ({actual!r} -> {expected!r})"
        if self.corpus_path and other.corpus_path:
            if os.path.abspath(self.corpus_path) != os.path.abspath(other.corpus_path):
                return f"corpus path changed ({other.corpus_path} -> {self.corpus_path})"
        return None


@dataclass
class PhaseADecision:
    action: str
    reason: str
    manifest: PassagesManifest
    skip_docs: int = 0
    # Documents whose stored vectors must be checked against the current model
    # before the file is trusted. Empty whenever a manifest already vouches for it.
    samples: List["SampledDocument"] = field(default_factory=list)

    @property
    def reuses_encoded_corpus(self) -> bool:
        return self.action == REUSE


def passages_manifest_path(base_dir: str) -> str:
    return os.path.join(base_dir, PASSAGES_MANIFEST_NAME)


def read_passages_manifest(manifest_path: str) -> Optional[PassagesManifest]:
    value = read_json_or_none(manifest_path)
    if value is None:
        return None
    if value.get("schema_version") != PASSAGES_MANIFEST_SCHEMA:
        logging.info("Ignoring %s: unsupported schema_version %r", manifest_path, value.get("schema_version"))
        return None
    return PassagesManifest.from_dict(value)


def decide_phase_a(manifest_path: str, passages_path: str, expected: PassagesManifest,
                   trust_legacy: bool = False, sample_size: int = 64) -> PhaseADecision:
    """Pick REUSE, RESUME or REBUILD for the encoding phase.

    ``expected`` carries the current configuration (fingerprint, sequence length,
    quantization, corpus signature). With a manifest present this is all cheap
    stats; only an encoded corpus without one is walked against the corpus, which
    is the price of recovering an artifact from before manifests existed.
    """
    stored = read_passages_manifest(manifest_path)
    if not os.path.exists(passages_path):
        return PhaseADecision(REBUILD, "no encoded corpus on disk", expected)
    if stored is None:
        return _decide_legacy(passages_path, expected, trust_legacy, sample_size)
    mismatch = expected.describes_same_run(stored)
    if mismatch is not None:
        return PhaseADecision(REBUILD, f"manifest does not match the current run: {mismatch}", expected)
    size = os.path.getsize(passages_path)
    if stored.complete:
        if size == stored.written_bytes:
            return PhaseADecision(REUSE, f"manifest reports {stored.written_doc_count} documents encoded", stored)
        return PhaseADecision(
            REBUILD, f"complete manifest disagrees with the file size ({size} != {stored.written_bytes})", expected
        )
    if stored.written_doc_count <= 0 or stored.written_bytes <= 0:
        return PhaseADecision(REBUILD, "manifest records no usable documents", expected)
    if size < stored.written_bytes:
        return PhaseADecision(
            REBUILD, f"encoded corpus is shorter than the manifest ({size} < {stored.written_bytes})", expected
        )
    return PhaseADecision(
        RESUME, f"manifest reports a partial file with {stored.written_doc_count} documents", stored,
        skip_docs=stored.written_doc_count,
    )


def _decide_legacy(passages_path: str, expected: PassagesManifest, trust_legacy: bool,
                   sample_size: int) -> PhaseADecision:
    """Reconstruct a manifest for an encoded corpus written before manifests existed.

    Interrupted runs left behind partially encoded corpora with nothing recording
    how far they got - for msmarco-pl that is hours of GPU time. The order is
    recoverable even so: passages.jsonl is written in corpus order, so walking
    both files side by side and comparing ids establishes how many documents are
    trustworthy. What the file cannot tell us is which model produced it, which
    is why the caller still has to check a sample of the vectors before appending
    to it.
    """
    if not expected.corpus_path or not os.path.exists(expected.corpus_path):
        return PhaseADecision(REBUILD, "no manifest and no corpus to line the encoded corpus up against", expected)
    adoption = adopt_legacy_passages(passages_path, expected.corpus_path, sample_size=sample_size)
    if not adoption.ok:
        return PhaseADecision(REBUILD, f"cannot adopt the encoded corpus: {adoption.reason}", expected)
    adopted = PassagesManifest(**expected.to_dict())
    adopted.written_doc_count = adoption.written_doc_count
    adopted.written_bytes = adoption.written_bytes
    adopted.last_doc_id = adoption.last_doc_id
    adopted.complete = adoption.complete
    if adoption.complete:
        adopted.corpus_doc_count = adoption.written_doc_count
    action = REUSE if adoption.complete else RESUME
    state = "complete" if adoption.complete else "partial"
    return PhaseADecision(
        action, f"adopted a {state} encoded corpus of {adoption.written_doc_count} documents without a manifest",
        adopted, skip_docs=0 if adoption.complete else adoption.written_doc_count,
        # Nothing on disk says which model wrote those vectors: re-encoding a
        # sample is the only way to find out, and skipping it is opt-in.
        samples=[] if trust_legacy else adoption.samples,
    )


@dataclass
class LegacyAdoption:
    ok: bool
    reason: str
    written_doc_count: int = 0
    written_bytes: int = 0
    last_doc_id: Optional[str] = None
    complete: bool = False
    samples: List["SampledDocument"] = None


@dataclass
class SampledDocument:
    """A document whose stored vector can be compared against a fresh encoding."""

    index: int
    doc_id: str
    text: str
    vector: Dict[str, Any]


def _extract_id(line: bytes, scan_bytes: int = 512) -> Optional[str]:
    """Pull the id out of a JSONL line without parsing the whole document.

    Corpus lines carry the passage text, so parsing every one of nine million of
    them would dominate the cost of lining the two files up.
    """
    match = _ID_PATTERN.search(line[:scan_bytes])
    if match is not None:
        return match.group(1).decode("utf-8", errors="replace")
    try:
        value = json.loads(line.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    doc_id = value.get("id") if isinstance(value, dict) else None
    return str(doc_id) if doc_id is not None else None


def adopt_legacy_passages(passages_path: str, corpus_path: str, sample_size: int = 64,
                          text_field: str = "contents") -> LegacyAdoption:
    """Walk an unlabelled encoded corpus against the corpus it came from.

    Returns how many leading documents line up by id, where the trustworthy part
    of the file ends, and a sample of documents for the caller to verify against
    the current model.
    """
    half = max(1, sample_size // 2)
    head: List[Tuple[int, bytes, bytes]] = []
    tail: Deque[Tuple[int, bytes, bytes]] = deque(maxlen=half)
    count, offset, last_id = 0, 0, None
    with open(passages_path, "rb") as passages_file, open(corpus_path, "rb") as corpus_file:
        while True:
            line = passages_file.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                logging.info("Ignoring a torn last line of %s (%d bytes)", passages_path, len(line))
                break
            doc_id = _extract_id(line)
            if doc_id is None:
                logging.info("Ignoring an unreadable line %d of %s", count + 1, passages_path)
                break
            corpus_line = corpus_file.readline()
            if not corpus_line:
                return LegacyAdoption(False, f"the encoded corpus holds more than the {count} documents of the corpus")
            corpus_id = _extract_id(corpus_line)
            if corpus_id != doc_id:
                return LegacyAdoption(
                    False, f"document {count + 1} is {doc_id!r} but the corpus has {corpus_id!r}"
                )
            if len(head) < half:
                head.append((count, line, corpus_line))
            else:
                tail.append((count, line, corpus_line))
            count += 1
            offset += len(line)
            last_id = doc_id
        complete = not corpus_file.readline()
    if count == 0:
        return LegacyAdoption(False, "no usable document in the encoded corpus")
    samples = _build_samples(head, tail, text_field)
    if not samples:
        return LegacyAdoption(False, "no document could be sampled for verification")
    return LegacyAdoption(True, "lined up against the corpus", count, offset, last_id, complete, samples)


def _build_samples(head, tail, text_field: str) -> List[SampledDocument]:
    """Parse the sampled lines, preferring long documents.

    A different max_seq_length only shows up on documents long enough to be
    truncated, so the longest samples are the informative ones.
    """
    samples = []
    for index, passages_line, corpus_line in list(head) + list(tail):
        document = _parse_document(passages_line)
        if document is None:
            continue
        try:
            corpus_document = json.loads(corpus_line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        text = corpus_document.get(text_field) if isinstance(corpus_document, dict) else None
        if not isinstance(text, str):
            continue
        samples.append(SampledDocument(index, str(document["id"]), text, document["vector"]))
    samples.sort(key=lambda sample: len(sample.text), reverse=True)
    return samples


def _last_document_id(passages_path: str) -> Optional[str]:
    for line, _end in reversed(_tail_lines(passages_path)):
        document = _parse_document(line)
        if document is not None:
            return document.get("id")
    return None


def _parse_document(line: bytes) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(line.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(value, dict) or "id" not in value or "vector" not in value:
        return None
    return value


def _tail_lines(path: str, block_bytes: int = TAIL_BLOCK_BYTES) -> List[Tuple[bytes, int]]:
    """Complete lines at the end of the file, with the offset just past each."""
    size = os.path.getsize(path)
    start = max(0, size - block_bytes)
    with open(path, "rb") as input_file:
        input_file.seek(start)
        data = input_file.read()
    lines: List[Tuple[bytes, int]] = []
    offset = start
    for chunk in data.split(b"\n")[:-1]:
        offset += len(chunk) + 1
        lines.append((chunk, offset))
    if start > 0 and lines:
        # The first line in the block may have started before it.
        lines = lines[1:]
    return lines


def prepare_resume(passages_path: str, manifest: PassagesManifest,
                   validate_lines: int = 3) -> Tuple[bool, str, PassagesManifest]:
    """Cut the file back to the manifest and verify the documents at its end.

    Returns ``(ok, reason, manifest)``; when ``ok`` is False the caller must fall
    back to a full rebuild. The returned manifest may describe fewer documents
    than the stored one if the tail turned out to be unusable.
    """
    size = os.path.getsize(passages_path)
    if size > manifest.written_bytes:
        logging.info(
            "Truncating %s from %d to the %d bytes covered by the manifest",
            passages_path, size, manifest.written_bytes,
        )
        with open(passages_path, "r+b") as output_file:
            output_file.truncate(manifest.written_bytes)
            output_file.flush()
            os.fsync(output_file.fileno())
    lines = _tail_lines(passages_path)
    if not lines:
        return False, "no complete document at the end of the encoded corpus", manifest
    dropped = 0
    valid_id: Optional[str] = None
    end_offset = manifest.written_bytes
    for line, end in reversed(lines[-max(1, validate_lines):]):
        document = _parse_document(line)
        if document is not None:
            valid_id, end_offset = document.get("id"), end
            break
        dropped += 1
    if valid_id is None:
        return False, f"the last {dropped} documents of the encoded corpus are unreadable", manifest
    if dropped > 0:
        logging.info("Dropping %d unreadable document(s) from the end of %s", dropped, passages_path)
        with open(passages_path, "r+b") as output_file:
            output_file.truncate(end_offset)
            output_file.flush()
            os.fsync(output_file.fileno())
        manifest = PassagesManifest.from_dict(manifest.to_dict())
        manifest.written_doc_count -= dropped
        manifest.written_bytes = end_offset
        manifest.last_doc_id = valid_id
    elif manifest.last_doc_id is not None and manifest.last_doc_id != valid_id:
        return False, f"last document id {valid_id!r} does not match the manifest ({manifest.last_doc_id!r})", manifest
    if manifest.written_doc_count <= 0:
        return False, "nothing usable is left in the encoded corpus", manifest
    return True, f"resuming after document {manifest.last_doc_id!r}", manifest


def skip_documents(docs: Iterable, count: int, expected_last_id: Optional[str]) -> Iterator:
    """Advance the corpus stream past ``count`` documents, verifying the position.

    ``RetrievalTask.passages()`` streams the corpus line by line with no shuffling
    and no parallelism, so the n-th document of the output always corresponds to
    the n-th line of the input. That is what makes resuming safe - and what has
    to be re-checked before appending anything.
    """
    stream = iter(docs)
    seen, last = 0, None
    for doc in stream:
        last = doc
        seen += 1
        if seen >= count:
            break
    if seen < count:
        raise ResumeMismatch(f"the corpus holds {seen} documents, fewer than the {count} already encoded")
    if expected_last_id is not None and last is not None and last.id != expected_last_id:
        raise ResumeMismatch(
            f"document {count} of the corpus is {last.id!r}, but {expected_last_id!r} was encoded"
        )
    return stream


class PassagesWriter:
    """Append encoded documents while keeping a conservative manifest on disk.

    The manifest is rewritten every ``flush_batches`` batches, always after
    flushing and fsyncing the data file, so ``written_bytes`` can only ever lag
    behind what is really on disk - never lead it.
    """

    def __init__(self, passages_path: str, manifest_path: str, manifest: PassagesManifest,
                 flush_batches: int = 100, append: bool = False):
        self.passages_path = passages_path
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.flush_batches = max(1, int(flush_batches))
        self.append = append
        self.file = None
        self._batches_since_checkpoint = 0

    def __enter__(self) -> "PassagesWriter":
        os.makedirs(os.path.dirname(os.path.abspath(self.passages_path)), exist_ok=True)
        if not self.append:
            self.manifest.written_doc_count = 0
            self.manifest.written_bytes = 0
            self.manifest.last_doc_id = None
        self.manifest.complete = False
        self.file = open(self.passages_path, "a" if self.append else "w", encoding="utf-8")
        self.checkpoint()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            self.checkpoint()
        except OSError as error:
            logging.warning("Could not checkpoint %s: %s", self.manifest_path, error)
        finally:
            self.file.close()
            self.file = None
        return False

    def note_batch(self, documents: int, last_doc_id: Optional[str]) -> None:
        """Record a written batch, checkpointing the manifest every so often."""
        self.manifest.written_doc_count += documents
        if last_doc_id is not None:
            self.manifest.last_doc_id = last_doc_id
        self._batches_since_checkpoint += 1
        if self._batches_since_checkpoint >= self.flush_batches:
            self.checkpoint()

    def checkpoint(self) -> None:
        self.file.flush()
        os.fsync(self.file.fileno())
        self.manifest.written_bytes = os.fstat(self.file.fileno()).st_size
        atomic_write_json(self.manifest_path, self.manifest.to_dict())
        self._batches_since_checkpoint = 0

    def finalize(self) -> PassagesManifest:
        """Mark the encoded corpus complete; the document count is now known."""
        self.manifest.complete = True
        self.manifest.corpus_doc_count = self.manifest.written_doc_count
        self.checkpoint()
        if self.manifest.corpus_path:
            # Phase A already streamed the whole corpus, so the count is free.
            store_line_count(self.manifest.corpus_path, self.manifest.written_doc_count)
        return self.manifest


