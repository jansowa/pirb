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
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from utils.artifacts import (
    atomic_write_json,
    cached_line_count,
    count_lines,
    read_json_or_none,
    store_line_count,
)

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
                   trust_legacy: bool = False) -> PhaseADecision:
    """Pick REUSE, RESUME or REBUILD for the encoding phase.

    ``expected`` carries the current configuration (fingerprint, sequence length,
    quantization, corpus signature); only the cheap checks run here, nothing
    reads the corpus unless legacy artifacts are explicitly trusted.
    """
    stored = read_passages_manifest(manifest_path)
    if not os.path.exists(passages_path):
        return PhaseADecision(REBUILD, "no encoded corpus on disk", expected)
    if stored is None:
        return _decide_legacy(passages_path, expected, trust_legacy)
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


def _decide_legacy(passages_path: str, expected: PassagesManifest, trust_legacy: bool) -> PhaseADecision:
    """Handle an encoded corpus written before manifests existed."""
    if not trust_legacy:
        return PhaseADecision(
            REBUILD,
            "no manifest next to the encoded corpus, so the model and max_seq_length behind it are unknown "
            "(set PIRB_TRUST_LEGACY_ARTIFACTS=1 to trust it)",
            expected,
        )
    corpus_doc_count = expected.corpus_doc_count
    if corpus_doc_count is None and expected.corpus_path:
        corpus_doc_count = cached_line_count(expected.corpus_path)
    if corpus_doc_count is None:
        return PhaseADecision(REBUILD, "legacy artifacts trusted, but the corpus size is unknown", expected)
    written = count_lines(passages_path)
    if written != corpus_doc_count:
        return PhaseADecision(
            REBUILD, f"legacy encoded corpus has {written} of {corpus_doc_count} documents", expected
        )
    migrated = PassagesManifest(**expected.to_dict())
    migrated.corpus_doc_count = corpus_doc_count
    migrated.written_doc_count = written
    migrated.written_bytes = os.path.getsize(passages_path)
    migrated.last_doc_id = _last_document_id(passages_path)
    migrated.complete = True
    return PhaseADecision(REUSE, f"legacy encoded corpus trusted with {written} documents", migrated)


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


