"""Helpers for cache artifacts that have to survive a killed process.

Index builds run for hours on a batch queue, so every artifact written along
the way has to be either valid or recognizably incomplete after a SIGKILL.
Nothing here may read a multi-gigabyte file unless it is asked to.
"""

import glob
import hashlib
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Tuple

# Files that identify a checkpoint well enough for a fingerprint. Weights are
# fingerprinted by (name, size, mtime): hashing gigabytes of tensors would cost
# seconds of I/O per task for no extra safety.
FINGERPRINT_SUFFIXES = (".safetensors", ".bin", ".pt", ".onnx", ".json", ".txt", ".model")
DOC_COUNT_CACHE_NAME = ".doc_count.json"
TRUST_LEGACY_ENV = "PIRB_TRUST_LEGACY_ARTIFACTS"
INDEX_MANIFEST_SCHEMA = 1
INDEX_MANIFEST_NAME = ".index_manifest.json"


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write JSON so that readers see either the old file or the new one."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = os.path.join(directory, f".{os.path.basename(path)}.tmp-{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as tmp_file:
        json.dump(payload, tmp_file, ensure_ascii=False, indent=2)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
    os.replace(tmp_path, path)


def read_json_or_none(path: str) -> Optional[Dict[str, Any]]:
    """Read a JSON file, treating a missing or corrupted one as absent."""
    try:
        with open(path, "r", encoding="utf-8") as input_file:
            value = json.load(input_file)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def model_fingerprint(model_name_or_path: str) -> Optional[str]:
    """Fingerprint a local checkpoint directory, or None when unavailable.

    None means "cannot be verified" (a hub id, a deleted directory), and callers
    must then refuse to reuse anything encoded by an earlier run.
    """
    if not model_name_or_path or not os.path.isdir(model_name_or_path):
        return None
    entries = []
    for root, _dirs, files in os.walk(model_name_or_path):
        for name in sorted(files):
            if not name.endswith(FINGERPRINT_SUFFIXES):
                continue
            path = os.path.join(root, name)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            relative = os.path.relpath(path, model_name_or_path)
            entries.append((relative, stat.st_size, stat.st_mtime))
    if not entries:
        return None
    digest = hashlib.sha256()
    for relative, size, mtime in sorted(entries):
        digest.update(f"{relative}\0{size}\0{mtime!r}\n".encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


def corpus_signature(corpus_path: Optional[str]) -> Tuple[Optional[int], Optional[float]]:
    """Size and mtime of the corpus: one stat call, no reads."""
    if not corpus_path:
        return None, None
    try:
        stat = os.stat(corpus_path)
    except OSError:
        return None, None
    return stat.st_size, stat.st_mtime


def count_lines(path: str) -> int:
    count = 0
    with open(path, "rb") as input_file:
        while True:
            chunk = input_file.read(1024 * 1024)
            if not chunk:
                break
            count += chunk.count(b"\n")
    return count


def cached_line_count(path: str, compute: bool = True) -> Optional[int]:
    """Number of lines in a corpus file, cached next to it.

    Counting lines in the larger corpora means reading tens of gigabytes, so the
    result is cached and keyed by the file size and mtime. ``compute=False``
    returns None instead of reading the file.
    """
    size, mtime = corpus_signature(path)
    if size is None:
        return None
    cache_path = os.path.join(os.path.dirname(os.path.abspath(path)), DOC_COUNT_CACHE_NAME)
    cache = read_json_or_none(cache_path) or {}
    entry = cache.get(os.path.basename(path))
    if isinstance(entry, dict) and entry.get("size") == size and entry.get("mtime") == mtime:
        count = entry.get("count")
        if isinstance(count, int):
            return count
    if not compute:
        return None
    count = count_lines(path)
    store_line_count(path, count)
    return count


def store_line_count(path: str, count: int) -> None:
    """Remember a known line count for a corpus file."""
    size, mtime = corpus_signature(path)
    if size is None:
        return
    cache_path = os.path.join(os.path.dirname(os.path.abspath(path)), DOC_COUNT_CACHE_NAME)
    cache = read_json_or_none(cache_path) or {}
    cache[os.path.basename(path)] = {"size": size, "mtime": mtime, "count": count}
    try:
        atomic_write_json(cache_path, cache)
    except OSError as error:
        # A read-only data directory must not fail a build.
        logging.debug("Could not cache the line count of %s: %s", path, error)


def trust_legacy_artifacts() -> bool:
    """Whether artifacts without a manifest may be trusted (opt-in)."""
    value = os.environ.get(TRUST_LEGACY_ENV, "")
    return value.strip().lower() in ("1", "true", "yes", "on")


def index_manifest_path(index_dir: str) -> str:
    return os.path.join(index_dir, INDEX_MANIFEST_NAME)


def write_index_manifest(index_dir: str, doc_count: Optional[int]) -> None:
    """Record how many documents a finished index is supposed to hold."""
    atomic_write_json(index_manifest_path(index_dir), {
        "schema_version": INDEX_MANIFEST_SCHEMA,
        "doc_count": doc_count,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })


def read_index_manifest(index_dir: str) -> Optional[Dict[str, Any]]:
    value = read_json_or_none(index_manifest_path(index_dir))
    if value is None or value.get("schema_version") != INDEX_MANIFEST_SCHEMA:
        return None
    return value


def has_lucene_segments(index_dir: str) -> bool:
    return bool(glob.glob(os.path.join(index_dir, "segments_*")))


def temporary_dir_name(target_dir: str, kind: str = "tmp") -> str:
    return f"{target_dir}.{kind}-{os.getpid()}-{int(time.time())}"


def remove_stale_dirs(target_dir: str, max_age_seconds: float = 24 * 60 * 60,
                      kinds: Iterable[str] = ("tmp", "old")) -> int:
    """Drop leftover ``<target_dir>.tmp-*`` / ``.old-*`` dirs from killed runs."""
    parent = os.path.dirname(os.path.abspath(target_dir))
    prefixes = tuple(f"{os.path.basename(target_dir)}.{kind}-" for kind in kinds)
    removed = 0
    now = time.time()
    try:
        names = os.listdir(parent)
    except OSError:
        return 0
    for name in names:
        if not name.startswith(prefixes):
            continue
        path = os.path.join(parent, name)
        if not os.path.isdir(path):
            continue
        try:
            age = now - os.stat(path).st_mtime
        except OSError:
            continue
        if age < max_age_seconds:
            continue
        logging.info("Removing stale build directory %s (%.1f h old)", path, age / 3600)
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    return removed
