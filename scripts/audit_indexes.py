#!/usr/bin/env python
"""Audit the index cache after an interrupted benchmark.

A job killed mid-benchmark can leave two kinds of leftovers behind: a Lucene
index that opens cleanly but holds only part of the corpus (which quietly lowers
NDCG), and a partially encoded ``docs/passages.jsonl`` (which is worth GPU
hours). This tool reports both and, with --fix, deletes the first while keeping
the second.

Usage:
    python scripts/audit_indexes.py --cache-dir cache --data-dir data
    python scripts/audit_indexes.py --fix
    python scripts/audit_indexes.py --purge-passages-matching final --fix
"""

import argparse
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.splade_artifacts import PASSAGES_MANIFEST_NAME, read_passages_manifest  # noqa: E402
from utils.artifacts import (  # noqa: E402
    cached_line_count,
    count_lines,
    has_lucene_segments,
    read_index_manifest,
    trust_legacy_artifacts,
    write_index_manifest,
)
from utils.lucene import lucene_available, lucene_num_docs  # noqa: E402

# Index verdicts.
COMPLETE = "complete"
TRUNCATED = "truncated"
NO_SEGMENTS = "no-segments"
UNREADABLE = "unreadable"
MIGRATED = "migrated"
UNVERIFIABLE = "unverifiable"
# Encoded corpus verdicts.
PARTIAL = "partial"
LEGACY = "no-manifest"


@dataclass
class IndexReport:
    path: str
    task_id: str
    model: str
    verdict: str
    num_docs: Optional[int] = None
    expected_docs: Optional[int] = None
    size_bytes: int = 0
    action: str = "none"

    @property
    def broken(self) -> bool:
        return self.verdict in (TRUNCATED, NO_SEGMENTS, UNREADABLE)


@dataclass
class PassagesReport:
    path: str
    task_id: str
    model: str
    verdict: str
    written_docs: Optional[int] = None
    expected_docs: Optional[int] = None
    size_bytes: int = 0
    model_path: Optional[str] = None
    action: str = "none"


@dataclass
class Audit:
    indexes: List[IndexReport] = field(default_factory=list)
    passages: List[PassagesReport] = field(default_factory=list)


def directory_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                continue
    return total


def corpus_doc_count(data_dir: str, task_id: str, compute: bool) -> Optional[int]:
    corpus_path = os.path.join(data_dir, task_id, "passages", "passages.jsonl")
    if not os.path.exists(corpus_path):
        return None
    return cached_line_count(corpus_path, compute=compute)


def audit_model_dir(model_dir: str, task_id: str, data_dir: str, count_corpus: bool) -> Audit:
    """Inspect one <cache>/<task>/<model> directory."""
    audit = Audit()
    model = os.path.basename(model_dir)
    manifest_path = os.path.join(model_dir, PASSAGES_MANIFEST_NAME)
    manifest = read_passages_manifest(manifest_path)
    passages_path = os.path.join(model_dir, "docs", "passages.jsonl")
    expected = manifest.corpus_doc_count if manifest is not None else None
    if expected is None:
        expected = corpus_doc_count(data_dir, task_id, compute=count_corpus)

    if os.path.exists(passages_path):
        audit.passages.append(_audit_passages(passages_path, task_id, model, manifest, expected, count_corpus))

    for name in sorted(os.listdir(model_dir)):
        index_dir = os.path.join(model_dir, name)
        if not os.path.isdir(index_dir) or not name.endswith("_index"):
            continue
        audit.indexes.append(_audit_index(index_dir, task_id, model, expected))
    return audit


def _audit_passages(passages_path, task_id, model, manifest, expected, count_corpus) -> PassagesReport:
    size = os.path.getsize(passages_path)
    if manifest is None:
        written = count_lines(passages_path) if count_corpus else None
        verdict = LEGACY
        if written is not None and expected is not None and written != expected:
            verdict = PARTIAL
        return PassagesReport(passages_path, task_id, model, verdict, written, expected, size)
    verdict = COMPLETE if manifest.complete and size == manifest.written_bytes else PARTIAL
    return PassagesReport(
        passages_path, task_id, model, verdict, manifest.written_doc_count, expected, size,
        model_path=manifest.model_path,
    )


def _audit_index(index_dir, task_id, model, expected) -> IndexReport:
    size = directory_size(index_dir)
    if not has_lucene_segments(index_dir):
        return IndexReport(index_dir, task_id, model, NO_SEGMENTS, size_bytes=size)
    index_manifest = read_index_manifest(index_dir)
    recorded = index_manifest.get("doc_count") if index_manifest else None
    num_docs = lucene_num_docs(index_dir)
    if num_docs is None:
        return IndexReport(index_dir, task_id, model, UNREADABLE, expected_docs=recorded or expected, size_bytes=size)
    target = recorded if recorded is not None else expected
    if target is None:
        # Nothing to compare against; refuse to guess either way.
        return IndexReport(index_dir, task_id, model, UNVERIFIABLE, num_docs, None, size)
    if num_docs != target:
        return IndexReport(index_dir, task_id, model, TRUNCATED, num_docs, target, size)
    verdict = COMPLETE if index_manifest is not None else MIGRATED
    return IndexReport(index_dir, task_id, model, verdict, num_docs, target, size)


def audit_cache(cache_dir: str, data_dir: str, count_corpus: bool) -> Audit:
    audit = Audit()
    if not os.path.isdir(cache_dir):
        raise SystemExit(f"No such cache directory: {cache_dir}")
    for task_id in sorted(os.listdir(cache_dir)):
        task_dir = os.path.join(cache_dir, task_id)
        if not os.path.isdir(task_dir):
            continue
        for model in sorted(os.listdir(task_dir)):
            model_dir = os.path.join(task_dir, model)
            if not os.path.isdir(model_dir) or model == "results":
                continue
            part = audit_model_dir(model_dir, task_id, data_dir, count_corpus)
            audit.indexes.extend(part.indexes)
            audit.passages.extend(part.passages)
    return audit


def apply_fixes(audit: Audit, purge_passages: bool, purge_matching: Optional[str]) -> None:
    """Delete broken indexes and add manifests to verified artifacts.

    Rebuilding a Lucene index costs minutes, so deleting one that might be broken
    is cheap; keeping it risks silently wrong results. Encoded corpora are never
    deleted here - they cost GPU hours and have their own flag.
    """
    for report in audit.indexes:
        if report.broken:
            logging.info("Removing %s (%s)", report.path, report.verdict)
            shutil.rmtree(report.path, ignore_errors=True)
            report.action = "removed"
        elif report.verdict == MIGRATED:
            write_index_manifest(report.path, report.num_docs)
            report.action = "manifest written"
    for report in audit.passages:
        if purge_passages and _matches_purge(report, purge_matching):
            logging.info("Removing %s", report.path)
            os.remove(report.path)
            report.action = "removed"


def _matches_purge(report: PassagesReport, purge_matching: Optional[str]) -> bool:
    if purge_matching is None:
        return True
    haystack = f"{report.model_path or ''} {report.model} {report.path}"
    return purge_matching in haystack


def _format_docs(value: Optional[int]) -> str:
    return "?" if value is None else str(value)


def print_report(audit: Audit, fixing: bool) -> None:
    print(f"\n{'INDEXES':<8} {len(audit.indexes)} found")
    header = f"{'verdict':<13} {'docs':>10} {'expected':>10} {'size':>9}  {'action':<15} path"
    print(header)
    print("-" * len(header))
    for report in sorted(audit.indexes, key=lambda item: (item.verdict, item.path)):
        print(f"{report.verdict:<13} {_format_docs(report.num_docs):>10} {_format_docs(report.expected_docs):>10} "
              f"{report.size_bytes / 1e9:>8.2f}G  {report.action:<15} {report.path}")

    print(f"\n{'ENCODED CORPORA':<8} {len(audit.passages)} found")
    print(header)
    print("-" * len(header))
    for report in sorted(audit.passages, key=lambda item: (item.verdict, item.path)):
        print(f"{report.verdict:<13} {_format_docs(report.written_docs):>10} {_format_docs(report.expected_docs):>10} "
              f"{report.size_bytes / 1e9:>8.2f}G  {report.action:<15} {report.path}")

    broken = [report for report in audit.indexes if report.broken]
    freed = sum(report.size_bytes for report in broken)
    removed_passages = [report for report in audit.passages if report.action == "removed"]
    freed += sum(report.size_bytes for report in removed_passages)
    incomplete = [report for report in audit.passages if report.verdict != COMPLETE]
    verb = "freed" if fixing else "would be freed"
    print(f"\n{len(broken)} broken index(es), {freed / 1e9:.2f} GB {verb}")
    # A partial encoded corpus is resumed from where it stopped rather than
    # recomputed, as long as it still lines up with its corpus.
    print(f"{len(incomplete)} incomplete encoded corpus/corpora to resume, {len(broken)} index rebuild(s)")
    if not fixing and (broken or removed_passages):
        print("Nothing was changed; re-run with --fix to apply.")


def audit_to_json(audit: Audit) -> Dict:
    return {
        "indexes": [report.__dict__ for report in audit.indexes],
        "passages": [report.__dict__ for report in audit.passages],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", default="cache", help="Directory with cached indexes")
    parser.add_argument("--data-dir", default="data", help="Directory with task corpora")
    parser.add_argument("--json", dest="json_path", default=None, help="Also write the report as JSON")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Report only (default)")
    parser.add_argument("--fix", action="store_true",
                        help="Delete broken indexes and add manifests to verified artifacts")
    parser.add_argument("--purge-passages", action="store_true",
                        help="With --fix, also delete encoded corpora (hours of GPU time)")
    parser.add_argument("--purge-passages-matching", default=None,
                        help="Restrict --purge-passages to artifacts whose model path contains this text")
    parser.add_argument("--no-count-corpus", dest="count_corpus", action="store_false",
                        help="Never read corpora to count documents (leaves some verdicts unverifiable)")
    parser.add_argument("--trust-legacy-artifacts", action="store_true",
                        help="Treat artifacts without a manifest as valid when their document count matches")
    args = parser.parse_args(argv)
    logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.INFO)

    if args.trust_legacy_artifacts:
        os.environ["PIRB_TRUST_LEGACY_ARTIFACTS"] = "1"
    if args.purge_passages_matching is not None:
        args.purge_passages = True

    if args.fix and not lucene_available():
        raise SystemExit(
            "pyserini is not installed, so no index can be read and every one would look broken. "
            "Run --fix in the benchmark environment instead."
        )

    audit = audit_cache(args.cache_dir, args.data_dir, args.count_corpus)
    if args.fix:
        apply_fixes(audit, args.purge_passages, args.purge_passages_matching)
    else:
        legacy = [report for report in audit.passages if report.verdict == LEGACY]
        if legacy and not trust_legacy_artifacts():
            logging.info("%d encoded corpus/corpora have no manifest and will be re-encoded", len(legacy))
    print_report(audit, fixing=args.fix)
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as output_file:
            json.dump(audit_to_json(audit), output_file, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
