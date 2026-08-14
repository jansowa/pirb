"""End-to-end tests for surviving a SIGKILL in the middle of a splade build.

Every interruption here is a real SIGKILL of a real subprocess: an exception
raised in-process would still unwind and flush, which is precisely what the
batch scheduler does not do when it kills a job.
"""

import json
import os
import subprocess
import sys

import pytest

import _stubs

_stubs.install()

from _fake_index import ENCODER_LOG, encoded_documents, fake_num_docs, read_passages, write_corpus  # noqa: E402
from _resume_worker import TASK_ID, make_index  # noqa: E402
from utils.artifacts import has_lucene_segments, read_index_manifest  # noqa: E402
from search.splade_artifacts import PASSAGES_MANIFEST_NAME, read_passages_manifest  # noqa: E402

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_resume_worker.py")
DOCUMENTS = 200


@pytest.fixture
def root(tmp_path):
    """A workspace with a corpus and a checkpoint directory to fingerprint."""
    root = tmp_path / "run"
    model_dir = root / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights" * 16)
    write_corpus(str(root / "data" / TASK_ID / "passages" / "passages.jsonl"), DOCUMENTS)
    return root


def run_worker(root, *args, expect_kill=False):
    process = subprocess.run(
        [sys.executable, WORKER, "--root", str(root), *args],
        capture_output=True, text=True,
    )
    if expect_kill:
        assert process.returncode == -9, f"expected SIGKILL, got {process.returncode}: {process.stderr[-2000:]}"
        return None
    assert process.returncode == 0, process.stderr[-4000:]
    return json.loads(process.stdout.strip().splitlines()[-1])


def paths(root):
    index = make_index(str(root))[0]
    return index


def test_a_kill_while_encoding_resumes_where_it_stopped(root):
    run_worker(root, "--kill-after-docs", "80", expect_kill=True)
    index = paths(root)
    manifest = read_passages_manifest(index.manifest_path)
    assert manifest is not None and not manifest.complete
    assert 0 < manifest.written_doc_count <= 80
    encoded_before = encoded_documents(str(root / ENCODER_LOG))

    result = run_worker(root)

    assert result["encoded_here"] == DOCUMENTS - manifest.written_doc_count
    assert encoded_documents(str(root / ENCODER_LOG)) == encoded_before + result["encoded_here"]
    documents = read_passages(index.docs_path)
    assert len(documents) == DOCUMENTS
    assert len({document["id"] for document in documents}) == DOCUMENTS
    assert [document["id"] for document in documents] == [f"doc_{i}" for i in range(DOCUMENTS)]
    assert result["complete"] is True


def test_a_kill_while_indexing_keeps_the_encoded_corpus(root):
    run_worker(root, "--kill-in-phase-b", expect_kill=True)
    index = paths(root)
    assert read_passages_manifest(index.manifest_path).complete is True
    assert index.exists() is False, "a half-written index must not look ready"
    encoded_before = encoded_documents(str(root / ENCODER_LOG))
    assert encoded_before == DOCUMENTS

    result = run_worker(root)

    assert result["encoded_here"] == 0, "the encoder must not run again"
    assert encoded_documents(str(root / ENCODER_LOG)) == encoded_before
    assert result["complete"] is True
    assert fake_num_docs(index.index_dir) == DOCUMENTS


def test_a_kill_after_indexing_repeats_neither_phase(root):
    run_worker(root)
    index = paths(root)
    encoded_before = encoded_documents(str(root / ENCODER_LOG))
    index_mtime = os.stat(index.index_dir).st_mtime

    result = run_worker(root)

    assert result["skipped_build"] is True
    assert result["encoded_here"] == 0
    assert encoded_documents(str(root / ENCODER_LOG)) == encoded_before
    assert os.stat(index.index_dir).st_mtime == index_mtime


def test_a_kill_while_indexing_leaves_the_previous_index_usable(root):
    run_worker(root)
    index = paths(root)
    assert index.exists() is True

    run_worker(root, "--force-build", "--kill-in-phase-b", expect_kill=True)

    assert has_lucene_segments(index.index_dir)
    assert fake_num_docs(index.index_dir) == DOCUMENTS
    assert index.exists() is True, "the working index must survive a failed rebuild"
    leftovers = [name for name in os.listdir(index.base_dir) if ".tmp-" in name]
    assert leftovers, "the interrupted build should be visible as a temporary directory"


def test_artifacts_are_identical_with_and_without_interruptions(root, tmp_path):
    run_worker(root, "--kill-after-docs", "37", expect_kill=True)
    run_worker(root, "--kill-after-docs", "150", expect_kill=True)
    interrupted = run_worker(root)

    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    for name in ("model", "data"):
        subprocess.run(["cp", "-r", str(root / name), str(clean_root / name)], check=True)
    clean = run_worker(clean_root)

    interrupted_docs = read_passages(paths(root).docs_path)
    clean_docs = read_passages(paths(clean_root).docs_path)
    assert interrupted_docs == clean_docs
    assert interrupted["complete"] == clean["complete"] is True
    assert encoded_documents(str(clean_root / ENCODER_LOG)) == DOCUMENTS


def test_legacy_artifacts_are_adopted_instead_of_recomputed(root, monkeypatch):
    run_worker(root)
    index = paths(root)
    os.remove(index.manifest_path)
    os.remove(os.path.join(index.index_dir, ".index_manifest.json"))
    encoded_before = encoded_documents(str(root / ENCODER_LOG))

    monkeypatch.setenv("PIRB_TRUST_LEGACY_ARTIFACTS", "1")
    result = run_worker(root)

    assert result["skipped_build"] is True, "a verified legacy index must count as complete"
    assert encoded_documents(str(root / ENCODER_LOG)) == encoded_before
    assert read_index_manifest(index.index_dir)["doc_count"] == DOCUMENTS


SAMPLE_SIZE = 4


def _valid_lines(path):
    """Lines that a reader could trust, independent of the code under test."""
    count = 0
    with open(path, "rb") as input_file:
        for line in input_file:
            if not line.endswith(b"\n"):
                break
            try:
                json.loads(line)
            except ValueError:
                break
            count += 1
    return count


def test_a_partial_corpus_without_a_manifest_is_still_resumed(root):
    """The msmarco case: killed mid-encoding before manifests existed."""
    run_worker(root, "--kill-after-docs", "80", expect_kill=True)
    index = paths(root)
    os.remove(index.manifest_path)
    already_encoded = _valid_lines(index.docs_path)
    assert 0 < already_encoded < DOCUMENTS
    encoded_before = encoded_documents(str(root / ENCODER_LOG))

    result = run_worker(root)

    assert result["encoded_here"] == DOCUMENTS - already_encoded + SAMPLE_SIZE, \
        "only the remaining documents plus the verification sample may be encoded"
    assert encoded_documents(str(root / ENCODER_LOG)) == encoded_before + result["encoded_here"]
    documents = read_passages(index.docs_path)
    assert [document["id"] for document in documents] == [f"doc_{i}" for i in range(DOCUMENTS)]
    assert result["complete"] is True


def test_a_rejected_resume_still_encodes_the_whole_corpus(root):
    """A rejected resume must not silently drop the documents it skipped.

    skip_documents consumes the passage stream before it can notice that the
    corpus moved under it, so falling back to a rebuild has to re-read the corpus
    from the start. Encoding what is left of a partly consumed stream would write
    an index missing its leading documents and report success.
    """
    index, task, data_dir = make_index(str(root))
    os.makedirs(os.path.dirname(index.docs_path), exist_ok=True)
    # Ids that parse and line up with the manifest, but not with the corpus.
    with open(index.docs_path, "w", encoding="utf-8") as passages_file:
        for i in range(8):
            passages_file.write(json.dumps({"id": f"other_{i}", "content": "", "vector": {"a": 1}}) + "\n")
    manifest = index._current_manifest()
    manifest.written_doc_count = 8
    manifest.written_bytes = os.path.getsize(index.docs_path)
    manifest.last_doc_id = "other_7"
    index._store_manifest(manifest)

    index.build(task.passages(data_dir, verbose=False))

    documents = read_passages(index.docs_path)
    assert [document["id"] for document in documents] == [f"doc_{i}" for i in range(DOCUMENTS)]
    assert index.encoder.encoded == DOCUMENTS
    assert read_passages_manifest(index.manifest_path).written_doc_count == DOCUMENTS
    assert index.exists() is True


def test_a_complete_corpus_without_a_manifest_is_reused(root):
    run_worker(root)
    index = paths(root)
    os.remove(index.manifest_path)
    encoded_before = encoded_documents(str(root / ENCODER_LOG))

    result = run_worker(root, "--force-build")

    assert result["encoded_here"] == SAMPLE_SIZE, "the sample is the only thing worth re-encoding"
    assert encoded_documents(str(root / ENCODER_LOG)) == encoded_before + SAMPLE_SIZE
    assert os.path.exists(os.path.join(index.base_dir, PASSAGES_MANIFEST_NAME))


def test_a_corpus_encoded_by_another_model_is_recomputed(root):
    run_worker(root)
    index = paths(root)
    os.remove(index.manifest_path)
    encoded_before = encoded_documents(str(root / ENCODER_LOG))

    result = run_worker(root, "--force-build", "--model-tag", "other-checkpoint")

    assert result["encoded_here"] == DOCUMENTS + SAMPLE_SIZE, "a foreign encoding must not be appended to"
    assert encoded_documents(str(root / ENCODER_LOG)) == encoded_before + DOCUMENTS + SAMPLE_SIZE
    assert read_passages(index.docs_path)[0]["vector"], "the file was rewritten by the current model"
