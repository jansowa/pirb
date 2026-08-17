"""Build a splade index in a throwaway process that can be SIGKILLed mid-build.

Killing a subprocess is the only faithful way to test this: raising an exception
inside Python would still unwind, flush buffers and run `finally` blocks, which
is exactly what a job killed by the batch scheduler does not do.

Run as:
    python tests/_resume_worker.py --root DIR [--kill-after-docs N] [--kill-in-phase-b]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _stubs  # noqa: E402

_stubs.install()

from _fake_index import ENCODER_LOG, FakeEncoder, fake_backend_build, fake_num_docs  # noqa: E402
from data import RetrievalTask  # noqa: E402
from search.splade import SpladeIndex  # noqa: E402

TASK_ID = "resume-task"


def make_index(root: str, kill_after_docs=None, kill_in_phase_b=False, drop_docs=0,
               index_batch_size=10, manifest_flush_batches=1, model_dir=None,
               model_tag="m1", legacy_sample_size=4):
    data_dir = os.path.join(root, "data")
    cache_dir = os.path.join(root, "cache", TASK_ID)
    model_dir = model_dir or os.path.join(root, "model")
    task = RetrievalTask(TASK_ID, "test")
    config = {
        "name": model_dir,
        "type": "splade",
        "index_batch_size": index_batch_size,
        "manifest_flush_batches": manifest_flush_batches,
        "index_build_workers": 0,
        "max_seq_length": 128,
        "legacy_sample_size": legacy_sample_size,
    }
    index = SpladeIndex(config, data_dir=cache_dir, task_id=TASK_ID,
                        corpus_path=task.passages_path(data_dir))
    index.encoder = FakeEncoder(os.path.join(root, ENCODER_LOG), kill_after=kill_after_docs, model_tag=model_tag)
    index._get_encoder = lambda: index.encoder
    index.backend.build = fake_backend_build(index, kill_in_phase_b=kill_in_phase_b, drop_docs=drop_docs)
    index.backend.num_docs = lambda index_dir=None: fake_num_docs(index_dir or index.backend.index_dir)
    return index, task, data_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--kill-after-docs", type=int, default=None)
    parser.add_argument("--kill-in-phase-b", action="store_true")
    parser.add_argument("--drop-docs", type=int, default=0)
    parser.add_argument("--index-batch-size", type=int, default=10)
    parser.add_argument("--manifest-flush-batches", type=int, default=1)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--model-tag", default="m1",
                        help="Stands in for the checkpoint identity behind the encoded vectors")
    parser.add_argument("--legacy-sample-size", type=int, default=4)
    parser.add_argument("--force-build", action="store_true",
                        help="Build even if the index already looks complete, as --overwrite does")
    args = parser.parse_args(argv)

    index, task, data_dir = make_index(
        args.root, kill_after_docs=args.kill_after_docs, kill_in_phase_b=args.kill_in_phase_b,
        drop_docs=args.drop_docs, index_batch_size=args.index_batch_size,
        manifest_flush_batches=args.manifest_flush_batches, model_dir=args.model_dir,
        model_tag=args.model_tag, legacy_sample_size=args.legacy_sample_size,
    )
    already_complete = index.exists() and not args.force_build
    if not already_complete:
        index.build(task.passages(data_dir, verbose=False))
    print(json.dumps({
        "skipped_build": already_complete,
        "encoded_here": index.encoder.encoded,
        "index_dir": index.index_dir,
        "complete": index.exists(),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
