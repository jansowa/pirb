"""Minimal stand-ins for the heavy retrieval dependencies.

The resume logic never calls into pyserini, pylate or a JVM: it decides what to
encode, appends to a JSONL file and renames directories. Stubbing the imports
that `search.splade` pulls in transitively lets those tests run anywhere,
including a machine with no Java and no pyserini installed.

Anything actually installed is left untouched, so on a full environment the same
tests exercise the real modules.
"""

import importlib.util
import sys
import types


def _missing(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ValueError):
        return True


def _module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], child, module)
    return module


def install() -> None:
    """Register stub modules for dependencies that are not installed."""
    if _missing("pylate"):
        _module("pylate")
        indexes = _module("pylate.indexes")
        indexes.PLAID = type("PLAID", (), {})
    if _missing("pyserini"):
        _module("pyserini")
        encode = _module("pyserini.encode")
        encode.QueryEncoder = type("QueryEncoder", (), {})
        search = _module("pyserini.search")
        search.LuceneImpactSearcher = type("LuceneImpactSearcher", (), {})
        index = _module("pyserini.index")
        lucene = _module("pyserini.index.lucene")

        class LuceneIndexReader:
            def __init__(self, index_dir: str):
                raise RuntimeError(f"no Lucene available for {index_dir}")

        lucene.LuceneIndexReader = LuceneIndexReader
        index.lucene = lucene
