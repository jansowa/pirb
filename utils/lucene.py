"""Reading a Lucene index without committing to a searcher.

Kept separate from the backends so that tools which only need to know whether an
index is intact do not have to import pyserini (and start a JVM) at module load.
"""

import importlib.util
import logging
from typing import Optional


def lucene_available() -> bool:
    """Whether an index can be read at all in this environment.

    Without pyserini every index looks unreadable, which must not be mistaken
    for every index being broken.
    """
    try:
        return importlib.util.find_spec("pyserini") is not None
    except (ImportError, ValueError):
        return False


def lucene_num_docs(index_dir: str) -> Optional[int]:
    """Documents in the index at ``index_dir``, or None if it cannot be read.

    A directory left behind by a killed build, or one whose segments were partly
    removed, must answer None rather than raise: callers use this to decide that
    an index is unusable.
    """
    try:
        reader = _open_index_reader(index_dir)
    except Exception as error:
        logging.debug("Could not open the index %s: %s", index_dir, error)
        return None
    try:
        return int(reader.stats()["documents"])
    except Exception as error:
        logging.debug("Could not read the document count of %s: %s", index_dir, error)
        return None
    finally:
        close = getattr(reader, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass


def _open_index_reader(index_dir: str):
    try:
        from pyserini.index.lucene import LuceneIndexReader as Reader
    except ImportError:
        from pyserini.index.lucene import IndexReader as Reader
    return Reader(index_dir)
