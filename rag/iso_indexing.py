"""ISO 42001 document indexing: directory scan, collection assignment, auto-index.

Used both by the CLI script (scripts/index_iso.py) and by AS-1's startup
hook, which automatically indexes the ISO documents mounted at
/app/iso_docs the first time the instance starts.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List

from rag.collections import (
    COLLECTION_ISO_CL456,
    COLLECTION_ISO_CL78_A26,
    COLLECTION_ISO_CL910_A710,
    COLLECTION_ISO_FULL,
    get_collection,
    initialize_collections,
    reset_collection,
)
from rag.indexer import index_iso_document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".rst", ".text"}

ISO_COLLECTIONS = [
    COLLECTION_ISO_CL456,
    COLLECTION_ISO_CL78_A26,
    COLLECTION_ISO_CL910_A710,
    COLLECTION_ISO_FULL,
]


def determine_collections(filename: str) -> List[str]:
    """
    Determine which collections a file should be indexed into based on its name.

    Returns a list of collection names. All documents also go into ISO-FULL.
    """
    name = filename.lower()
    collections = set()

    # Clauses 4, 5, 6
    if re.search(r"cl[_-]?[456]|clause[_-]?[456]|part[_-]?[456]|ch[_-]?[456]", name):
        collections.add(COLLECTION_ISO_CL456)

    # Clauses 7, 8 + Annex A.2-A.6
    if re.search(r"cl[_-]?[78]|clause[_-]?[78]|part[_-]?[78]|ch[_-]?[78]", name):
        collections.add(COLLECTION_ISO_CL78_A26)
    if re.search(r"annex[_-]?a[_-]?[2-6]|annex_a[2-6]|a\.[2-6]", name):
        collections.add(COLLECTION_ISO_CL78_A26)

    # Clauses 9, 10 + Annex A.7-A.10
    if re.search(r"cl[_-]?(?:9|10)|clause[_-]?(?:9|10)|part[_-]?(?:9|10)|ch[_-]?(?:9|10)", name):
        collections.add(COLLECTION_ISO_CL910_A710)
    if re.search(r"annex[_-]?a[_-]?(?:[7-9]|10)|annex_a[7-9]|annex_a10|a\.(?:[7-9]|10)", name):
        collections.add(COLLECTION_ISO_CL910_A710)

    # Always add to ISO-FULL
    collections.add(COLLECTION_ISO_FULL)

    return list(collections)


def find_iso_files(docs_dir: str) -> List[Path]:
    """Find all supported ISO document files in a directory (recursive)."""
    docs_path = Path(docs_dir)
    if not docs_path.is_dir():
        return []
    files: List[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(docs_path.rglob(f"*{ext}"))
    return sorted(files)


def index_directory(docs_dir: str, dry_run: bool = False, reset: bool = False) -> Dict[str, int]:
    """
    Index all ISO documents in the given directory.

    Args:
        docs_dir: Directory containing ISO documents.
        dry_run: If True, only log what would be indexed.
        reset: If True, wipe the ISO collections first.

    Returns:
        Summary dict with chunk counts per collection.
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Directory not found: {docs_dir}")
    if not docs_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {docs_dir}")

    files = find_iso_files(docs_dir)
    if not files:
        logger.warning(f"No supported documents found in {docs_dir}")
        logger.warning(f"Supported extensions: {SUPPORTED_EXTENSIONS}")
        return {}

    logger.info(f"Found {len(files)} document(s) in {docs_dir}")

    if not dry_run:
        logger.info("Initializing ChromaDB collections...")
        initialize_collections()
        if reset:
            for name in ISO_COLLECTIONS:
                logger.info(f"Resetting collection '{name}'...")
                reset_collection(name)

    summary: Dict[str, int] = {name: 0 for name in ISO_COLLECTIONS}

    for file_path in files:
        collections = determine_collections(file_path.name)
        logger.info(f"File: {file_path.name} → collections: {', '.join(collections)}")

        if dry_run:
            logger.info(f"  [DRY RUN] Would index {file_path.name} into: {collections}")
            continue

        for collection_name in collections:
            try:
                chunks_indexed = index_iso_document(str(file_path), collection_name)
                summary[collection_name] = summary.get(collection_name, 0) + chunks_indexed
                logger.info(f"  Indexed {chunks_indexed} chunks into {collection_name}")
            except Exception as exc:
                logger.error(
                    f"  Failed to index {file_path.name} into {collection_name}: {exc}",
                    exc_info=True,
                )

    if not dry_run:
        logger.info("Indexing complete. Summary:")
        for collection, count in summary.items():
            logger.info(f"  {collection}: {count} chunks")

    return summary


def ensure_iso_indexed(docs_dir: str) -> None:
    """
    Auto-index the ISO standard at service startup (idempotent).

    If the ISO-FULL collection already contains chunks, does nothing.
    Otherwise indexes every supported file found in docs_dir. Intended
    to be called in a background thread from AS-1's startup hook so a
    fresh company instance needs no manual indexing step.
    """
    files = find_iso_files(docs_dir)
    if not files:
        logger.info(
            f"No ISO documents found in {docs_dir} — skipping auto-indexing. "
            f"Place the ISO 42001 file (txt/pdf) there and restart, or run "
            f"scripts/index_iso.py manually."
        )
        return

    try:
        count = get_collection(COLLECTION_ISO_FULL).count()
    except Exception as exc:
        logger.error(f"Could not check ISO-FULL collection: {exc}", exc_info=True)
        return

    if count > 0:
        logger.info(
            f"ISO already indexed ({count} chunks in {COLLECTION_ISO_FULL}) — "
            f"skipping auto-indexing"
        )
        return

    logger.info(f"ISO-FULL is empty — auto-indexing {len(files)} file(s) from {docs_dir}...")
    try:
        index_directory(docs_dir)
        logger.info("Automatic ISO indexing completed")
    except Exception as exc:
        logger.error(f"Automatic ISO indexing failed: {exc}", exc_info=True)
