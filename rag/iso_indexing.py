"""ISO 42001 document indexing: directory scan, collection assignment, auto-index.

Used both by the CLI script (scripts/index_iso.py) and by AS-1's startup
hook, which automatically indexes the ISO documents mounted at
/app/iso_docs the first time the instance starts.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Indexing status
#
# Kept in a file next to the ChromaDB data, which every service mounts, so the
# orchestrator and all agents can refuse to start an analysis while the
# standard is still being indexed (a partial index would silently yield
# fewer requirements and missing ISO context).
# ---------------------------------------------------------------------------

STATE_READY = "ready"
STATE_INDEXING = "indexing"
STATE_FAILED = "failed"
STATE_MISSING = "missing"

_STATUS_MESSAGES = {
    STATE_READY: "The ISO/IEC 42001 index is complete.",
    STATE_INDEXING: (
        "The ISO/IEC 42001 standard is still being indexed. "
        "Retry when indexing is complete."
    ),
    STATE_FAILED: (
        "ISO/IEC 42001 indexing failed. Check the AS-1 logs, then restart AS-1 "
        "or run scripts/index_iso.py --reset."
    ),
    STATE_MISSING: (
        "The ISO/IEC 42001 standard is not indexed. Place the ISO file in "
        "iso_docs/ and restart AS-1, or run scripts/index_iso.py."
    ),
}


def _status_path() -> Path:
    return Path(os.getenv("CHROMADB_PATH", "/data/chromadb")) / "iso_index_status.json"


def _write_status(state: str, **details: Any) -> None:
    path = _status_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state": state, "updated_at": datetime.utcnow().isoformat() + "Z", **details}
    # Write-then-rename so readers never see a half-written file
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def iso_index_status() -> Dict[str, Any]:
    """Current indexing state: ready, indexing, failed or missing, with a message."""
    try:
        status = json.loads(_status_path().read_text())
    except (OSError, ValueError):
        status = {"state": STATE_MISSING}
    if status.get("state") not in _STATUS_MESSAGES:
        status = {"state": STATE_MISSING}
    status["message"] = _STATUS_MESSAGES[status["state"]]
    return status


def iso_index_ready() -> bool:
    return iso_index_status()["state"] == STATE_READY


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

    if dry_run:
        summary, _ = _index_files(files, dry_run=True)
        return summary

    _write_status(STATE_INDEXING, files=[f.name for f in files])
    try:
        logger.info("Initializing ChromaDB collections...")
        initialize_collections()
        if reset:
            for name in ISO_COLLECTIONS:
                logger.info(f"Resetting collection '{name}'...")
                reset_collection(name)
        summary, failed = _index_files(files, dry_run=False)
    except Exception as exc:
        _write_status(STATE_FAILED, error=str(exc))
        raise

    if failed or summary.get(COLLECTION_ISO_FULL, 0) == 0:
        _write_status(
            STATE_FAILED,
            error=f"failed files: {failed}" if failed else "no chunks indexed",
            chunks=summary,
        )
    else:
        _write_status(STATE_READY, chunks=summary)

    logger.info("Indexing complete. Summary:")
    for collection, count in summary.items():
        logger.info(f"  {collection}: {count} chunks")

    return summary


def _index_files(files: List[Path], dry_run: bool) -> Tuple[Dict[str, int], List[str]]:
    """Index each file into its collections.

    Returns the chunk count per collection and the names of the files that
    failed to index.
    """
    summary: Dict[str, int] = {name: 0 for name in ISO_COLLECTIONS}
    failed: List[str] = []

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
                failed.append(file_path.name)
                logger.error(
                    f"  Failed to index {file_path.name} into {collection_name}: {exc}",
                    exc_info=True,
                )

    return summary, failed


def ensure_iso_indexed(docs_dir: str) -> None:
    """
    Auto-index the ISO standard at service startup (idempotent).

    Does nothing when the status file says a previous indexing completed.
    Otherwise (never indexed, interrupted by a restart, failed, or indexed
    before the status file existed) rebuilds the ISO collections from
    docs_dir, so the index is known to be complete. Intended to be called in
    a background thread from AS-1's startup hook.
    """
    if iso_index_ready():
        logger.info("ISO index already complete — skipping auto-indexing")
        return

    files = find_iso_files(docs_dir)
    if not files:
        _adopt_existing_index(docs_dir)
        return

    logger.info(
        f"ISO index not marked complete — indexing {len(files)} file(s) from {docs_dir}..."
    )
    try:
        index_directory(docs_dir, reset=True)
        logger.info("Automatic ISO indexing completed")
    except Exception as exc:
        logger.error(f"Automatic ISO indexing failed: {exc}", exc_info=True)


def _adopt_existing_index(docs_dir: str) -> None:
    """No ISO files to index from: keep an index built earlier by hand, if any."""
    try:
        count: Optional[int] = get_collection(COLLECTION_ISO_FULL).count()
    except Exception as exc:
        logger.error(f"Could not check ISO-FULL collection: {exc}", exc_info=True)
        count = None

    if count:
        logger.info(
            f"No ISO documents in {docs_dir}, but {COLLECTION_ISO_FULL} already has "
            f"{count} chunks — treating the existing index as complete"
        )
        _write_status(STATE_READY, chunks={COLLECTION_ISO_FULL: count}, adopted=True)
    else:
        logger.info(
            f"No ISO documents found in {docs_dir} — skipping auto-indexing. "
            f"Place the ISO 42001 file (txt/pdf) there and restart, or run "
            f"scripts/index_iso.py manually."
        )
