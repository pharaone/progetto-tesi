"""Load ISO/IEC 42001 requirements dynamically from ChromaDB.

Requirements are populated in chunk metadata during ISO document indexing
(see indexer.py). This module groups chunks by requirement_id and returns
a list of {"id": req_id, "text": combined_text} for use in agent loops.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from rag.collections import get_collection

logger = logging.getLogger(__name__)


def load_requirements_from_rag(
    collection_name: str,
    section_prefixes: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Load ISO requirements from a ChromaDB collection.

    Each requirement is reconstructed by grouping all chunks that share
    the same requirement_id metadata field. Chunks are sorted by
    chunk_index and concatenated to form the complete requirement text.

    Args:
        collection_name: ChromaDB collection to query (e.g., "ISO-CL456").
        section_prefixes: Optional list of ID prefixes to filter.
            E.g., ["cl-4", "cl-5", "cl-6"] for AS-1.

    Returns:
        Sorted list of {"id": requirement_id, "text": requirement_text}.
        Returns empty list if the collection has no indexed requirements —
        callers should treat this as a 503 condition.
    """
    try:
        collection = get_collection(collection_name)
        if collection.count() == 0:
            logger.warning(f"Collection '{collection_name}' is empty — ISO not yet indexed")
            return []

        results = collection.get(
            where={"requirement_id": {"$ne": ""}},
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        logger.error(
            f"Failed to load requirements from '{collection_name}': {exc}",
            exc_info=True,
        )
        return []

    docs = results.get("documents") or []
    metas = results.get("metadatas") or []

    if not docs:
        logger.warning(f"No requirement chunks found in '{collection_name}'")
        return []

    # Group chunks by requirement_id, applying prefix filter
    grouped: Dict[str, List[tuple[int, str]]] = {}
    for doc, meta in zip(docs, metas):
        if not meta:
            continue
        req_id = str(meta.get("requirement_id", "")).strip()
        if not req_id:
            continue
        if section_prefixes and not any(req_id.startswith(p) for p in section_prefixes):
            continue
        chunk_index = int(meta.get("chunk_index", 0))
        grouped.setdefault(req_id, []).append((chunk_index, doc or ""))

    if not grouped:
        logger.warning(
            f"No matching requirements in '{collection_name}' for prefixes={section_prefixes}"
        )
        return []

    # Sort chunks within each requirement and join text
    requirements = []
    for req_id, chunks in grouped.items():
        chunks.sort(key=lambda x: x[0])
        combined = " ".join(text for _, text in chunks).strip()
        requirements.append({"id": req_id, "text": combined})

    requirements.sort(key=_sort_key)

    logger.info(
        f"Loaded {len(requirements)} requirements from '{collection_name}' "
        f"(prefixes={section_prefixes})"
    )
    return requirements


def _sort_key(req: Dict[str, str]) -> tuple:
    """Natural sort: cl-4.1 < cl-10.2 < A.2.1 < A.10.3."""
    req_id = req["id"]
    if req_id.startswith("cl-"):
        prefix = 0
        tail = req_id[3:]
    elif req_id.startswith("A."):
        prefix = 1
        tail = req_id[2:]
    else:
        prefix = 2
        tail = req_id

    parts: list = []
    for segment in tail.split("."):
        try:
            parts.append(int(segment))
        except ValueError:
            parts.append(segment)

    return (prefix, *parts)
