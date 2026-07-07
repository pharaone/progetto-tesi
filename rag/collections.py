"""ChromaDB collection manager for ISO/IEC 42001 gap analysis system."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

# Collection names
COLLECTION_ISO_CL456 = "ISO-CL456"
COLLECTION_ISO_CL78_A26 = "ISO-CL78-A26"
COLLECTION_ISO_CL910_A710 = "ISO-CL910-A710"
COLLECTION_ISO_FULL = "ISO-FULL"
COLLECTION_ORG_DOCS = "ORG-DOCS"
COLLECTION_ORG_HISTORY = "ORG-HISTORY"

ALL_COLLECTIONS = [
    COLLECTION_ISO_CL456,
    COLLECTION_ISO_CL78_A26,
    COLLECTION_ISO_CL910_A710,
    COLLECTION_ISO_FULL,
    COLLECTION_ORG_DOCS,
    COLLECTION_ORG_HISTORY,
]

_client: Optional[chromadb.PersistentClient] = None
_embedding_function: Optional[embedding_functions.ONNXMiniLM_L6_V2] = None


def _get_client() -> chromadb.PersistentClient:
    """Get or create the ChromaDB persistent client."""
    global _client
    if _client is None:
        chromadb_path = os.getenv("CHROMADB_PATH", "/data/chromadb")
        os.makedirs(chromadb_path, exist_ok=True)
        _client = chromadb.PersistentClient(path=chromadb_path)
        logger.info(f"ChromaDB client initialized at {chromadb_path}")
    return _client


def _get_embedding_function() -> embedding_functions.ONNXMiniLM_L6_V2:
    """Get or create the ONNX-based all-MiniLM-L6-v2 embedding function.

    Uses onnxruntime instead of torch — cross-platform, no CUDA deps,
    same model as the sentence-transformers version.
    """
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = embedding_functions.ONNXMiniLM_L6_V2()
        logger.info("Embedding function initialized: ONNXMiniLM_L6_V2 (all-MiniLM-L6-v2)")
    return _embedding_function


def get_collection(name: str) -> chromadb.Collection:
    """Get or create a ChromaDB collection by name."""
    if name not in ALL_COLLECTIONS:
        raise ValueError(
            f"Unknown collection '{name}'. Valid collections: {ALL_COLLECTIONS}"
        )
    client = _get_client()
    ef = _get_embedding_function()
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def initialize_collections() -> Dict[str, chromadb.Collection]:
    """Initialize all collections and return a mapping."""
    collections = {}
    for name in ALL_COLLECTIONS:
        collections[name] = get_collection(name)
        logger.info(f"Collection ready: {name}")
    return collections


def add_documents(
    collection_name: str,
    docs: List[str],
    metadatas: List[dict],
    ids: List[str],
) -> None:
    """
    Add documents to a ChromaDB collection.

    Args:
        collection_name: Name of the target collection.
        docs: List of document text strings.
        metadatas: List of metadata dicts (one per document).
        ids: List of unique document IDs.
    """
    if not docs:
        logger.warning(f"add_documents called with empty docs for {collection_name}")
        return

    collection = get_collection(collection_name)

    # Upsert in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        collection.upsert(
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids,
        )
        logger.debug(
            f"Upserted batch {i // batch_size + 1} "
            f"({len(batch_docs)} docs) into {collection_name}"
        )

    logger.info(f"Added {len(docs)} documents to collection '{collection_name}'")


def query(
    collection_name: str,
    query_text: str,
    n_results: int = 5,
    where: Optional[dict] = None,
) -> Dict:
    """
    Query a ChromaDB collection for relevant documents.

    Args:
        collection_name: Name of the collection to query.
        query_text: Query string.
        n_results: Number of results to return.
        where: Optional metadata filter.

    Returns:
        ChromaDB query result dict with keys: ids, documents, metadatas, distances.
    """
    collection = get_collection(collection_name)
    count = collection.count()
    if count == 0:
        logger.warning(f"Collection '{collection_name}' is empty, returning no results")
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    # Clamp n_results to available documents
    actual_n = min(n_results, count)

    kwargs: dict = {"query_texts": [query_text], "n_results": actual_n}
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)
    return results


def reset_collection(name: str) -> None:
    """Delete and recreate a collection, removing all its documents."""
    if name not in ALL_COLLECTIONS:
        raise ValueError(
            f"Unknown collection '{name}'. Valid collections: {ALL_COLLECTIONS}"
        )
    client = _get_client()
    try:
        client.delete_collection(name)
        logger.info(f"Deleted collection '{name}'")
    except Exception:
        # Collection didn't exist yet — nothing to delete
        pass
    get_collection(name)


def query_iso_with_fallback(
    partition_collection: str,
    query_text: str,
    n_results: int = 5,
) -> Dict:
    """Query an ISO partition collection, falling back to ISO-FULL when it is empty.

    When the full ISO was indexed as a single file (into ISO-FULL only), the
    partition collections are empty. This helper transparently retries against
    ISO-FULL so callers always get semantic search results.
    """
    result = query(partition_collection, query_text, n_results)
    docs = result.get("documents", [[]])[0]
    if not docs and partition_collection != COLLECTION_ISO_FULL:
        logger.info(
            f"'{partition_collection}' returned no results, "
            f"falling back to '{COLLECTION_ISO_FULL}'"
        )
        result = query(COLLECTION_ISO_FULL, query_text, n_results)
    return result


def delete_documents_by_prefix(collection_name: str, id_prefix: str) -> int:
    """Delete all documents whose ID starts with the given prefix."""
    collection = get_collection(collection_name)
    # Retrieve all IDs first
    all_data = collection.get(include=[])
    ids_to_delete = [
        doc_id
        for doc_id in all_data.get("ids", [])
        if doc_id.startswith(id_prefix)
    ]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        logger.info(
            f"Deleted {len(ids_to_delete)} documents with prefix '{id_prefix}' "
            f"from '{collection_name}'"
        )
    return len(ids_to_delete)
