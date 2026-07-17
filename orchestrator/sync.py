"""Incremental sync of external documentation sources into the corpus.

For each configured connector (connectors/*.py):
- list remote documents with a cheap version hash
- skip unchanged ones, download and (re)index new/changed ones
- remove documents that disappeared upstream (DB row + RAG chunks)

Synced documents live in the same documents table as manual uploads
(source = "github"/"confluence", uploader = "sync:<source>") and are
indexed into ORG-DOCS exactly like uploads, so the analysis pipeline is
untouched and reproducibility (input hash over the stored corpus) holds.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import List, Optional

from connectors.base import DocumentSource
from connectors.confluence import ConfluenceSource
from connectors.github import GitHubSource
from orchestrator import db
from shared.config import get_settings

logger = logging.getLogger(__name__)

ALL_CONNECTORS = [GitHubSource, ConfluenceSource]


def build_source(cls) -> tuple:
    """Instantiate one connector: UI-saved config first, env vars as fallback.

    Returns (source_or_None, origin) where origin is "ui" | "env" | None.
    """
    config = db.get_source_config(cls.name)
    if config:
        src = cls.from_config(config)
        if src is not None:
            return src, "ui"
    src = cls.from_env()
    if src is not None:
        return src, "env"
    return None, None


def get_source(name: str) -> Optional[DocumentSource]:
    for cls in ALL_CONNECTORS:
        if cls.name == name:
            return build_source(cls)[0]
    return None


def configured_sources() -> List[DocumentSource]:
    """Instantiate every configured connector (UI config or env)."""
    sources = []
    for cls in ALL_CONNECTORS:
        src, _ = build_source(cls)
        if src is not None:
            sources.append(src)
    return sources


def source_overview() -> List[dict]:
    """Configuration + last sync state for every known connector."""
    states = db.get_sync_states()
    overview = []
    for cls in ALL_CONNECTORS:
        src, origin = build_source(cls)
        overview.append({
            "name": cls.name,
            "configured": src is not None,
            "config_origin": origin,
            "last_sync": states.get(cls.name),
        })
    return overview


def _purge_chunks(filename: str) -> None:
    """Remove a document's chunks from ORG-DOCS (best-effort)."""
    settings = get_settings()
    try:
        from rag.collections import COLLECTION_ORG_DOCS, delete_documents_by_metadata
        delete_documents_by_metadata(
            COLLECTION_ORG_DOCS,
            where={"$and": [
                {"org_id": {"$eq": settings.ORG_ID}},
                {"source": {"$eq": filename}},
            ]},
        )
    except Exception as exc:
        logger.warning(f"Chunk purge failed for '{filename}': {exc}")


def _index(content: str, filename: str) -> None:
    settings = get_settings()
    try:
        from rag.indexer import index_text_as_org_doc
        index_text_as_org_doc(content, filename, settings.ORG_ID)
    except Exception as exc:
        logger.warning(f"Indexing failed for '{filename}': {exc}")


def sync_source(source: DocumentSource) -> dict:
    """Run one incremental sync for a single source. Returns a summary."""
    existing = {d["external_id"]: d for d in db.get_synced_documents(source.name)}
    remote = source.list_remote()

    added = updated = unchanged = removed = 0
    errors: List[str] = []
    seen = set()

    for ref in remote:
        seen.add(ref.external_id)
        current = existing.get(ref.external_id)

        if current and current["version_hash"] == ref.version_hash:
            unchanged += 1
            continue

        try:
            content = source.fetch_content(ref)
        except Exception as exc:
            errors.append(f"{ref.external_id}: {exc}")
            logger.warning(f"[{source.name}] fetch failed for {ref.external_id}: {exc}")
            continue

        if not content.strip():
            logger.info(f"[{source.name}] empty content, skipping {ref.external_id}")
            continue

        if current:
            # Purge chunks under the OLD filename (it may have changed)
            _purge_chunks(current["filename"])
            db.update_synced_document(current["id"], ref.filename, content, ref.version_hash)
            updated += 1
        else:
            db.add_document(
                ref.filename,
                uploader=f"sync:{source.name}",
                content=content,
                source=source.name,
                external_id=ref.external_id,
                version_hash=ref.version_hash,
            )
            added += 1

        _index(content, ref.filename)

    # Documents deleted upstream disappear from the corpus too
    for external_id, current in existing.items():
        if external_id not in seen:
            _purge_chunks(current["filename"])
            db.delete_document(current["id"])
            removed += 1

    summary = {
        "added": added,
        "updated": updated,
        "unchanged": unchanged,
        "removed": removed,
        "errors": errors[:10],
    }
    logger.info(f"[{source.name}] sync completed: {summary}")
    return summary


async def run_full_sync() -> None:
    """Sync every configured source (background task)."""
    sources = configured_sources()
    if not sources:
        logger.info("No external sources configured, nothing to sync")
        return

    for source in sources:
        db.set_sync_state(source.name, "RUNNING", "")
        try:
            # Connectors are synchronous httpx code — run off the event loop
            summary = await asyncio.to_thread(sync_source, source)
            status = "OK" if not summary["errors"] else "PARTIAL"
            db.set_sync_state(source.name, status, json.dumps(summary))
        except Exception as exc:
            logger.error(f"[{source.name}] sync failed: {exc}", exc_info=True)
            db.set_sync_state(source.name, "ERROR", str(exc)[:500])
