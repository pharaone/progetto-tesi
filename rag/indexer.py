"""Document indexer for ISO/IEC 42001 gap analysis system.

Handles PDF and TXT files using RecursiveCharacterTextSplitter.
Uses sentence-transformers all-MiniLM-L6-v2 via ChromaDB embedding function.

ISO documents are indexed with a requirement_id metadata field extracted
from clause/control headings (e.g. "4.1 ..." → "cl-4.1", "A.5.3 ..." → "A.5.3").
Organizational documents are indexed without requirement_id.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter

from rag.collections import (
    COLLECTION_ORG_DOCS,
    add_documents,
)

logger = logging.getLogger(__name__)

# Chunking parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Regex patterns for ISO clause/control headings.
# Allow optional leading whitespace (tabs/spaces) because PDF-extracted text
# often indents headings slightly while still placing them at the start of a line.
_ANNEX_HEADING = re.compile(r"^[ \t]*A\.(\d+)\.(\d+)(?:\.(\d+))?(?=\s|$)", re.MULTILINE)
_CLAUSE_HEADING = re.compile(r"^[ \t]*(\d{1,2})\.(\d+)(?:\.(\d+))?(?=\s|$)", re.MULTILINE)

# Table-of-contents lines use long dot leaders ("5.1 Leadership.......... 42").
# They match the heading regexes and would tag TOC chunks with requirement_ids,
# polluting requirement text with TOC junk — strip them before chunking.
_TOC_LINE = re.compile(r"^.*\.{5,}.*$", re.MULTILINE)


def _strip_toc_lines(text: str) -> str:
    """Remove table-of-contents lines (long dot leaders) from extracted text."""
    stripped = _TOC_LINE.sub("", text)
    # Collapse the blank gaps left behind
    return re.sub(r"\n{3,}", "\n\n", stripped)


def _extract_requirement_id(text: str) -> str:
    """Extract the primary ISO requirement ID from a text chunk.

    Checks for Annex A controls (A.x.y[.z]) first, then main clause
    subsections (4.1 – 10.2). Returns empty string if none found.
    """
    m = _ANNEX_HEADING.search(text)
    if m:
        a, b, c = m.group(1), m.group(2), m.group(3)
        return f"A.{a}.{b}.{c}" if c else f"A.{a}.{b}"

    m = _CLAUSE_HEADING.search(text)
    if m:
        major, minor, sub = m.group(1), m.group(2), m.group(3)
        if 4 <= int(major) <= 10:
            return f"cl-{major}.{minor}.{sub}" if sub else f"cl-{major}.{minor}"

    return ""

_text_splitter: Optional[RecursiveCharacterTextSplitter] = None


def _get_splitter() -> RecursiveCharacterTextSplitter:
    global _text_splitter
    if _text_splitter is None:
        _text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
    return _text_splitter


def _extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file, tolerating minor corruption."""
    import pypdf

    text_parts = []

    # First attempt: lenient mode (handles missing EOF markers, truncated xref)
    try:
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f, strict=False)
            for page in reader.pages:
                try:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                except Exception as page_exc:
                    logger.warning(f"Skipping page in {file_path}: {page_exc}")
        if text_parts:
            return "\n\n".join(text_parts)
        logger.warning(f"pypdf extracted no text from {file_path}")
    except Exception as exc:
        logger.warning(f"pypdf failed on {file_path}: {exc}, trying pdfminer fallback")

    # Second attempt: pdfminer (more robust for complex/scanned PDFs)
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        text = pdfminer_extract(file_path)
        if text and text.strip():
            return text
        logger.warning(f"pdfminer also extracted no text from {file_path}")
    except ImportError:
        logger.debug("pdfminer.six not installed, skipping fallback")
    except Exception as exc:
        logger.warning(f"pdfminer failed on {file_path}: {exc}")

    raise RuntimeError(
        f"Could not extract text from {file_path}. "
        "The PDF may be encrypted, scanned-only, or severely corrupted."
    )


def _extract_text(file_path: str) -> str:
    """Extract text from a file (PDF or TXT)."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_text_from_pdf(file_path)
    elif suffix in (".txt", ".md", ".rst", ".text"):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    else:
        # Try reading as text for unknown extensions
        logger.warning(
            f"Unknown file extension '{suffix}' for {file_path}, attempting text read"
        )
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()


def _chunk_text(
    text: str,
    source_path: str,
    extra_metadata: Optional[dict] = None,
) -> Tuple[List[str], List[dict]]:
    """Split text into chunks and build metadata."""
    splitter = _get_splitter()
    chunks = splitter.split_text(text)
    filename = Path(source_path).name

    metadatas = []
    for i, _ in enumerate(chunks):
        meta = {
            "source": filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        if extra_metadata:
            meta.update(extra_metadata)
        metadatas.append(meta)

    return chunks, metadatas


def _make_chunk_id(source: str, chunk_index: int, prefix: str = "") -> str:
    """Generate a deterministic chunk ID."""
    raw = f"{prefix}{source}_{chunk_index}"
    short_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{prefix}{Path(source).stem}_{chunk_index}_{short_hash}"


def index_iso_document(file_path: str, collection_name: str) -> int:
    """
    Index an ISO document into the specified ChromaDB collection.

    Args:
        file_path: Path to the ISO document (PDF or TXT).
        collection_name: Target ChromaDB collection name.

    Returns:
        Number of chunks indexed.
    """
    logger.info(f"Indexing ISO document: {file_path} → {collection_name}")
    text = _extract_text(file_path)
    if not text.strip():
        logger.warning(f"No text extracted from {file_path}")
        return 0

    text = _strip_toc_lines(text)

    chunks, metadatas = _chunk_text(
        text, file_path, extra_metadata={"collection": collection_name, "type": "iso"}
    )

    # Enrich each chunk's metadata with the requirement_id it belongs to
    for meta, chunk in zip(metadatas, chunks):
        meta["requirement_id"] = _extract_requirement_id(chunk)

    ids = [
        _make_chunk_id(file_path, i, prefix="iso_")
        for i in range(len(chunks))
    ]

    add_documents(collection_name, chunks, metadatas, ids)
    tagged = sum(1 for m in metadatas if m.get("requirement_id"))
    logger.info(
        f"Indexed {len(chunks)} chunks ({tagged} with requirement_id) "
        f"from {Path(file_path).name} into '{collection_name}'"
    )
    return len(chunks)


def index_org_docs(file_paths: List[str], org_id: str) -> int:
    """
    Index organizational documents into the ORG-DOCS collection.

    Args:
        file_paths: List of file paths to index.
        org_id: Organization identifier (used as prefix for chunk IDs).

    Returns:
        Total number of chunks indexed.
    """
    total_chunks = 0

    for file_path in file_paths:
        logger.info(f"Indexing org doc: {file_path} for org_id={org_id}")
        try:
            text = _extract_text(file_path)
            if not text.strip():
                logger.warning(f"No text extracted from {file_path}")
                continue

            chunks, metadatas = _chunk_text(
                text,
                file_path,
                extra_metadata={"org_id": org_id, "type": "org_doc"},
            )

            # Prefix IDs with org_id for multi-tenancy
            ids = [
                _make_chunk_id(file_path, i, prefix=f"{org_id}_")
                for i in range(len(chunks))
            ]

            add_documents(COLLECTION_ORG_DOCS, chunks, metadatas, ids)
            total_chunks += len(chunks)

        except Exception as exc:
            logger.error(f"Failed to index {file_path}: {exc}", exc_info=True)

    logger.info(
        f"Indexed {total_chunks} total chunks for org_id={org_id} "
        f"into '{COLLECTION_ORG_DOCS}'"
    )
    return total_chunks


def index_text_as_org_doc(
    content: str,
    filename: str,
    org_id: str,
) -> int:
    """
    Index raw text content as an organizational document.

    Args:
        content: Text content to index.
        filename: Logical filename for metadata.
        org_id: Organization identifier.

    Returns:
        Number of chunks indexed.
    """
    if not content.strip():
        logger.warning(f"Empty content for {filename}, skipping")
        return 0

    chunks, metadatas = _chunk_text(
        content,
        filename,
        extra_metadata={"org_id": org_id, "type": "org_doc"},
    )

    ids = [
        _make_chunk_id(filename, i, prefix=f"{org_id}_")
        for i in range(len(chunks))
    ]

    add_documents(COLLECTION_ORG_DOCS, chunks, metadatas, ids)
    logger.info(
        f"Indexed {len(chunks)} chunks from text '{filename}' "
        f"for org_id={org_id}"
    )
    return len(chunks)
