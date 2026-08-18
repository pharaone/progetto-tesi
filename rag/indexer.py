"""Document indexer for ISO/IEC 42001 gap analysis system.

Handles PDF and TXT files. Uses the ONNX all-MiniLM-L6-v2 embedding
function provided by ChromaDB.

ISO documents are indexed **requirement-first**: the raw text is segmented
at every clause/control heading (one requirement = one segment) BEFORE the
generic text splitter runs, and the splitter only breaks up segments that
are still longer than CHUNK_SIZE. Every chunk therefore carries the
requirement_id of the heading it belongs to (e.g. "4.1 ..." → "cl-4.1",
"A.5.3 ..." → "A.5.3").

Segmenting first is what guarantees one evaluation card per requirement:
chunking first and tagging afterwards assigned a single id per chunk, so a
512-character chunk spanning several short headings (typical of the Annex A
control table) silently swallowed all but the first requirement — those
requirements never became a group in requirements_loader and were never
evaluated.

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

# Every ISO heading that can start a segment, matched at the start of a line.
# Leading whitespace is allowed because PDF-extracted text often indents
# headings slightly. Alternatives are ordered longest-first so that "A.6.2.3"
# is not truncated to "A.6.2" and "6.1.2" is not truncated to "6.1".
_HEADING = re.compile(
    r"^[ \t]*"
    r"(?P<num>"
    r"A\.\d{1,2}(?:\.\d{1,2}){1,2}"       # A.4.3 / A.6.2.3  → Annex A control
    r"|A\.\d{1,2}"                        # A.4              → Annex A section title
    r"|(?:10|[4-9])\.\d{1,2}(?:\.\d{1,2})?"  # 8.3 / 6.1.2   → clause requirement
    r"|(?:10|[4-9])"                      # 5                → clause title
    r")"
    r"(?=[ \t]|$)"
    r"(?P<rest>[^\n]*)",
    re.MULTILINE,
)

# Table-of-contents lines use long dot leaders ("5.1 Leadership.......... 42").
# They match the heading regex and would create bogus requirement segments
# full of TOC junk — strip them before segmenting.
_TOC_LINE = re.compile(r"^.*\.{5,}.*$", re.MULTILINE)


def _strip_toc_lines(text: str) -> str:
    """Remove table-of-contents lines (long dot leaders) from extracted text."""
    stripped = _TOC_LINE.sub("", text)
    # Collapse the blank gaps left behind
    return re.sub(r"\n{3,}", "\n\n", stripped)


def _looks_like_section_title(rest: str) -> bool:
    """Heuristic: does the text after a bare number look like a section title?

    Guards the bare-number alternatives ("5 Leadership", "A.4 Resources")
    against false positives such as page numbers on their own line or a
    sentence that happens to start with a figure.
    """
    title = rest.strip()
    if not title or len(title) > 80:
        return False
    if title[-1] in ".;,:":
        return False
    return title[0].isalpha()


def _classify_heading(num: str, rest: str) -> Optional[str]:
    """Map a heading match to the requirement it opens.

    Returns the requirement_id, "" for a section boundary that is not itself
    a requirement (a clause/annex title), or None when the match should be
    ignored altogether.
    """
    if num.startswith("A."):
        # A.x.y[.z] is a control; a bare A.x is just the family title
        if num.count(".") >= 2:
            return num
        return "" if _looks_like_section_title(rest) else None

    if "." in num:
        return f"cl-{num}"

    # Bare clause number: a title line such as "5 Leadership"
    return "" if _looks_like_section_title(rest) else None


def segment_by_requirement(text: str) -> List[Tuple[str, str]]:
    """Split raw ISO text into (requirement_id, segment_text) pairs.

    Each segment runs from one heading to the next, so a requirement's text
    is never merged into the previous one. Text before the first heading is
    kept as a segment with an empty requirement_id.
    """
    boundaries: List[Tuple[int, str]] = []
    for match in _HEADING.finditer(text):
        requirement_id = _classify_heading(match.group("num"), match.group("rest"))
        if requirement_id is None:
            continue
        boundaries.append((match.start(), requirement_id))

    segments: List[Tuple[str, str]] = []

    preamble = text[: boundaries[0][0]] if boundaries else text
    if preamble.strip():
        segments.append(("", preamble))

    for i, (start, requirement_id) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        segment = text[start:end]
        if segment.strip():
            segments.append((requirement_id, segment))

    return segments


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

    # Requirement-first: cut at every heading, then split only the segments
    # that are still too long for a single embedding
    segments = segment_by_requirement(text)
    splitter = _get_splitter()
    filename = Path(file_path).name

    chunks: List[str] = []
    metadatas: List[dict] = []

    for requirement_id, segment in segments:
        parts = splitter.split_text(segment) if len(segment) > CHUNK_SIZE else [segment]
        for part in parts:
            if not part.strip():
                continue
            metadatas.append(
                {
                    "source": filename,
                    # Global position: keeps a requirement's parts in document
                    # order when requirements_loader regroups them
                    "chunk_index": len(chunks),
                    "requirement_id": requirement_id,
                    "collection": collection_name,
                    "type": "iso",
                }
            )
            chunks.append(part)

    if not chunks:
        logger.warning(f"No indexable content in {filename}")
        return 0

    for meta in metadatas:
        meta["total_chunks"] = len(chunks)

    ids = [
        _make_chunk_id(file_path, i, prefix="iso_")
        for i in range(len(chunks))
    ]

    add_documents(collection_name, chunks, metadatas, ids)

    requirements = {m["requirement_id"] for m in metadatas if m["requirement_id"]}
    logger.info(
        f"Indexed {len(chunks)} chunks covering {len(requirements)} requirements "
        f"from {filename} into '{collection_name}'"
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
