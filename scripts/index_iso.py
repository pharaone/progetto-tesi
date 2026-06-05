#!/usr/bin/env python3
"""Script to index ISO 42001 documents into ChromaDB collections.

Usage:
    python scripts/index_iso.py --docs-dir /path/to/iso/docs [--dry-run]

The script assigns documents to collections based on filename patterns:
    - *cl4* | *cl5* | *cl6* | *clause4* | *clause5* | *clause6* → ISO-CL456
    - *cl7* | *cl8* | *annex_a2* .. *annex_a6* → ISO-CL78-A26
    - *cl9* | *cl10* | *annex_a7* .. *annex_a10* → ISO-CL910-A710
    - Everything else (or *full* | *complete*) → ISO-FULL
    - All documents also go into ISO-FULL
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from rag.collections import (
    COLLECTION_ISO_CL456,
    COLLECTION_ISO_CL78_A26,
    COLLECTION_ISO_CL910_A710,
    COLLECTION_ISO_FULL,
    initialize_collections,
)
from rag.indexer import index_iso_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".rst", ".text"}


def _determine_collections(filename: str) -> list[str]:
    """
    Determine which collections a file should be indexed into based on its name.

    Returns a list of collection names. All documents also go into ISO-FULL.
    """
    name = filename.lower()
    collections = set()

    # Pattern matching for clause/annex assignment
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

    # If no specific collection matched, it still goes to ISO-FULL
    return list(collections)


def index_directory(docs_dir: str, dry_run: bool = False) -> dict:
    """
    Index all ISO documents in the given directory.

    Args:
        docs_dir: Directory containing ISO documents.
        dry_run: If True, only print what would be indexed.

    Returns:
        Summary dict with counts per collection.
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Directory not found: {docs_dir}")
    if not docs_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {docs_dir}")

    # Find all supported files
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(docs_path.rglob(f"*{ext}"))

    if not files:
        logger.warning(f"No supported documents found in {docs_dir}")
        logger.warning(f"Supported extensions: {SUPPORTED_EXTENSIONS}")
        return {}

    logger.info(f"Found {len(files)} document(s) in {docs_dir}")

    if not dry_run:
        logger.info("Initializing ChromaDB collections...")
        initialize_collections()

    summary: dict[str, int] = {
        COLLECTION_ISO_CL456: 0,
        COLLECTION_ISO_CL78_A26: 0,
        COLLECTION_ISO_CL910_A710: 0,
        COLLECTION_ISO_FULL: 0,
    }

    for file_path in sorted(files):
        collections = _determine_collections(file_path.name)
        logger.info(
            f"File: {file_path.name} → collections: {', '.join(collections)}"
        )

        if dry_run:
            logger.info(f"  [DRY RUN] Would index {file_path.name} into: {collections}")
            continue

        for collection_name in collections:
            try:
                chunks_indexed = index_iso_document(str(file_path), collection_name)
                summary[collection_name] = summary.get(collection_name, 0) + chunks_indexed
                logger.info(
                    f"  Indexed {chunks_indexed} chunks into {collection_name}"
                )
            except Exception as exc:
                logger.error(
                    f"  Failed to index {file_path.name} into {collection_name}: {exc}",
                    exc_info=True,
                )

    if not dry_run:
        logger.info("\nIndexing complete. Summary:")
        for collection, count in summary.items():
            logger.info(f"  {collection}: {count} chunks")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index ISO 42001 documents into ChromaDB collections"
    )
    parser.add_argument(
        "--docs-dir",
        required=True,
        help="Directory containing ISO 42001 documents (PDF and TXT files)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be indexed without actually indexing",
    )
    parser.add_argument(
        "--chromadb-path",
        default=None,
        help="Override CHROMADB_PATH environment variable",
    )

    args = parser.parse_args()

    if args.chromadb_path:
        os.environ["CHROMADB_PATH"] = args.chromadb_path

    chromadb_path = os.getenv("CHROMADB_PATH", "/data/chromadb")
    logger.info(f"ChromaDB path: {chromadb_path}")
    logger.info(f"Indexing documents from: {args.docs_dir}")
    if args.dry_run:
        logger.info("DRY RUN MODE — no documents will be indexed")

    try:
        summary = index_directory(args.docs_dir, dry_run=args.dry_run)

        if not args.dry_run:
            print("\nIndexing Summary:")
            print("-" * 50)
            total_chunks = 0
            for collection, count in summary.items():
                print(f"  {collection:<25} {count:>8} chunks")
                total_chunks += count
            print("-" * 50)
            print(f"  {'TOTAL':<25} {total_chunks:>8} chunks")
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Indexing failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
