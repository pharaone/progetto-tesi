#!/usr/bin/env python3
"""Script to index ISO 42001 documents into ChromaDB collections.

Usage:
    python scripts/index_iso.py --docs-dir /path/to/iso/docs [--dry-run] [--reset]

NOTE: AS-1 also indexes /app/iso_docs automatically at startup when the
ISO-FULL collection is empty (see rag/iso_indexing.py). This script is
for manual/forced re-indexing, e.g. after updating the ISO text file.

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
import sys

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from rag.iso_indexing import index_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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
        "--reset",
        action="store_true",
        help="Wipe the ISO collections before indexing (removes stale chunks)",
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
        summary = index_directory(args.docs_dir, dry_run=args.dry_run, reset=args.reset)

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
