# ISO Documents

Place the ISO/IEC 42001:2023 standard here as a **.txt** (or .pdf/.md) file,
e.g. `iso42001.txt`.

This directory is mounted read-only into the AS-1 container at
`/app/iso_docs`. On startup, AS-1 automatically chunks and indexes any file
found here into the ChromaDB collections — **no manual step needed** when
spinning up a fresh company instance.

Auto-indexing only runs when the `ISO-FULL` collection is empty (first boot
or after `docker compose down -v`). To force a re-index after changing the
file:

```bash
docker exec as1 python scripts/index_iso.py --docs-dir /app/iso_docs --reset
```

Files in this directory are **git-ignored** (the ISO standard is licensed
material and must never be committed) — only this README is tracked.
