"""Connector abstraction for external documentation sources.

Sync/ingestion model: connectors list what exists remotely (with a cheap
version hash) and fetch content only for new/changed documents. The synced
documents flow through the exact same pipeline as manual uploads (documents
table + ORG-DOCS indexing), so analyses stay reproducible: the corpus is
whatever was synced, hashed as usual.

Adding a new source (GitLab, SharePoint, Notion, ...) = one new subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RemoteDoc:
    """A document as listed by a remote source (no content yet)."""
    external_id: str      # stable identity within the source (path, page id)
    filename: str         # logical filename used for storage/indexing
    version_hash: str     # cheap change indicator (blob SHA, page version)


class DocumentSource(ABC):
    """A remote documentation source that can be synced."""

    name: str = "base"

    @classmethod
    @abstractmethod
    def from_env(cls) -> Optional["DocumentSource"]:
        """Build the connector from environment variables.

        Returns None when the source is not configured.
        """

    @abstractmethod
    def list_remote(self) -> List[RemoteDoc]:
        """List all remote documents with their version hashes."""

    @abstractmethod
    def fetch_content(self, ref: RemoteDoc) -> str:
        """Download and extract the text content of one document."""
