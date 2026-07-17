"""Confluence Cloud connector.

Syncs the pages of a Confluence space into the company documentation corpus.
Page HTML (storage format) is converted to plain text before indexing.

Environment variables:
    CONFLUENCE_URL        base URL, e.g. "https://acme.atlassian.net"
    CONFLUENCE_SPACE      space key, e.g. "COMP"
    CONFLUENCE_EMAIL      account email (basic auth username)
    CONFLUENCE_API_TOKEN  API token from id.atlassian.com
"""

from __future__ import annotations

import logging
import os
import re
from typing import List, Optional

import httpx

from connectors.base import DocumentSource, RemoteDoc

logger = logging.getLogger(__name__)


def _html_to_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n")
    except ImportError:
        # Crude fallback: strip tags with a regex
        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s{2,}", " ", text)


def _slugify(title: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", title.strip())[:60].strip("_")
    return slug or "pagina"


class ConfluenceSource(DocumentSource):
    name = "confluence"

    CONFIG_FIELDS = [
        {"key": "base_url", "label": "URL del sito (https://azienda.atlassian.net)", "secret": False, "required": True},
        {"key": "space", "label": "Chiave dello spazio (es. COMP)", "secret": False, "required": True},
        {"key": "email", "label": "Email account Atlassian", "secret": False, "required": True},
        {"key": "api_token", "label": "API token (id.atlassian.com)", "secret": True, "required": True},
    ]

    def __init__(self, base_url: str, space: str, email: str, api_token: str):
        self.base_url = base_url.rstrip("/")
        self.space = space
        self.email = email
        self.api_token = api_token

    @classmethod
    def from_env(cls) -> Optional["ConfluenceSource"]:
        url = os.getenv("CONFLUENCE_URL", "").strip()
        space = os.getenv("CONFLUENCE_SPACE", "").strip()
        email = os.getenv("CONFLUENCE_EMAIL", "").strip()
        token = os.getenv("CONFLUENCE_API_TOKEN", "").strip()
        if not (url and space and email and token):
            return None
        return cls(url, space, email, token)

    @classmethod
    def from_config(cls, config: dict) -> Optional["ConfluenceSource"]:
        url = (config.get("base_url") or "").strip()
        space = (config.get("space") or "").strip()
        email = (config.get("email") or "").strip()
        token = (config.get("api_token") or "").strip()
        if not (url and space and email and token):
            return None
        return cls(url, space, email, token)

    def _auth(self) -> tuple:
        return (self.email, self.api_token)

    def list_remote(self) -> List[RemoteDoc]:
        docs: List[RemoteDoc] = []
        start = 0
        with httpx.Client(timeout=30.0) as client:
            while True:
                resp = client.get(
                    f"{self.base_url}/wiki/rest/api/content",
                    params={
                        "spaceKey": self.space,
                        "type": "page",
                        "status": "current",
                        "expand": "version",
                        "limit": 50,
                        "start": start,
                    },
                    auth=self._auth(),
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])

                for page in results:
                    page_id = str(page.get("id", ""))
                    title = page.get("title", "pagina")
                    version = str((page.get("version") or {}).get("number", "0"))
                    docs.append(
                        RemoteDoc(
                            external_id=page_id,
                            filename=f"confluence__{page_id}__{_slugify(title)}.txt",
                            version_hash=version,
                        )
                    )

                if len(results) < data.get("limit", 50):
                    break
                start += len(results)

        logger.info(f"Confluence space {self.space}: {len(docs)} page(s)")
        return docs

    def fetch_content(self, ref: RemoteDoc) -> str:
        with httpx.Client(timeout=60.0) as client:
            resp = client.get(
                f"{self.base_url}/wiki/rest/api/content/{ref.external_id}",
                params={"expand": "body.storage"},
                auth=self._auth(),
            )
            resp.raise_for_status()
            page = resp.json()

        title = page.get("title", "")
        html = ((page.get("body") or {}).get("storage") or {}).get("value", "")
        text = _html_to_text(html)
        return f"{title}\n\n{text}".strip()
