"""GitHub repository connector.

Syncs text documents (.md, .txt, .rst, .pdf) from a repository into the
company documentation corpus.

Environment variables:
    GITHUB_TOKEN    fine-grained PAT with read access to the repository
    GITHUB_REPO     "owner/name"
    GITHUB_BRANCH   branch to read (default: repository default branch)
    GITHUB_PATH     optional path prefix filter (e.g. "docs/")
"""

from __future__ import annotations

import base64
import logging
import os
from typing import List, Optional

import httpx

from connectors.base import DocumentSource, RemoteDoc

logger = logging.getLogger(__name__)

_API = "https://api.github.com"
_SUPPORTED_EXT = (".md", ".txt", ".rst", ".text", ".pdf")


class GitHubSource(DocumentSource):
    name = "github"

    CONFIG_FIELDS = [
        {"key": "token", "label": "Personal Access Token (contenuti: sola lettura)", "secret": True, "required": True},
        {"key": "repo", "label": "Repository (owner/nome)", "secret": False, "required": True},
        {"key": "branch", "label": "Branch (vuoto = default del repo)", "secret": False, "required": False},
        {"key": "path_prefix", "label": "Percorso da sincronizzare (es. docs/)", "secret": False, "required": False},
    ]

    def __init__(self, token: str, repo: str, branch: str = "", path_prefix: str = ""):
        self.token = token
        self.repo = repo
        self.branch = branch
        self.path_prefix = path_prefix.lstrip("/")

    @classmethod
    def from_env(cls) -> Optional["GitHubSource"]:
        token = os.getenv("GITHUB_TOKEN", "").strip()
        repo = os.getenv("GITHUB_REPO", "").strip()
        if not token or not repo:
            return None
        return cls(
            token=token,
            repo=repo,
            branch=os.getenv("GITHUB_BRANCH", "").strip(),
            path_prefix=os.getenv("GITHUB_PATH", "").strip(),
        )

    @classmethod
    def from_config(cls, config: dict) -> Optional["GitHubSource"]:
        token = (config.get("token") or "").strip()
        repo = (config.get("repo") or "").strip()
        if not token or not repo:
            return None
        return cls(
            token=token,
            repo=repo,
            branch=(config.get("branch") or "").strip(),
            path_prefix=(config.get("path_prefix") or "").strip(),
        )

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _resolve_branch(self, client: httpx.Client) -> str:
        if self.branch:
            return self.branch
        resp = client.get(f"{_API}/repos/{self.repo}", headers=self._headers())
        resp.raise_for_status()
        return resp.json().get("default_branch", "main")

    def list_remote(self) -> List[RemoteDoc]:
        with httpx.Client(timeout=30.0) as client:
            branch = self._resolve_branch(client)
            resp = client.get(
                f"{_API}/repos/{self.repo}/git/trees/{branch}",
                params={"recursive": "1"},
                headers=self._headers(),
            )
            # GitHub answers 409 Conflict for a repository with no commits
            if resp.status_code == 409:
                logger.warning(f"GitHub {self.repo}: repository is empty, nothing to sync")
                return []
            if resp.status_code == 404:
                raise RuntimeError(
                    f"Branch '{branch}' non trovato nel repository {self.repo} "
                    f"(o il token non ha accesso ai contenuti)"
                )
            resp.raise_for_status()
            tree = resp.json().get("tree", [])

        docs = []
        for entry in tree:
            if entry.get("type") != "blob":
                continue
            path = entry.get("path", "")
            if not path.lower().endswith(_SUPPORTED_EXT):
                continue
            if self.path_prefix and not path.startswith(self.path_prefix):
                continue
            # Flatten the path into a readable, stable filename
            flat = path.replace("/", "__")
            docs.append(
                RemoteDoc(
                    external_id=path,
                    filename=f"github__{flat}",
                    version_hash=entry.get("sha", ""),
                )
            )
        logger.info(f"GitHub {self.repo}: {len(docs)} syncable document(s)")
        return docs

    def fetch_content(self, ref: RemoteDoc) -> str:
        with httpx.Client(timeout=60.0) as client:
            branch = self._resolve_branch(client)
            resp = client.get(
                f"{_API}/repos/{self.repo}/contents/{ref.external_id}",
                params={"ref": branch},
                headers=self._headers(),
            )
            resp.raise_for_status()
            payload = resp.json()

        raw = base64.b64decode(payload.get("content", ""))

        if ref.external_id.lower().endswith(".pdf"):
            import io
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(raw), strict=False)
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n\n".join(pages)

        return raw.decode("utf-8", errors="replace")
