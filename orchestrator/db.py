"""SQLite persistence for users, documents, and gap reports.

Single-company deployment: every document and report belongs to the one
company this instance serves (ORG_ID env var), so no org scoping is stored —
documents are scoped by uploader (employee) instead.

Report lifecycle (async job pattern — /analyze returns immediately):
    RUNNING         →  PENDING_REVIEW  →  APPROVED  (visible to employees)
                    →  FAILED                       →  REJECTED
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from shared.auth import ROLE_CERTIFIER, ROLE_EMPLOYEE, hash_password

logger = logging.getLogger(__name__)

STATUS_RUNNING = "RUNNING"
STATUS_FAILED = "FAILED"
STATUS_PENDING = "PENDING_REVIEW"
STATUS_APPROVED = "APPROVED"
STATUS_REJECTED = "REJECTED"

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _db_path() -> str:
    return os.getenv("APP_DB_PATH", "/data/app/app.db")


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        path = _db_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _conn = sqlite3.connect(path, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
    return _conn


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def init_db() -> None:
    """Create tables and seed the certifier account from env vars."""
    with _lock:
        conn = _get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                username    TEXT PRIMARY KEY,
                password    TEXT NOT NULL,
                role        TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT NOT NULL,
                uploader    TEXT NOT NULL,
                content     TEXT NOT NULL,
                uploaded_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reports (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                status          TEXT NOT NULL,
                report_json     TEXT NOT NULL,
                created_by      TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                reviewed_by     TEXT,
                reviewed_at     TEXT,
                review_comment  TEXT
            );
            """
        )
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sync_state (
                source        TEXT PRIMARY KEY,
                status        TEXT NOT NULL,
                detail        TEXT,
                last_sync_at  TEXT
            );
            CREATE TABLE IF NOT EXISTS source_config (
                source       TEXT PRIMARY KEY,
                config_json  TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );
            """
        )
        # Idempotent migrations for databases created by earlier versions
        for ddl in (
            "ALTER TABLE reports ADD COLUMN error TEXT",
            "ALTER TABLE documents ADD COLUMN source TEXT NOT NULL DEFAULT 'upload'",
            "ALTER TABLE documents ADD COLUMN external_id TEXT",
            "ALTER TABLE documents ADD COLUMN version_hash TEXT",
        ):
            try:
                conn.execute(ddl)
            except sqlite3.OperationalError:
                pass  # column already exists
        conn.commit()

    # Seed the certifier account (idempotent)
    certifier_user = os.getenv("CERTIFIER_USERNAME", "certifier")
    certifier_pass = os.getenv("CERTIFIER_PASSWORD", "")
    if certifier_pass:
        if get_user(certifier_user) is None:
            create_user(certifier_user, certifier_pass, ROLE_CERTIFIER)
            logger.info(f"Seeded certifier account '{certifier_user}'")
    else:
        logger.warning(
            "CERTIFIER_PASSWORD not set — no certifier account seeded. "
            "Set CERTIFIER_USERNAME/CERTIFIER_PASSWORD in .env."
        )


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def get_user(username: str) -> Optional[Dict[str, Any]]:
    with _lock:
        row = _get_conn().execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return dict(row) if row else None


def create_user(username: str, password: str, role: str = ROLE_EMPLOYEE) -> Dict[str, Any]:
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO users (username, password, role, created_at) VALUES (?, ?, ?, ?)",
            (username, hash_password(password), role, _now()),
        )
        conn.commit()
    return {"username": username, "role": role}


def list_users() -> List[Dict[str, Any]]:
    with _lock:
        rows = _get_conn().execute(
            "SELECT username, role, created_at FROM users ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


def update_user_role(username: str, role: str) -> bool:
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            "UPDATE users SET role = ? WHERE username = ?", (role, username)
        )
        conn.commit()
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def add_document(
    filename: str,
    uploader: str,
    content: str,
    source: str = "upload",
    external_id: Optional[str] = None,
    version_hash: Optional[str] = None,
) -> int:
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            "INSERT INTO documents (filename, uploader, content, uploaded_at, source, external_id, version_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (filename, uploader, content, _now(), source, external_id, version_hash),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_documents(uploader: Optional[str] = None) -> List[Dict[str, Any]]:
    """List document metadata.

    Employees (uploader given) see their own uploads PLUS all synced
    documents (company-wide by design — they have no individual owner).
    Certifier (uploader=None) sees everything.
    """
    sql = "SELECT id, filename, uploader, uploaded_at, source FROM documents"
    params: tuple = ()
    if uploader is not None:
        sql += " WHERE uploader = ? OR source != 'upload'"
        params = (uploader,)
    sql += " ORDER BY uploaded_at DESC"
    with _lock:
        rows = _get_conn().execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_all_documents_with_content() -> List[Dict[str, Any]]:
    """All documents including content — used to run the company-wide analysis."""
    with _lock:
        rows = _get_conn().execute(
            "SELECT id, filename, uploader, content, uploaded_at FROM documents ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def get_document(doc_id: int) -> Optional[Dict[str, Any]]:
    with _lock:
        row = _get_conn().execute(
            "SELECT id, filename, uploader, uploaded_at FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()
    return dict(row) if row else None


def delete_document(doc_id: int, uploader: Optional[str] = None) -> bool:
    """Delete a document. If uploader is given, only delete if they own it."""
    sql = "DELETE FROM documents WHERE id = ?"
    params: list = [doc_id]
    if uploader is not None:
        sql += " AND uploader = ?"
        params.append(uploader)
    with _lock:
        conn = _get_conn()
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Synced documents (external sources: GitHub, Confluence, ...)
# ---------------------------------------------------------------------------

def get_synced_documents(source: str) -> List[Dict[str, Any]]:
    """All documents previously synced from a given external source."""
    with _lock:
        rows = _get_conn().execute(
            "SELECT id, filename, external_id, version_hash FROM documents WHERE source = ?",
            (source,),
        ).fetchall()
    return [dict(r) for r in rows]


def update_synced_document(
    doc_id: int, filename: str, content: str, version_hash: str
) -> None:
    with _lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE documents SET filename = ?, content = ?, version_hash = ?, uploaded_at = ? WHERE id = ?",
            (filename, content, version_hash, _now(), doc_id),
        )
        conn.commit()


def set_sync_state(source: str, status: str, detail: str = "") -> None:
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO sync_state (source, status, detail, last_sync_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(source) DO UPDATE SET status = ?, detail = ?, last_sync_at = ?",
            (source, status, detail, _now(), status, detail, _now()),
        )
        conn.commit()


def get_sync_states() -> Dict[str, Dict[str, Any]]:
    with _lock:
        rows = _get_conn().execute("SELECT * FROM sync_state").fetchall()
    return {r["source"]: dict(r) for r in rows}


def get_source_config(source: str) -> Optional[Dict[str, Any]]:
    with _lock:
        row = _get_conn().execute(
            "SELECT config_json FROM source_config WHERE source = ?", (source,)
        ).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row["config_json"])
    except json.JSONDecodeError:
        return None


def set_source_config(source: str, config: Dict[str, Any]) -> None:
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO source_config (source, config_json, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(source) DO UPDATE SET config_json = ?, updated_at = ?",
            (source, json.dumps(config), _now(), json.dumps(config), _now()),
        )
        conn.commit()


def delete_source_config(source: str) -> bool:
    with _lock:
        conn = _get_conn()
        cur = conn.execute("DELETE FROM source_config WHERE source = ?", (source,))
        conn.commit()
        return cur.rowcount > 0


def is_sync_running() -> bool:
    with _lock:
        row = _get_conn().execute(
            "SELECT 1 FROM sync_state WHERE status = 'RUNNING' LIMIT 1"
        ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def create_running_report(created_by: str) -> int:
    """Create a report row in RUNNING state; the pipeline fills it in later."""
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            "INSERT INTO reports (status, report_json, created_by, created_at) VALUES (?, ?, ?, ?)",
            (STATUS_RUNNING, "{}", created_by, _now()),
        )
        conn.commit()
        return int(cur.lastrowid)


def complete_report(report_id: int, report: Dict[str, Any]) -> None:
    """Store the pipeline output and move the report to PENDING_REVIEW."""
    with _lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE reports SET status = ?, report_json = ? WHERE id = ? AND status = ?",
            (STATUS_PENDING, json.dumps(report), report_id, STATUS_RUNNING),
        )
        conn.commit()


def fail_report(report_id: int, error: str) -> None:
    """Mark a RUNNING report as FAILED with the error message."""
    with _lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE reports SET status = ?, error = ? WHERE id = ? AND status = ?",
            (STATUS_FAILED, error[:1000], report_id, STATUS_RUNNING),
        )
        conn.commit()


def fail_stale_running_reports() -> int:
    """Mark leftover RUNNING reports as FAILED (called at startup: a RUNNING
    row surviving a restart means the pipeline died with the process)."""
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            "UPDATE reports SET status = ?, error = ? WHERE status = ?",
            (STATUS_FAILED, "Interrotta dal riavvio del sistema", STATUS_RUNNING),
        )
        conn.commit()
        return cur.rowcount


def has_running_report() -> bool:
    with _lock:
        row = _get_conn().execute(
            "SELECT 1 FROM reports WHERE status = ? LIMIT 1", (STATUS_RUNNING,)
        ).fetchone()
    return row is not None


def get_latest_report_status() -> Optional[Dict[str, Any]]:
    """Lightweight status of the most recent report (no content) — safe to
    expose to employees so they can track analysis progress."""
    with _lock:
        row = _get_conn().execute(
            "SELECT id, status, created_by, created_at, error FROM reports "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def _report_summary(row: sqlite3.Row) -> Dict[str, Any]:
    try:
        report = json.loads(row["report_json"] or "{}")
    except json.JSONDecodeError:
        report = {}
    keys = row.keys()
    return {
        "id": row["id"],
        "status": row["status"],
        "created_by": row["created_by"],
        "created_at": row["created_at"],
        "reviewed_by": row["reviewed_by"],
        "reviewed_at": row["reviewed_at"],
        "review_comment": row["review_comment"],
        "error": row["error"] if "error" in keys else None,
        "overall_compliance_score": report.get("overall_compliance_score"),
        "total_requirements": report.get("total_requirements"),
    }


def list_reports(only_status: Optional[str] = None) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM reports"
    params: tuple = ()
    if only_status:
        sql += " WHERE status = ?"
        params = (only_status,)
    sql += " ORDER BY created_at DESC"
    with _lock:
        rows = _get_conn().execute(sql, params).fetchall()
    return [_report_summary(r) for r in rows]


def get_report(report_id: int) -> Optional[Dict[str, Any]]:
    with _lock:
        row = _get_conn().execute(
            "SELECT * FROM reports WHERE id = ?", (report_id,)
        ).fetchone()
    if row is None:
        return None
    summary = _report_summary(row)
    summary["report"] = json.loads(row["report_json"])
    return summary


def review_report(
    report_id: int,
    reviewer: str,
    approve: bool,
    comment: str = "",
) -> bool:
    """Certifier approves or rejects a pending report."""
    new_status = STATUS_APPROVED if approve else STATUS_REJECTED
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            """UPDATE reports
               SET status = ?, reviewed_by = ?, reviewed_at = ?, review_comment = ?
               WHERE id = ? AND status = ?""",
            (new_status, reviewer, _now(), comment, report_id, STATUS_PENDING),
        )
        conn.commit()
        return cur.rowcount > 0
