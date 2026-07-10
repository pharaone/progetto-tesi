"""SQLite persistence for users, documents, and gap reports.

Single-company deployment: every document and report belongs to the one
company this instance serves (ORG_ID env var), so no org scoping is stored —
documents are scoped by uploader (employee) instead.

Report lifecycle:
    PENDING_REVIEW  →  APPROVED  (certifier approves, employees can see it)
                    →  REJECTED  (certifier rejects with a comment)
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


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def add_document(filename: str, uploader: str, content: str) -> int:
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            "INSERT INTO documents (filename, uploader, content, uploaded_at) VALUES (?, ?, ?, ?)",
            (filename, uploader, content, _now()),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_documents(uploader: Optional[str] = None) -> List[Dict[str, Any]]:
    """List document metadata. Filter by uploader for employees; None = all (certifier)."""
    sql = "SELECT id, filename, uploader, uploaded_at FROM documents"
    params: tuple = ()
    if uploader is not None:
        sql += " WHERE uploader = ?"
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
# Reports
# ---------------------------------------------------------------------------

def create_report(report: Dict[str, Any], created_by: str) -> int:
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            "INSERT INTO reports (status, report_json, created_by, created_at) VALUES (?, ?, ?, ?)",
            (STATUS_PENDING, json.dumps(report), created_by, _now()),
        )
        conn.commit()
        return int(cur.lastrowid)


def _report_summary(row: sqlite3.Row) -> Dict[str, Any]:
    report = json.loads(row["report_json"])
    return {
        "id": row["id"],
        "status": row["status"],
        "created_by": row["created_by"],
        "created_at": row["created_at"],
        "reviewed_by": row["reviewed_by"],
        "reviewed_at": row["reviewed_at"],
        "review_comment": row["review_comment"],
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
