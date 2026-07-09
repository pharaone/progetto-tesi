"""Simple authentication utilities: password hashing + HMAC-signed tokens.

Uses only the standard library (hashlib/hmac) — no extra dependencies.
Tokens are of the form  base64(json_payload).hex_signature  and expire
after TOKEN_TTL_SECONDS.

Roles:
- "employee":  can upload/see own documents, view APPROVED reports, chat
- "certifier": reviews gap reports before they are released to employees
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Optional

ROLE_EMPLOYEE = "employee"
ROLE_CERTIFIER = "certifier"
VALID_ROLES = {ROLE_EMPLOYEE, ROLE_CERTIFIER}

TOKEN_TTL_SECONDS = int(os.getenv("TOKEN_TTL_SECONDS", str(8 * 3600)))


def _get_secret() -> bytes:
    secret = os.getenv("SECRET_KEY", "")
    if not secret:
        # Fallback for dev — tokens won't survive a restart, which is fine
        secret = "dev-secret-change-me"
    return secret.encode("utf-8")


# ---------------------------------------------------------------------------
# Password hashing (salted SHA-256)
# ---------------------------------------------------------------------------

def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Return 'salt$hash' for storage."""
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return f"{salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, _ = stored.split("$", 1)
    except ValueError:
        return False
    return hmac.compare_digest(hash_password(password, salt), stored)


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

def create_token(username: str, role: str) -> str:
    payload = {
        "username": username,
        "role": role,
        "exp": int(time.time()) + TOKEN_TTL_SECONDS,
    }
    payload_b64 = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    signature = hmac.new(
        _get_secret(), payload_b64.encode("ascii"), hashlib.sha256
    ).hexdigest()
    return f"{payload_b64}.{signature}"


def decode_token(token: str) -> Optional[dict]:
    """Return the payload dict if the token is valid and unexpired, else None."""
    try:
        payload_b64, signature = token.rsplit(".", 1)
    except ValueError:
        return None

    expected = hmac.new(
        _get_secret(), payload_b64.encode("ascii"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None

    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")))
    except Exception:
        return None

    if payload.get("exp", 0) < time.time():
        return None
    if payload.get("role") not in VALID_ROLES:
        return None
    return payload
