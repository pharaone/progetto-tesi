"""Orchestrator — FastAPI entry point.

Single-company deployment: each container group serves exactly one company
(ORG_ID env var). Authentication distinguishes two roles:

- employee:  uploads documents (sees only their own), starts the analysis,
             views APPROVED reports, chats with the AIU consultant.
- certifier: reviews PENDING_REVIEW gap reports and approves/rejects them
             before employees can see the results.

Endpoints:
- GET  /health
- POST /auth/login          — {username, password} → {token, role}
- POST /auth/register       — employee self-registration
- GET  /documents           — own docs (employee) / all docs (certifier)
- POST /documents           — upload documents (multipart)
- DELETE /documents/{id}    — delete own document
- POST /analyze             — run pipeline on ALL company documents → PENDING_REVIEW report
- GET  /reports             — APPROVED only (employee) / all (certifier)
- GET  /reports/{id}        — full report (role-checked)
- POST /reports/{id}/review — certifier approves/rejects
- POST /chat                — proxy to AIU (authenticated)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.auth import (
    ROLE_CERTIFIER,
    ROLE_EMPLOYEE,
    create_token,
    decode_token,
    verify_password,
)
from shared.config import get_settings
from shared.metrics import ANALYSES, ANALYSIS_DURATION, setup_metrics
from shared.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
)
from orchestrator import db
from orchestrator.graph import run_analysis_pipeline

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ISO/IEC 42001 Gap Analysis Orchestrator",
    version="2.0.0",
    description="Orchestrates the multi-agent ISO 42001 gap analysis pipeline",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_metrics(app, "orchestrator")


@app.on_event("startup")
async def startup() -> None:
    db.init_db()
    stale = db.fail_stale_running_reports()
    if stale:
        logger.warning(f"Marked {stale} stale RUNNING report(s) as FAILED after restart")
    # Sync jobs die with the process too
    for source, state in db.get_sync_states().items():
        if state.get("status") == "RUNNING":
            db.set_sync_state(source, "ERROR", "Interrotta dal riavvio del sistema")


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------

def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Validate the Bearer token and return the user payload."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authentication token")
    payload = decode_token(authorization[len("Bearer "):].strip())
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


def require_certifier(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if user["role"] != ROLE_CERTIFIER:
        raise HTTPException(status_code=403, detail="Certifier role required")
    return user


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6)


@app.post("/auth/login")
async def login(request: LoginRequest) -> dict:
    user = db.get_user(request.username.strip())
    if user is None or not verify_password(request.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_token(user["username"], user["role"])
    return {"token": token, "username": user["username"], "role": user["role"]}


@app.post("/auth/register")
async def register(request: RegisterRequest) -> dict:
    """Employee self-registration. The certifier account is seeded from env."""
    username = request.username.strip()
    if db.get_user(username) is not None:
        raise HTTPException(status_code=409, detail="Username already taken")
    db.create_user(username, request.password, ROLE_EMPLOYEE)
    token = create_token(username, ROLE_EMPLOYEE)
    return {"token": token, "username": username, "role": ROLE_EMPLOYEE}


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

async def _read_file_content(upload_file: UploadFile) -> str:
    """Read content from an uploaded file, handling PDF and text."""
    content_bytes = await upload_file.read()
    filename = upload_file.filename or "unknown"

    if filename.lower().endswith(".pdf"):
        try:
            import io
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(content_bytes), strict=False)
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages)
        except Exception as exc:
            logger.warning(f"PDF extraction failed for {filename}: {exc}, falling back to text")
            return content_bytes.decode("utf-8", errors="replace")
    else:
        return content_bytes.decode("utf-8", errors="replace")


@app.get("/documents")
async def get_documents(user: Dict[str, Any] = Depends(get_current_user)) -> List[dict]:
    """Employees see only their own documents; the certifier sees all."""
    if user["role"] == ROLE_CERTIFIER:
        return db.list_documents()
    return db.list_documents(uploader=user["username"])


def _index_org_doc_background(content: str, filename: str, org_id: str) -> None:
    """Index a document into ORG-DOCS — runs as a background task so the
    HTTP response doesn't wait for chunking + embedding."""
    from rag.indexer import index_text_as_org_doc
    try:
        index_text_as_org_doc(content, filename, org_id)
        logger.info(f"Background indexing completed for '{filename}'")
    except Exception as exc:
        logger.error(f"Background indexing failed for '{filename}': {exc}", exc_info=True)


@app.post("/documents")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    user: Dict[str, Any] = Depends(get_current_user),
) -> dict:
    """Store uploaded documents (owned by the uploader) and index them into RAG.

    The documents are persisted immediately; chunking and embedding run in
    the background so the UI stays responsive.
    """
    settings = get_settings()

    saved = []
    for upload_file in files:
        filename = upload_file.filename or "document.txt"
        content = await _read_file_content(upload_file)
        if not content.strip():
            logger.warning(f"Empty content in file: {filename}")
            continue

        doc_id = db.add_document(filename, user["username"], content)
        background_tasks.add_task(
            _index_org_doc_background, content, filename, settings.ORG_ID
        )
        saved.append({"id": doc_id, "filename": filename})

    if not saved:
        raise HTTPException(
            status_code=400, detail="All uploaded files were empty or unreadable"
        )
    return {"uploaded": saved, "indexing": "in_background"}


@app.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: int,
    user: Dict[str, Any] = Depends(get_current_user),
) -> dict:
    """Delete a document — employees only their own, certifier any.

    Also removes the document's indexed chunks from the ORG-DOCS RAG
    collection, so deleted content (e.g. an outdated clarification) no
    longer influences future analyses.
    """
    settings = get_settings()

    doc = db.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    uploader = None if user["role"] == ROLE_CERTIFIER else user["username"]
    if not db.delete_document(doc_id, uploader=uploader):
        raise HTTPException(status_code=404, detail="Document not found")

    # Purge the RAG chunks (best-effort — the DB row is already gone)
    try:
        from rag.collections import COLLECTION_ORG_DOCS, delete_documents_by_metadata
        removed = delete_documents_by_metadata(
            COLLECTION_ORG_DOCS,
            where={"$and": [
                {"org_id": {"$eq": settings.ORG_ID}},
                {"source": {"$eq": doc["filename"]}},
            ]},
        )
        logger.info(f"Removed {removed} RAG chunks for deleted document '{doc['filename']}'")
    except Exception as exc:
        logger.warning(f"Failed to purge RAG chunks for '{doc['filename']}': {exc}")

    return {"deleted": doc_id}


# ---------------------------------------------------------------------------
# External documentation sources (GitHub / Confluence sync)
# ---------------------------------------------------------------------------

@app.get("/sources")
async def get_sources(user: Dict[str, Any] = Depends(get_current_user)) -> List[dict]:
    """Known connectors, whether they are configured, and their last sync."""
    from orchestrator.sync import source_overview
    return source_overview()


@app.post("/sources/sync")
async def sync_sources(
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(get_current_user),
) -> dict:
    """Start an on-demand incremental sync of all configured sources.

    Runs in the background; progress and results are visible via
    GET /sources (per-source status: RUNNING / OK / PARTIAL / ERROR).
    """
    from orchestrator.sync import configured_sources, run_full_sync

    if not configured_sources():
        raise HTTPException(
            status_code=400,
            detail=(
                "No external source configured. Set GITHUB_TOKEN/GITHUB_REPO "
                "and/or CONFLUENCE_URL/CONFLUENCE_SPACE/CONFLUENCE_EMAIL/"
                "CONFLUENCE_API_TOKEN in .env."
            ),
        )
    if db.is_sync_running():
        raise HTTPException(status_code=409, detail="A sync is already running.")

    background_tasks.add_task(run_full_sync)
    logger.info(f"External sources sync started by {user['username']}")
    return {"message": "Sincronizzazione avviata in background."}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

async def _run_analysis_background(report_id: int, org_id: str, documents: list) -> None:
    """Execute the pipeline and settle the report row (async job pattern)."""
    import time as _time

    pipeline_start = _time.perf_counter()
    try:
        final_state = await run_analysis_pipeline(org_id, documents)
        aga_report = final_state.get("aga_report")
        if aga_report is None:
            raise RuntimeError(
                "Pipeline completed but no report was generated (all agents failed?)"
            )
    except Exception as exc:
        ANALYSES.labels(status="error").inc()
        logger.error(f"Pipeline failed for report {report_id}: {exc}", exc_info=True)
        db.fail_report(report_id, str(exc))
        return

    ANALYSES.labels(status="success").inc()
    ANALYSIS_DURATION.observe(_time.perf_counter() - pipeline_start)
    db.complete_report(report_id, aga_report)
    logger.info(f"Report {report_id} completed → PENDING_REVIEW")


@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """
    Start the ISO 42001 gap analysis over ALL stored company documents.

    Async job pattern: returns immediately with the report id in RUNNING
    state; the pipeline executes in the background. Track progress via
    GET /analysis/status, results appear as PENDING_REVIEW for the
    certifier and become visible to employees once APPROVED.
    """
    settings = get_settings()

    docs = db.get_all_documents_with_content()
    if not docs:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded yet. Upload company documents first.",
        )

    # One analysis at a time: the agents share a single LLM backend and a
    # concurrent run would only queue on it while doubling the wait
    if db.has_running_report():
        raise HTTPException(
            status_code=409,
            detail="An analysis is already running. Wait for it to finish.",
        )

    documents = [
        {
            "filename": d["filename"],
            "content": d["content"],
            "metadata": {"org_id": settings.ORG_ID, "uploader": d["uploader"]},
        }
        for d in docs
    ]

    report_id = db.create_running_report(created_by=user["username"])
    background_tasks.add_task(
        _run_analysis_background, report_id, settings.ORG_ID, documents
    )

    logger.info(
        f"Analysis started in background: report={report_id}, "
        f"documents={len(documents)}, requested_by={user['username']}"
    )

    return JSONResponse(
        content={
            "report_id": report_id,
            "status": db.STATUS_RUNNING,
            "message": (
                "Analisi avviata in background. Puoi continuare a usare "
                "l'applicazione: al termine il report passerà in revisione "
                "al certificatore."
            ),
        }
    )


@app.get("/analysis/status")
async def analysis_status(user: Dict[str, Any] = Depends(get_current_user)) -> dict:
    """Lightweight status of the most recent analysis (no report content) —
    visible to every authenticated user, including employees who cannot yet
    read the report itself."""
    latest = db.get_latest_report_status()
    if latest is None:
        return {"status": None}
    return latest


# ---------------------------------------------------------------------------
# Reports + review workflow
# ---------------------------------------------------------------------------

class ReviewRequest(BaseModel):
    approve: bool = Field(..., description="True to approve, False to reject")
    comment: str = Field(default="", description="Reviewer comment")


@app.get("/reports")
async def get_reports(user: Dict[str, Any] = Depends(get_current_user)) -> List[dict]:
    """Certifier sees all reports; employees only APPROVED ones."""
    if user["role"] == ROLE_CERTIFIER:
        return db.list_reports()
    return db.list_reports(only_status=db.STATUS_APPROVED)


@app.get("/reports/{report_id}")
async def get_report(
    report_id: int,
    user: Dict[str, Any] = Depends(get_current_user),
) -> dict:
    report = db.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    if user["role"] != ROLE_CERTIFIER and report["status"] != db.STATUS_APPROVED:
        raise HTTPException(
            status_code=403,
            detail="This report has not been approved by the certifier yet",
        )
    return report


@app.post("/reports/{report_id}/review")
async def review_report(
    report_id: int,
    request: ReviewRequest,
    user: Dict[str, Any] = Depends(require_certifier),
) -> dict:
    """Certifier approves or rejects a PENDING_REVIEW report."""
    report = db.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    if report["status"] != db.STATUS_PENDING:
        raise HTTPException(
            status_code=409,
            detail=f"Report is already {report['status']} — only PENDING_REVIEW reports can be reviewed",
        )

    db.review_report(report_id, user["username"], request.approve, request.comment)
    new_status = db.STATUS_APPROVED if request.approve else db.STATUS_REJECTED
    logger.info(f"Report {report_id} reviewed by {user['username']}: {new_status}")
    return {"report_id": report_id, "status": new_status}


# ---------------------------------------------------------------------------
# System memory (ORG-HISTORY): saved chats and report summaries
# ---------------------------------------------------------------------------

@app.get("/history")
async def get_history(user: Dict[str, Any] = Depends(get_current_user)) -> List[dict]:
    """
    List the entries stored in the ORG-HISTORY knowledge base: saved chat
    exchanges and gap report summaries. These entries feed the context of
    future chats (AIU) and analyses (AGA), so users can inspect what the
    system "remembers".
    """
    settings = get_settings()
    try:
        from rag.collections import COLLECTION_ORG_HISTORY, get_collection
        collection = get_collection(COLLECTION_ORG_HISTORY)
        data = collection.get(
            where={"org_id": settings.ORG_ID},
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        logger.error(f"Failed to read ORG-HISTORY: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not read system memory")

    entries = []
    for doc_id, doc, meta in zip(
        data.get("ids", []), data.get("documents", []), data.get("metadatas", [])
    ):
        meta = meta or {}
        entries.append({
            "id": doc_id,
            "type": meta.get("type", "unknown"),  # "chat" | "gap_report"
            "timestamp": meta.get("timestamp", ""),
            "text": (doc or "")[:600],
        })

    entries.sort(key=lambda e: e["timestamp"], reverse=True)
    return entries


@app.delete("/history/{entry_id}")
async def delete_history_entry(
    entry_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
) -> dict:
    """Remove a single entry from the ORG-HISTORY knowledge base."""
    settings = get_settings()
    try:
        from rag.collections import COLLECTION_ORG_HISTORY, get_collection
        collection = get_collection(COLLECTION_ORG_HISTORY)
        existing = collection.get(ids=[entry_id], include=["metadatas"])
        ids = existing.get("ids", [])
        metas = existing.get("metadatas", []) or []
        if not ids:
            raise HTTPException(status_code=404, detail="Memory entry not found")
        # Safety: never delete another organization's data through this instance
        if metas and (metas[0] or {}).get("org_id") not in (settings.ORG_ID, None):
            raise HTTPException(status_code=404, detail="Memory entry not found")
        collection.delete(ids=[entry_id])
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to delete ORG-HISTORY entry: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not delete memory entry")

    logger.info(f"ORG-HISTORY entry deleted by {user['username']}: {entry_id}")
    return {"deleted": entry_id}


# ---------------------------------------------------------------------------
# Clarifications
# ---------------------------------------------------------------------------

class ClarificationRequest(BaseModel):
    requirement_id: str = Field(..., description="Requirement the clarification refers to")
    text: str = Field(..., min_length=10, description="Explanation / additional detail")


@app.post("/reports/{report_id}/clarifications")
async def add_clarification(
    report_id: int,
    request: ClarificationRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(get_current_user),
) -> dict:
    """
    Attach an explanation to a NON_CONFORME / PARZIALMENTE_CONFORME requirement.

    The clarification is stored as a company document (owned by the author)
    and indexed into the RAG, so the NEXT analysis run takes it into account
    as organizational evidence.
    """
    settings = get_settings()

    report = db.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    if user["role"] != ROLE_CERTIFIER and report["status"] != db.STATUS_APPROVED:
        raise HTTPException(
            status_code=403,
            detail="This report has not been approved by the certifier yet",
        )

    cards = report["report"].get("evaluation_cards", [])
    card = next(
        (c for c in cards if c.get("requirement_id") == request.requirement_id), None
    )
    if card is None:
        raise HTTPException(
            status_code=404,
            detail=f"Requirement '{request.requirement_id}' not found in report {report_id}",
        )
    if card.get("verdict") not in ("NON_CONFORME", "PARZIALMENTE_CONFORME"):
        raise HTTPException(
            status_code=400,
            detail="Clarifications can only be added to NON_CONFORME or "
                   "PARZIALMENTE_CONFORME requirements",
        )

    # Build a self-contained document: requirement context + the user's
    # explanation, so semantic retrieval matches it in the next analysis.
    gaps = card.get("gaps", [])
    gaps_section = "\n".join(f"- {g}" for g in gaps) if gaps else "-"
    content = (
        f"Chiarimento / documentazione integrativa per il requisito "
        f"ISO/IEC 42001 {request.requirement_id}\n"
        f"Fornito da: {user['username']} (in risposta al report #{report_id}, "
        f"verdetto: {card.get('verdict')})\n\n"
        f"Requisito: {card.get('requirement_text', '')[:500]}\n\n"
        f"Gap identificati nell'analisi:\n{gaps_section}\n\n"
        f"Spiegazione dell'organizzazione:\n{request.text.strip()}\n"
    )

    filename = f"chiarimento_{request.requirement_id}_report{report_id}_{user['username']}.txt"
    doc_id = db.add_document(filename, user["username"], content)

    # Chunking + embedding happen in the background so the UI stays responsive
    background_tasks.add_task(
        _index_org_doc_background, content, filename, settings.ORG_ID
    )

    logger.info(
        f"Clarification added: report={report_id}, requirement={request.requirement_id}, "
        f"by={user['username']}, doc_id={doc_id}"
    )
    return {
        "document_id": doc_id,
        "filename": filename,
        "message": (
            "Chiarimento salvato tra i documenti aziendali "
            "(indicizzazione in corso in background). "
            "Sarà considerato alla prossima analisi."
        ),
    }


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user: Dict[str, Any] = Depends(get_current_user),
) -> ChatResponse:
    """Proxy chat request to AIU service (authenticated users only)."""
    settings = get_settings()

    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    # Single-company instance — the org_id is fixed server-side
    request.org_id = settings.ORG_ID

    try:
        async with httpx.AsyncClient(timeout=float(settings.AIU_TIMEOUT)) as client:
            response = await client.post(
                f"{settings.AIU_URL}/chat",
                json=request.model_dump(),
            )
            response.raise_for_status()
            return ChatResponse(**response.json())
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="AIU service timed out")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"AIU service error: {exc.response.text}",
        )
    except Exception as exc:
        logger.error(f"Chat proxy failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat service error: {str(exc)}")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="orchestrator")


@app.get("/services/health")
async def services_health() -> dict:
    """Check health of all microservices."""
    settings = get_settings()
    services = {
        "AS-1": f"{settings.AS1_URL}/health",
        "AS-2": f"{settings.AS2_URL}/health",
        "AS-3": f"{settings.AS3_URL}/health",
        "AGA": f"{settings.AGA_URL}/health",
        "AIU": f"{settings.AIU_URL}/health",
    }

    statuses = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for service, url in services.items():
            try:
                resp = await client.get(url)
                statuses[service] = {
                    "status": "ok" if resp.status_code == 200 else "error",
                    "http_status": resp.status_code,
                }
            except Exception as exc:
                statuses[service] = {"status": "unreachable", "error": str(exc)}

    return {"orchestrator": "ok", "services": statuses}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
