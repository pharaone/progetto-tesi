"""Orchestrator — FastAPI entry point.

Endpoints:
- GET /health
- POST /analyze  — full pipeline (multipart form: org_id + files)
- POST /chat     — proxy to AIU with session context
"""

from __future__ import annotations

import logging
import os
import sys
from typing import List, Optional

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import get_settings
from shared.models import (
    ChatRequest,
    ChatResponse,
    GapReport,
    HealthResponse,
)
from orchestrator.graph import run_analysis_pipeline

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ISO/IEC 42001 Gap Analysis Orchestrator",
    version="1.0.0",
    description="Orchestrates the multi-agent ISO 42001 gap analysis pipeline",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _read_file_content(upload_file: UploadFile) -> str:
    """Read content from an uploaded file, handling PDF and text."""
    content_bytes = await upload_file.read()
    filename = upload_file.filename or "unknown"

    if filename.lower().endswith(".pdf"):
        try:
            import io
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(content_bytes))
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


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="orchestrator")


@app.post("/analyze")
async def analyze(
    org_id: str = Form(..., description="Organization identifier"),
    files: List[UploadFile] = File(..., description="Organizational documents"),
) -> JSONResponse:
    """
    Run the full ISO 42001 gap analysis pipeline.

    Steps:
    1. Read and decode uploaded files
    2. Index documents into ORG-DOCS ChromaDB collection
    3. Fan out to AS-1, AS-2, AS-3 in parallel via LangGraph
    4. Fan in to AGA for consolidation
    5. Run AIU for initial summary
    6. Return GapReport JSON
    """
    if not org_id or not org_id.strip():
        raise HTTPException(status_code=400, detail="org_id is required")

    org_id = org_id.strip()

    if not files:
        raise HTTPException(status_code=400, detail="At least one document file is required")

    # Index documents into ORG-DOCS
    from rag.indexer import index_text_as_org_doc

    documents = []
    for upload_file in files:
        filename = upload_file.filename or f"doc_{len(documents)}.txt"
        try:
            content = await _read_file_content(upload_file)
            if content.strip():
                # Index into ORG-DOCS
                try:
                    index_text_as_org_doc(content, filename, org_id)
                    logger.info(f"Indexed document: {filename} for org_id={org_id}")
                except Exception as exc:
                    logger.warning(f"Failed to index {filename}: {exc}")

                documents.append({
                    "filename": filename,
                    "content": content,
                    "metadata": {"org_id": org_id},
                })
            else:
                logger.warning(f"Empty content in file: {filename}")
        except Exception as exc:
            logger.error(f"Failed to read file {filename}: {exc}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read file '{filename}': {str(exc)}",
            )

    if not documents:
        raise HTTPException(
            status_code=400,
            detail="All uploaded files were empty or unreadable",
        )

    logger.info(
        f"Starting analysis pipeline: org_id={org_id}, documents={len(documents)}"
    )

    try:
        final_state = await run_analysis_pipeline(org_id, documents)
    except Exception as exc:
        logger.error(f"Pipeline failed for org_id={org_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline failed: {str(exc)}",
        )

    aga_report = final_state.get("aga_report")
    if aga_report is None:
        raise HTTPException(
            status_code=500,
            detail="Analysis pipeline completed but no report was generated. Check service logs.",
        )

    return JSONResponse(content=aga_report)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Proxy chat request to AIU service.

    The caller should include the gap_report in the request body so AIU
    has the full context.
    """
    settings = get_settings()

    if not request.org_id:
        raise HTTPException(status_code=400, detail="org_id is required")
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

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
