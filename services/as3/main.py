"""AS-3: ISO/IEC 42001 Compliance Agent — Clauses 9, 10 + Annex A (A.7–A.10).

Evaluates organizational documents against:
- Clause 9: Performance evaluation (9.1–9.3)
- Clause 10: Improvement (10.1–10.2)
- Annex A Controls: A.7.1–A.7.5, A.8.1–A.8.4, A.9.1–A.9.2, A.10.1–A.10.3

RAG collections used: ISO-CL910-A710 + ORG-DOCS
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.config import get_llm, get_settings
from shared.metrics import REQUIREMENTS_EVALUATED, setup_metrics, track_llm_call
from shared.models import (
    AnalyzeRequest,
    CorrectiveAction,
    EvaluationCard,
    Evidence,
    ExecutionMetadata,
    HealthResponse,
    Verdict,
)
from rag.collections import (
    COLLECTION_ISO_CL910_A710,
    COLLECTION_ORG_DOCS,
    query as rag_query,
    query_iso_with_fallback,
)
from rag.requirements_loader import load_requirements_from_rag

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AS-3: ISO/IEC 42001 Clauses 9-10 + Annex A.7-A.10 Compliance Agent",
    version="1.0.0",
    description="Evaluates compliance with ISO/IEC 42001 Clauses 9 (Performance evaluation), 10 (Improvement) and Annex A controls A.7-A.10",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_metrics(app, "AS-3")

# Section prefixes that AS-3 is responsible for
_AS3_PREFIXES = ["cl-9", "cl-10", "A.7", "A.8", "A.9", "A.10"]

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
EVALUATION_PROMPT = """\
You are an ISO/IEC 42001:2023 compliance auditor. Evaluate the following ISO requirement against the provided organizational documentation.

Requirement ID: {requirement_id}
Requirement Text: {requirement_text}

Organizational Documentation Context:
{org_context}

ISO Standard Context:
{iso_context}

Return ONLY a valid JSON object with these exact fields (no markdown, no extra text):
{{
  "requirement_id": "{requirement_id}",
  "requirement_text": "<the full requirement text>",
  "verdict": "<one of: CONFORME, NON_CONFORME, PARZIALMENTE_CONFORME, NON_APPLICABILE>",
  "evidences": [
    {{"chunk_id": "<id>", "source_doc": "<filename>", "excerpt": "<relevant excerpt>"}}
  ],
  "gaps": ["<gap description>"],
  "corrective_action": {{
    "description": "<what the organization should do>",
    "expected_document_type": "<type of document needed>"
  }},
  "execution_metadata": {{
    "timestamp": "<ISO8601>",
    "model_version": "<model>",
    "input_hash": "<sha256>"
  }}
}}

Rules:
- If organizational documentation clearly addresses the requirement: verdict = CONFORME
- If there is partial evidence or incomplete documentation: verdict = PARZIALMENTE_CONFORME
- If no relevant documentation found or requirement clearly not met: verdict = NON_CONFORME
- If the requirement is not applicable to this organization's context: verdict = NON_APPLICABILE
- evidences must list actual excerpts from the provided context
- If no relevant documentation found, set verdict = NON_APPLICABILE and note "evidences_insufficient" in gaps
- Return ONLY the JSON object, nothing else
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _format_rag_results(results: Dict[str, Any]) -> str:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]

    if not docs:
        return "No relevant documents found."

    parts = []
    for i, (doc, meta, doc_id) in enumerate(zip(docs, metas, ids), start=1):
        source = meta.get("source", "unknown") if meta else "unknown"
        parts.append(f"[{i}] Source: {source} | ID: {doc_id}\n{doc[:800]}")

    return "\n\n---\n\n".join(parts)


def _build_evidences(results: Dict[str, Any]) -> List[Evidence]:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]

    evidences = []
    for doc, meta, doc_id in zip(docs, metas, ids):
        source = meta.get("source", "unknown") if meta else "unknown"
        evidences.append(
            Evidence(chunk_id=doc_id, source_doc=source, excerpt=doc[:300] if doc else "")
        )
    return evidences


def _parse_llm_output(
    raw: str,
    requirement: Dict[str, str],
    fallback_evidences: List[Evidence],
    input_hash: str,
    model_version: str,
) -> EvaluationCard:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(f"Parse error for {requirement['id']}: {exc}")
        return EvaluationCard(
            requirement_id=requirement["id"],
            requirement_text=requirement["text"],
            verdict=Verdict.NON_APPLICABILE,
            evidences=fallback_evidences,
            gaps=["LLM output parsing failed — manual review required"],
            corrective_action=CorrectiveAction(
                description="Review requirement manually due to LLM output parse error",
                expected_document_type="compliance_review",
            ),
            execution_metadata=ExecutionMetadata(
                timestamp=datetime.utcnow().isoformat() + "Z",
                model_version=model_version,
                input_hash=input_hash,
            ),
        )

    evidences = []
    for ev in data.get("evidences", []):
        if isinstance(ev, dict):
            evidences.append(
                Evidence(
                    chunk_id=str(ev.get("chunk_id", "unknown")),
                    source_doc=str(ev.get("source_doc", "unknown")),
                    excerpt=str(ev.get("excerpt", ""))[:400],
                )
            )
    if not evidences:
        evidences = fallback_evidences

    ca_data = data.get("corrective_action", {})
    if isinstance(ca_data, dict):
        corrective_action = CorrectiveAction(
            description=str(ca_data.get("description", "No corrective action specified")),
            expected_document_type=str(ca_data.get("expected_document_type", "policy_document")),
        )
    else:
        corrective_action = CorrectiveAction(
            description="No corrective action specified",
            expected_document_type="policy_document",
        )

    meta_data = data.get("execution_metadata", {})
    execution_metadata = ExecutionMetadata(
        timestamp=str(meta_data.get("timestamp", datetime.utcnow().isoformat() + "Z")),
        model_version=str(meta_data.get("model_version", model_version)),
        input_hash=str(meta_data.get("input_hash", input_hash)),
    )

    raw_verdict = str(data.get("verdict", "NON_APPLICABILE")).upper().strip()
    try:
        verdict = Verdict(raw_verdict)
    except ValueError:
        verdict = Verdict.NON_APPLICABILE

    return EvaluationCard(
        requirement_id=requirement["id"],
        requirement_text=requirement["text"],
        verdict=verdict,
        evidences=evidences,
        gaps=data.get("gaps", []),
        corrective_action=corrective_action,
        execution_metadata=execution_metadata,
    )


def _evaluate_requirement(
    requirement: Dict[str, str],
    org_context: str,
    iso_context: str,
    input_hash: str,
) -> EvaluationCard:
    settings = get_settings()
    llm = get_llm(temperature=0.0)
    model_version = (
        settings.ANTHROPIC_MODEL
        if settings.LLM_PROVIDER == "anthropic"
        else settings.OLLAMA_MODEL
    )

    prompt_text = EVALUATION_PROMPT.format(
        requirement_id=requirement["id"],
        requirement_text=requirement["text"],
        org_context=org_context or "No organizational documentation provided.",
        iso_context=iso_context or "No ISO standard context available.",
    )

    try:
        from langchain_core.messages import HumanMessage
        with track_llm_call("AS-3"):
            response = llm.invoke([HumanMessage(content=prompt_text)])
        raw_output = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        logger.error(f"LLM failed for {requirement['id']}: {exc}")
        return EvaluationCard(
            requirement_id=requirement["id"],
            requirement_text=requirement["text"],
            verdict=Verdict.NON_APPLICABILE,
            evidences=[],
            gaps=[f"LLM invocation failed: {str(exc)}"],
            corrective_action=CorrectiveAction(
                description="LLM service unavailable — review requirement manually",
                expected_document_type="compliance_review",
            ),
            execution_metadata=ExecutionMetadata(
                timestamp=datetime.utcnow().isoformat() + "Z",
                model_version=model_version,
                input_hash=input_hash,
            ),
        )

    return _parse_llm_output(raw_output, requirement, [], input_hash, model_version)


def _retrieve_context(query_text: str, org_id: str) -> tuple[str, str, List[Evidence]]:
    try:
        org_results = rag_query(
            COLLECTION_ORG_DOCS,
            query_text,
            n_results=5,
            where={"org_id": org_id} if org_id else None,
        )
    except Exception as exc:
        logger.warning(f"ORG-DOCS query failed: {exc}")
        org_results = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    # Query ISO-CL910-A710 (falls back to ISO-FULL when partition is empty)
    try:
        iso_results = query_iso_with_fallback(COLLECTION_ISO_CL910_A710, query_text, n_results=3)
    except Exception as exc:
        logger.warning(f"ISO-CL910-A710 query failed: {exc}")
        iso_results = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    org_context = _format_rag_results(org_results)
    iso_context = _format_rag_results(iso_results)
    evidences = _build_evidences(org_results)

    return org_context, iso_context, evidences


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="AS-3")


@app.post("/analyze", response_model=List[EvaluationCard])
async def analyze(request: AnalyzeRequest) -> List[EvaluationCard]:
    """
    Evaluate organizational documents against ISO 42001 Clauses 9, 10 and Annex A.7-A.10.
    """
    org_id = request.org_id
    if not org_id:
        raise HTTPException(status_code=400, detail="org_id is required")

    if not request.documents:
        raise HTTPException(status_code=400, detail="At least one document is required")

    # Load requirements from ISO RAG collection
    requirements = load_requirements_from_rag(COLLECTION_ISO_CL910_A710, _AS3_PREFIXES)
    if not requirements:
        raise HTTPException(
            status_code=503,
            detail=(
                "ISO/IEC 42001 standard not indexed in the RAG collection ISO-CL910-A710. "
                "Index the ISO document first using scripts/index_iso.py."
            ),
        )

    input_hash = request.compute_input_hash()
    logger.info(
        f"AS-3 analyze: org_id={org_id}, docs={len(request.documents)}, "
        f"requirements={len(requirements)}, hash={input_hash[:8]}"
    )

    from rag.indexer import index_text_as_org_doc
    for doc in request.documents:
        try:
            index_text_as_org_doc(doc.content, doc.filename, org_id)
        except Exception as exc:
            logger.warning(f"Failed to index doc {doc.filename}: {exc}")

    evaluation_cards: List[EvaluationCard] = []

    for requirement in requirements:
        logger.info(f"Evaluating requirement {requirement['id']}")
        query_text = f"{requirement['id']} {requirement['text'][:200]}"

        org_context, iso_context, _ = _retrieve_context(query_text, org_id)

        card = _evaluate_requirement(
            requirement=requirement,
            org_context=org_context,
            iso_context=iso_context,
            input_hash=input_hash,
        )
        evaluation_cards.append(card)
        REQUIREMENTS_EVALUATED.labels(service="AS-3", verdict=card.verdict.value).inc()
        logger.info(f"Requirement {requirement['id']}: {card.verdict}")

    logger.info(f"AS-3 completed: {len(evaluation_cards)} cards for org_id={org_id}")
    return evaluation_cards


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")
