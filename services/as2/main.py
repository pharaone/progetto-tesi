"""AS-2: ISO/IEC 42001 Compliance Agent — Clauses 7, 8 + Annex A (A.2–A.6).

Evaluates organizational documents against:
- Clause 7: Support (7.1–7.5)
- Clause 8: Operations (8.1–8.4)
- Annex A Controls: A.2.1–A.2.2, A.3.1, A.4.1–A.4.2, A.5.1–A.5.7, A.6.1.1–A.6.2.8

RAG collections used: ISO-CL78-A26 + ORG-DOCS
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
    COLLECTION_ISO_CL78_A26,
    COLLECTION_ORG_DOCS,
    query as rag_query,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AS-2: ISO/IEC 42001 Clauses 7-8 + Annex A.2-A.6 Compliance Agent",
    version="1.0.0",
    description="Evaluates compliance with ISO/IEC 42001 Clauses 7 (Support), 8 (Operations) and Annex A controls A.2-A.6",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Requirements catalogue
# ---------------------------------------------------------------------------
REQUIREMENTS: List[Dict[str, str]] = [
    # Clause 7: Support
    {
        "id": "cl-7.1",
        "text": (
            "The organization shall determine and provide the resources needed for the "
            "establishment, implementation, maintenance, and continual improvement of the "
            "AI management system, including human resources and technical infrastructure."
        ),
    },
    {
        "id": "cl-7.2",
        "text": (
            "The organization shall determine the necessary competence of person(s) doing work "
            "under its control that affects its AI performance; ensure that these persons are "
            "competent on the basis of appropriate education, training, or experience; where "
            "applicable, take actions to acquire the necessary competence and evaluate the "
            "effectiveness of the actions taken; and retain appropriate documented information "
            "as evidence of competence."
        ),
    },
    {
        "id": "cl-7.3",
        "text": (
            "Persons doing work under the organization's control shall be made aware of the AI "
            "policy; their contribution to the effectiveness of the AI management system, "
            "including the benefits of improved AI performance; and the implications of not "
            "conforming with the AI management system requirements."
        ),
    },
    {
        "id": "cl-7.4",
        "text": (
            "The organization shall determine the need for internal and external communications "
            "relevant to the AI management system, including on what it will communicate, when "
            "to communicate, with whom to communicate, how to communicate, and who communicates."
        ),
    },
    {
        "id": "cl-7.5",
        "text": (
            "The organization's AI management system shall include documented information "
            "required by this document, determined by the organization as being necessary for "
            "the effectiveness of the AI management system. When creating and updating documented "
            "information, the organization shall ensure appropriate identification, format, "
            "review, approval, storage, protection, retrieval, distribution, and disposition."
        ),
    },
    # Clause 8: Operations
    {
        "id": "cl-8.1",
        "text": (
            "The organization shall plan, implement, control, monitor, and review the processes "
            "needed to meet requirements for the provision of AI systems, and to implement the "
            "actions determined in Clause 6, by establishing criteria for the processes and "
            "implementing control of the processes in accordance with the criteria."
        ),
    },
    {
        "id": "cl-8.2",
        "text": (
            "The organization shall implement processes to ensure that requirements related to "
            "AI systems are determined, including applicable legal requirements, and that "
            "technical and operational requirements for AI systems are documented."
        ),
    },
    {
        "id": "cl-8.3",
        "text": (
            "The organization shall establish, implement, and maintain a process for the design "
            "and development of AI systems that includes planning of design and development, "
            "design and development inputs, controls, outputs, and changes."
        ),
    },
    {
        "id": "cl-8.4",
        "text": (
            "The organization shall establish and implement processes to verify that externally "
            "provided AI systems, components, or services meet specified requirements before "
            "use, including supplier evaluation and monitoring."
        ),
    },
    # Annex A Controls A.2
    {
        "id": "A.2.1",
        "text": (
            "A.2.1 AI system impact assessment: The organization shall assess and document the "
            "potential impacts of AI systems on individuals, groups, and society before deployment "
            "and throughout the AI system lifecycle."
        ),
    },
    {
        "id": "A.2.2",
        "text": (
            "A.2.2 AI system impact assessment review: The organization shall review AI system "
            "impact assessments periodically and when significant changes are made to AI systems "
            "or their operating context."
        ),
    },
    # Annex A Controls A.3
    {
        "id": "A.3.1",
        "text": (
            "A.3.1 AI system inventory: The organization shall establish and maintain an "
            "inventory of AI systems that includes the purpose, intended use, deployment context, "
            "data used, and responsible parties for each AI system."
        ),
    },
    # Annex A Controls A.4
    {
        "id": "A.4.1",
        "text": (
            "A.4.1 Policies for responsible AI: The organization shall establish policies for "
            "responsible development, deployment, and use of AI systems, addressing ethical "
            "considerations, fairness, transparency, explainability, and accountability."
        ),
    },
    {
        "id": "A.4.2",
        "text": (
            "A.4.2 Processes for responsible AI: The organization shall implement processes "
            "to operationalize responsible AI policies throughout the AI system lifecycle."
        ),
    },
    # Annex A Controls A.5
    {
        "id": "A.5.1",
        "text": (
            "A.5.1 AI system data governance: The organization shall establish data governance "
            "processes for AI systems, including data quality, data sourcing, data documentation, "
            "and data management throughout the AI system lifecycle."
        ),
    },
    {
        "id": "A.5.2",
        "text": (
            "A.5.2 Data acquisition for AI systems: The organization shall implement processes "
            "to ensure that data used for AI systems is acquired in a lawful, ethical, and "
            "documented manner appropriate for the intended AI system purpose."
        ),
    },
    {
        "id": "A.5.3",
        "text": (
            "A.5.3 Data quality for AI systems: The organization shall implement controls to "
            "ensure that data used in AI systems meets defined quality criteria including "
            "accuracy, completeness, consistency, timeliness, and relevance."
        ),
    },
    {
        "id": "A.5.4",
        "text": (
            "A.5.4 Data preparation for AI systems: The organization shall implement and "
            "document data preparation processes including cleaning, transformation, labelling, "
            "and augmentation for AI systems."
        ),
    },
    {
        "id": "A.5.5",
        "text": (
            "A.5.5 Data documentation for AI systems: The organization shall create and maintain "
            "documentation of datasets used for training, testing, and operating AI systems, "
            "including data sources, characteristics, and known limitations."
        ),
    },
    {
        "id": "A.5.6",
        "text": (
            "A.5.6 Data access control for AI systems: The organization shall implement access "
            "controls for data used in AI systems to prevent unauthorized access, modification, "
            "or misuse."
        ),
    },
    {
        "id": "A.5.7",
        "text": (
            "A.5.7 AI system data provenance: The organization shall track and document the "
            "origin, transformations, and chain of custody of data used in AI systems to "
            "support auditability and traceability."
        ),
    },
    # Annex A Controls A.6
    {
        "id": "A.6.1.1",
        "text": (
            "A.6.1.1 Establishment of AI system operational objectives: The organization shall "
            "establish measurable operational objectives for AI systems aligned with the "
            "organization's AI policy and intended outcomes."
        ),
    },
    {
        "id": "A.6.1.2",
        "text": (
            "A.6.1.2 AI system design: The organization shall design AI systems with appropriate "
            "consideration of performance, safety, security, privacy, fairness, transparency, "
            "and explainability requirements."
        ),
    },
    {
        "id": "A.6.1.3",
        "text": (
            "A.6.1.3 AI system model documentation: The organization shall document AI system "
            "models including architecture, training procedures, hyperparameters, evaluation "
            "metrics, and known limitations."
        ),
    },
    {
        "id": "A.6.1.4",
        "text": (
            "A.6.1.4 AI system testing: The organization shall implement systematic testing "
            "of AI systems including functional testing, performance testing, safety testing, "
            "robustness testing, and bias testing before deployment."
        ),
    },
    {
        "id": "A.6.2.1",
        "text": (
            "A.6.2.1 AI system deployment: The organization shall implement controlled "
            "deployment processes for AI systems including staged rollout, monitoring setup, "
            "and rollback procedures."
        ),
    },
    {
        "id": "A.6.2.2",
        "text": (
            "A.6.2.2 Human oversight of AI systems: The organization shall implement "
            "appropriate human oversight mechanisms for AI systems, proportionate to the "
            "level of risk and autonomy of the AI system."
        ),
    },
    {
        "id": "A.6.2.3",
        "text": (
            "A.6.2.3 AI system monitoring: The organization shall implement continuous "
            "monitoring of AI systems in operation to detect performance degradation, "
            "unexpected behavior, and emerging risks."
        ),
    },
    {
        "id": "A.6.2.4",
        "text": (
            "A.6.2.4 AI system incident management: The organization shall establish processes "
            "for detecting, reporting, assessing, and responding to AI system incidents "
            "including adverse events and near-misses."
        ),
    },
    {
        "id": "A.6.2.5",
        "text": (
            "A.6.2.5 AI system change management: The organization shall implement change "
            "management processes for AI systems to ensure that changes are assessed, "
            "approved, tested, and documented before implementation."
        ),
    },
    {
        "id": "A.6.2.6",
        "text": (
            "A.6.2.6 AI system decommissioning: The organization shall implement processes "
            "for the planned retirement and decommissioning of AI systems, including data "
            "handling, user notification, and documentation."
        ),
    },
    {
        "id": "A.6.2.7",
        "text": (
            "A.6.2.7 AI system record keeping: The organization shall maintain records of "
            "AI system operations, decisions, incidents, and changes to support auditability, "
            "accountability, and continual improvement."
        ),
    },
    {
        "id": "A.6.2.8",
        "text": (
            "A.6.2.8 Feedback mechanisms for AI systems: The organization shall implement "
            "mechanisms to collect, analyze, and act on feedback from users and affected "
            "parties about AI system performance and impacts."
        ),
    },
]

# ---------------------------------------------------------------------------
# Prompt template (same structure as AS-1)
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

    try:
        iso_results = rag_query(COLLECTION_ISO_CL78_A26, query_text, n_results=3)
    except Exception as exc:
        logger.warning(f"ISO-CL78-A26 query failed: {exc}")
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
    return HealthResponse(service="AS-2")


@app.post("/analyze", response_model=List[EvaluationCard])
async def analyze(request: AnalyzeRequest) -> List[EvaluationCard]:
    """
    Evaluate organizational documents against ISO 42001 Clauses 7, 8 and Annex A.2-A.6.
    """
    org_id = request.org_id
    if not org_id:
        raise HTTPException(status_code=400, detail="org_id is required")

    if not request.documents:
        raise HTTPException(status_code=400, detail="At least one document is required")

    input_hash = request.compute_input_hash()
    logger.info(f"AS-2 analyze: org_id={org_id}, docs={len(request.documents)}, hash={input_hash[:8]}")

    from rag.indexer import index_text_as_org_doc
    for doc in request.documents:
        try:
            index_text_as_org_doc(doc.content, doc.filename, org_id)
        except Exception as exc:
            logger.warning(f"Failed to index doc {doc.filename}: {exc}")

    evaluation_cards: List[EvaluationCard] = []

    for requirement in REQUIREMENTS:
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
        logger.info(f"Requirement {requirement['id']}: {card.verdict}")

    logger.info(f"AS-2 completed: {len(evaluation_cards)} cards for org_id={org_id}")
    return evaluation_cards


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
