"""AGA: Gap Analysis Agent — Consolidator.

Receives outputs from AS-1, AS-2, AS-3 and produces a consolidated GapReport.
Uses ISO-FULL and ORG-HISTORY ChromaDB collections.

Endpoints:
- GET /health
- POST /consolidate → GapReport
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.config import get_llm, get_settings
from shared.metrics import setup_metrics, track_llm_call
from shared.models import (
    ActionPlanItem,
    ComplianceCounts,
    ConsolidateRequest,
    EvaluationCard,
    GapReport,
    HealthResponse,
    PrioritizedGap,
    Verdict,
)
from rag.collections import (
    COLLECTION_ISO_FULL,
    COLLECTION_ORG_HISTORY,
    add_documents,
    query as rag_query,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AGA: Gap Analysis Agent",
    version="1.0.0",
    description="Consolidates AS-1/2/3 outputs into a comprehensive ISO 42001 gap analysis report",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_metrics(app, "AGA")

# ---------------------------------------------------------------------------
# Consolidation logic
# ---------------------------------------------------------------------------

CONSOLIDATION_PROMPT = """\
You are an ISO/IEC 42001:2023 compliance expert. You have received evaluation results from a gap analysis audit.

Organization ID: {org_id}
Total Requirements Evaluated: {total_requirements}
Compliant: {compliant}
Non-Compliant: {non_compliant}
Partially Compliant: {partial}
Not Applicable: {not_applicable}

Key Gaps Found:
{gaps_summary}

Previous Assessment Context:
{history_context}

ISO Standard Context:
{iso_context}

Based on this information, provide a brief executive summary (2-3 sentences) of the organization's overall
ISO/IEC 42001 compliance posture and the most critical areas requiring attention.

Return ONLY a JSON object:
{{
  "executive_summary": "<2-3 sentence summary>",
  "critical_areas": ["<area1>", "<area2>", "<area3>"]
}}
"""


def _compute_compliance_score(cards: List[EvaluationCard]) -> float:
    """
    Compute weighted compliance score (0-100):
    - CONFORME = 1.0
    - PARZIALMENTE_CONFORME = 0.5
    - NON_CONFORME = 0.0
    - NON_APPLICABILE = excluded from denominator
    """
    if not cards:
        return 0.0

    total_weight = 0.0
    earned_weight = 0.0

    for card in cards:
        if card.verdict == Verdict.NON_APPLICABILE:
            continue
        total_weight += 1.0
        if card.verdict == Verdict.CONFORME:
            earned_weight += 1.0
        elif card.verdict == Verdict.PARZIALMENTE_CONFORME:
            earned_weight += 0.5
        # NON_CONFORME contributes 0.0

    if total_weight == 0.0:
        return 100.0  # All not applicable = technically compliant

    return round((earned_weight / total_weight) * 100.0, 2)


def _count_by_verdict(cards: List[EvaluationCard]) -> ComplianceCounts:
    counts = ComplianceCounts()
    for card in cards:
        if card.verdict == Verdict.CONFORME:
            counts.compliant += 1
        elif card.verdict == Verdict.NON_CONFORME:
            counts.non_compliant += 1
        elif card.verdict == Verdict.PARZIALMENTE_CONFORME:
            counts.partial += 1
        elif card.verdict == Verdict.NON_APPLICABILE:
            counts.not_applicable += 1
    return counts


def _build_prioritized_gaps(cards: List[EvaluationCard]) -> List[PrioritizedGap]:
    """
    Build prioritized gaps list:
    - NON_CONFORME first (severity=1)
    - PARZIALMENTE_CONFORME second (severity=2)
    """
    gaps = []

    # NON_CONFORME first
    for card in cards:
        if card.verdict == Verdict.NON_CONFORME:
            gaps.append(
                PrioritizedGap(
                    requirement_id=card.requirement_id,
                    requirement_text=card.requirement_text,
                    verdict=card.verdict,
                    gaps=card.gaps,
                    corrective_action=card.corrective_action,
                    severity=1,
                )
            )

    # PARZIALMENTE_CONFORME second
    for card in cards:
        if card.verdict == Verdict.PARZIALMENTE_CONFORME:
            gaps.append(
                PrioritizedGap(
                    requirement_id=card.requirement_id,
                    requirement_text=card.requirement_text,
                    verdict=card.verdict,
                    gaps=card.gaps,
                    corrective_action=card.corrective_action,
                    severity=2,
                )
            )

    return gaps


def _build_action_plan(prioritized_gaps: List[PrioritizedGap]) -> List[ActionPlanItem]:
    """Build an ordered action plan from prioritized gaps."""
    action_plan = []
    for priority, gap in enumerate(prioritized_gaps, start=1):
        action_plan.append(
            ActionPlanItem(
                priority=priority,
                requirement_id=gap.requirement_id,
                description=gap.corrective_action.description,
                expected_document_type=gap.corrective_action.expected_document_type,
                verdict=gap.verdict,
            )
        )
    return action_plan


def _get_gaps_summary(prioritized_gaps: List[PrioritizedGap], max_gaps: int = 10) -> str:
    """Format top gaps for the LLM prompt."""
    if not prioritized_gaps:
        return "No significant gaps identified."

    lines = []
    for gap in prioritized_gaps[:max_gaps]:
        gap_list = "; ".join(gap.gaps[:3]) if gap.gaps else "No specific gaps listed"
        lines.append(
            f"- {gap.requirement_id} ({gap.verdict}): {gap_list}"
        )
    return "\n".join(lines)


def _save_report_to_history(org_id: str, report: GapReport) -> None:
    """Save a summary of the gap report to ORG-HISTORY collection."""
    try:
        from datetime import datetime as dt
        timestamp = dt.utcnow().isoformat() + "Z"

        summary = (
            f"Gap Analysis Report for {org_id} at {timestamp}. "
            f"Score: {report.overall_compliance_score:.1f}/100. "
            f"Compliant: {report.counts.compliant}, "
            f"Non-Compliant: {report.counts.non_compliant}, "
            f"Partial: {report.counts.partial}, "
            f"Not Applicable: {report.counts.not_applicable}. "
            f"Critical gaps: {len([g for g in report.prioritized_gaps if g.severity == 1])}."
        )

        doc_id = f"{org_id}_report_{timestamp.replace(':', '-').replace('.', '-')}"
        add_documents(
            COLLECTION_ORG_HISTORY,
            [summary],
            [{"org_id": org_id, "timestamp": timestamp, "type": "gap_report"}],
            [doc_id],
        )
        logger.info(f"Saved report summary to ORG-HISTORY for org_id={org_id}")
    except Exception as exc:
        logger.warning(f"Failed to save report to ORG-HISTORY: {exc}")


def _get_llm_summary(
    org_id: str,
    all_cards: List[EvaluationCard],
    counts: ComplianceCounts,
    prioritized_gaps: List[PrioritizedGap],
) -> Dict[str, Any]:
    """Get an LLM-generated executive summary."""
    try:
        gaps_summary = _get_gaps_summary(prioritized_gaps, max_gaps=5)

        # Query ISO-FULL for context
        iso_context = ""
        try:
            iso_results = rag_query(
                COLLECTION_ISO_FULL,
                "ISO 42001 compliance requirements overview",
                n_results=3,
            )
            iso_docs = iso_results.get("documents", [[]])[0]
            iso_context = "\n".join(iso_docs[:2])[:800] if iso_docs else ""
        except Exception:
            iso_context = ""

        # Query ORG-HISTORY keyed on the CURRENT gaps, so clarifications and
        # prior assessments related to these requirements surface in the
        # synthesis (thesis §3.3.4: the AGA incorporates clarifications from
        # previous sessions when prioritizing gaps)
        history_context = ""
        try:
            hist_results = rag_query(
                COLLECTION_ORG_HISTORY,
                gaps_summary[:400] if prioritized_gaps else f"gap analysis {org_id}",
                n_results=4,
                where={"org_id": org_id},
            )
            hist_docs = hist_results.get("documents", [[]])[0]
            history_context = (
                "\n---\n".join(hist_docs[:3])[:1200] if hist_docs else "No previous assessments."
            )
        except Exception:
            history_context = "No previous assessments."

        prompt_text = CONSOLIDATION_PROMPT.format(
            org_id=org_id,
            total_requirements=len(all_cards),
            compliant=counts.compliant,
            non_compliant=counts.non_compliant,
            partial=counts.partial,
            not_applicable=counts.not_applicable,
            gaps_summary=gaps_summary,
            history_context=history_context,
            iso_context=iso_context,
        )

        llm = get_llm(temperature=0.0)
        from langchain_core.messages import HumanMessage
        with track_llm_call("AGA"):
            response = llm.invoke([HumanMessage(content=prompt_text)])
        raw = response.content if hasattr(response, "content") else str(response)

        # Parse JSON
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[: text.rfind("```")]
            text = text.strip()

        return json.loads(text)
    except Exception as exc:
        logger.warning(f"LLM summary generation failed: {exc}")
        return {
            "executive_summary": (
                f"Gap analysis completed for {org_id}. "
                f"Overall compliance score: {_compute_compliance_score(all_cards):.1f}/100. "
                f"Review the prioritized gaps for corrective actions."
            ),
            "critical_areas": [],
        }


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="AGA")


@app.post("/consolidate", response_model=GapReport)
async def consolidate(request: ConsolidateRequest) -> GapReport:
    """
    Consolidate AS-1, AS-2, AS-3 evaluation outputs into a comprehensive GapReport.
    """
    org_id = request.org_id
    if not org_id:
        raise HTTPException(status_code=400, detail="org_id is required")

    logger.info(
        f"AGA consolidate: org_id={org_id}, "
        f"as1={len(request.as1_output or [])}, "
        f"as2={len(request.as2_output or [])}, "
        f"as3={len(request.as3_output or [])} cards, "
        f"failed_agents={request.failed_agents}"
    )

    # Collect all evaluation cards
    all_cards: List[EvaluationCard] = []

    if request.as1_output:
        all_cards.extend(request.as1_output)
    if request.as2_output:
        all_cards.extend(request.as2_output)
    if request.as3_output:
        all_cards.extend(request.as3_output)

    if not all_cards:
        raise HTTPException(
            status_code=400,
            detail="No evaluation cards received from any AS agent",
        )

    # Compute metrics
    compliance_score = _compute_compliance_score(all_cards)
    counts = _count_by_verdict(all_cards)
    prioritized_gaps = _build_prioritized_gaps(all_cards)
    action_plan = _build_action_plan(prioritized_gaps)

    # LLM-generated summary
    llm_summary = _get_llm_summary(org_id, all_cards, counts, prioritized_gaps)

    partial_coverage = len(request.failed_agents) > 0

    report = GapReport(
        org_id=org_id,
        overall_compliance_score=compliance_score,
        total_requirements=len(all_cards),
        counts=counts,
        partial_coverage=partial_coverage,
        failed_agents=request.failed_agents,
        prioritized_gaps=prioritized_gaps,
        action_plan=action_plan,
        evaluation_cards=all_cards,
        execution_metadata={
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "as1_cards": len(request.as1_output or []),
            "as2_cards": len(request.as2_output or []),
            "as3_cards": len(request.as3_output or []),
            "total_cards": len(all_cards),
            "executive_summary": llm_summary.get("executive_summary", ""),
            "critical_areas": llm_summary.get("critical_areas", []),
        },
    )

    # Persist report summary to ORG-HISTORY
    _save_report_to_history(org_id, report)

    logger.info(
        f"AGA report: org_id={org_id}, score={compliance_score:.1f}, "
        f"gaps={len(prioritized_gaps)}, partial={partial_coverage}"
    )

    return report


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004, log_level="info")
