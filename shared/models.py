"""Shared Pydantic models for the ISO/IEC 42001 gap analysis system."""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Verdict(str, Enum):
    CONFORME = "CONFORME"
    NON_CONFORME = "NON_CONFORME"
    PARZIALMENTE_CONFORME = "PARZIALMENTE_CONFORME"
    NON_APPLICABILE = "NON_APPLICABILE"


class Evidence(BaseModel):
    chunk_id: str = Field(..., description="Unique identifier of the retrieved chunk")
    source_doc: str = Field(..., description="Source document filename or path")
    excerpt: str = Field(..., description="Relevant text excerpt from the chunk")


class CorrectiveAction(BaseModel):
    description: str = Field(..., description="Description of the corrective action to take")
    expected_document_type: str = Field(
        ..., description="Type of document expected to address the gap"
    )


class ExecutionMetadata(BaseModel):
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO 8601 timestamp of evaluation",
    )
    model_version: str = Field(default="unknown", description="Model version used")
    input_hash: str = Field(default="", description="SHA-256 of input documents")


class EvaluationCard(BaseModel):
    requirement_id: str = Field(..., description="Requirement identifier (e.g. cl-4.1, A.6.1.2)")
    requirement_text: str = Field(..., description="Full text of the ISO requirement")
    verdict: Verdict = Field(..., description="Compliance verdict")
    evidences: List[Evidence] = Field(default_factory=list, description="Supporting evidence")
    gaps: List[str] = Field(default_factory=list, description="Identified compliance gaps")
    corrective_action: CorrectiveAction = Field(
        ..., description="Corrective action recommendation"
    )
    execution_metadata: ExecutionMetadata = Field(
        default_factory=ExecutionMetadata,
        description="Execution metadata",
    )

    @field_validator("verdict", mode="before")
    @classmethod
    def normalize_verdict(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.upper().strip()
        return v


class PrioritizedGap(BaseModel):
    requirement_id: str
    requirement_text: str
    verdict: Verdict
    gaps: List[str]
    corrective_action: CorrectiveAction
    severity: int = Field(
        ..., description="Severity: 1=NON_CONFORME, 2=PARZIALMENTE_CONFORME"
    )


class ActionPlanItem(BaseModel):
    priority: int = Field(..., description="Priority order (1 = highest)")
    requirement_id: str
    description: str
    expected_document_type: str
    verdict: Verdict


class ComplianceCounts(BaseModel):
    compliant: int = 0
    non_compliant: int = 0
    partial: int = 0
    not_applicable: int = 0


class GapReport(BaseModel):
    org_id: str = Field(..., description="Organization identifier")
    overall_compliance_score: float = Field(
        ..., description="Compliance score 0-100", ge=0, le=100
    )
    total_requirements: int = Field(..., description="Total number of requirements evaluated")
    counts: ComplianceCounts = Field(..., description="Counts by verdict")
    partial_coverage: bool = Field(
        default=False, description="True if some AS agents timed out"
    )
    failed_agents: List[str] = Field(
        default_factory=list, description="List of agents that failed"
    )
    prioritized_gaps: List[PrioritizedGap] = Field(
        default_factory=list,
        description="Gaps sorted by severity: NON_CONFORME first, then PARZIALMENTE_CONFORME",
    )
    action_plan: List[ActionPlanItem] = Field(
        default_factory=list, description="Prioritized corrective actions"
    )
    evaluation_cards: List[EvaluationCard] = Field(
        default_factory=list, description="All evaluation cards from AS-1/2/3"
    )
    execution_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Execution metadata"
    )


class DocumentInput(BaseModel):
    filename: str = Field(..., description="Original filename")
    content: str = Field(..., description="Text content of the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AnalyzeRequest(BaseModel):
    org_id: str = Field(..., description="Organization identifier")
    documents: List[DocumentInput] = Field(
        ..., description="Organizational documents to analyze"
    )

    def compute_input_hash(self) -> str:
        """Compute SHA-256 of all document contents concatenated."""
        combined = "".join(sorted(doc.content for doc in self.documents))
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()


class ConsolidateRequest(BaseModel):
    org_id: str = Field(..., description="Organization identifier")
    as1_output: Optional[List[EvaluationCard]] = Field(
        None, description="Output from AS-1 agent"
    )
    as2_output: Optional[List[EvaluationCard]] = Field(
        None, description="Output from AS-2 agent"
    )
    as3_output: Optional[List[EvaluationCard]] = Field(
        None, description="Output from AS-3 agent"
    )
    failed_agents: List[str] = Field(
        default_factory=list, description="Agents that failed"
    )


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )


class ChatRequest(BaseModel):
    org_id: str = Field(..., description="Organization identifier")
    message: str = Field(..., description="User message")
    gap_report: Optional[GapReport] = Field(
        None, description="Gap report context (optional if already loaded)"
    )
    chat_history: List[ChatMessage] = Field(
        default_factory=list, description="Previous chat messages"
    )


class ChatResponse(BaseModel):
    org_id: str
    message: str = Field(..., description="Assistant response")
    chat_history: List[ChatMessage] = Field(..., description="Updated chat history")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    service: str = Field(..., description="Service name")
    version: str = Field(default="1.0.0")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
