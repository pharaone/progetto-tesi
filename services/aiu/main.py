"""AIU: User Interface Agent — Conversational Layer.

Holds a GapReport in context and answers user questions about it conversationally.
Saves/retrieves chat history from ORG-HISTORY ChromaDB collection.

Endpoints:
- GET /health
- POST /chat → ChatResponse
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from typing import List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.config import get_llm, get_settings
from shared.models import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    GapReport,
    HealthResponse,
    Verdict,
)
from rag.collections import (
    COLLECTION_ORG_HISTORY,
    add_documents,
    query as rag_query,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AIU: User Interface Agent",
    version="1.0.0",
    description="Conversational agent for ISO 42001 gap analysis results",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert ISO/IEC 42001:2023 AI Management System compliance consultant.
You have access to a detailed gap analysis report for the organization.

Your role is to:
1. Answer questions about the gap analysis results clearly and accurately
2. Explain compliance requirements in plain language
3. Prioritize corrective actions based on severity and business impact
4. Provide practical guidance on how to address identified gaps
5. Reference specific requirement IDs (e.g., cl-4.1, A.6.2.3) when relevant
6. Be constructive and actionable in your recommendations

Current Gap Analysis Summary:
- Organization: {org_id}
- Overall Compliance Score: {score:.1f}/100
- Compliant Requirements: {compliant}
- Non-Compliant Requirements: {non_compliant}
- Partially Compliant Requirements: {partial}
- Not Applicable: {not_applicable}
- Total Requirements: {total}
- Partial Coverage (some agents failed): {partial_coverage}

Top Priority Gaps:
{top_gaps}

Previous Conversation Context:
{history_context}

Be conversational, precise, and helpful. If asked about a specific requirement, provide detailed guidance.
"""


def _format_gap_report_for_prompt(report: GapReport) -> dict:
    """Format gap report data for the system prompt."""
    top_gaps_lines = []
    for gap in report.prioritized_gaps[:8]:
        severity_label = "CRITICAL" if gap.severity == 1 else "MODERATE"
        gaps_text = "; ".join(gap.gaps[:2]) if gap.gaps else "No specific gaps"
        top_gaps_lines.append(
            f"  [{severity_label}] {gap.requirement_id}: {gaps_text}"
        )
    top_gaps = "\n".join(top_gaps_lines) if top_gaps_lines else "  No significant gaps found."

    return {
        "org_id": report.org_id,
        "score": report.overall_compliance_score,
        "compliant": report.counts.compliant,
        "non_compliant": report.counts.non_compliant,
        "partial": report.counts.partial,
        "not_applicable": report.counts.not_applicable,
        "total": report.total_requirements,
        "partial_coverage": report.partial_coverage,
        "top_gaps": top_gaps,
    }


def _get_chat_history_from_rag(org_id: str, query_text: str) -> str:
    """Retrieve prior exchanges relevant to the current question from ORG-HISTORY.

    Semantic retrieval keyed on the user's actual message, so clarifications
    given in previous sessions surface when the same topic comes up again
    (thesis §3.3.4: contextual, per-organization knowledge base).
    """
    try:
        results = rag_query(
            COLLECTION_ORG_HISTORY,
            query_text,
            n_results=5,
            # NB: ChromaDB requires an explicit $and for multi-key filters —
            # a plain multi-key dict raises and would disable history retrieval
            where={"$and": [{"org_id": {"$eq": org_id}}, {"type": {"$eq": "chat"}}]},
        )
        docs = results.get("documents", [[]])[0]
        if docs:
            return "\n---\n".join(docs[:3])
        return "No previous conversation history."
    except Exception as exc:
        logger.warning(f"Failed to retrieve chat history: {exc}")
        return "No previous conversation history."


def _save_chat_to_history(org_id: str, user_msg: str, assistant_msg: str) -> None:
    """Save a chat exchange to ORG-HISTORY."""
    try:
        timestamp = datetime.utcnow().isoformat() + "Z"
        exchange = f"User: {user_msg}\nAssistant: {assistant_msg}"
        doc_id = f"{org_id}_chat_{timestamp.replace(':', '-').replace('.', '-')}"

        add_documents(
            COLLECTION_ORG_HISTORY,
            [exchange],
            [{"org_id": org_id, "timestamp": timestamp, "type": "chat"}],
            [doc_id],
        )
    except Exception as exc:
        logger.warning(f"Failed to save chat to history: {exc}")


def _build_conversation_prompt(
    system_prompt: str,
    chat_history: List[ChatMessage],
    user_message: str,
) -> List:
    """Build LangChain-compatible message list."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    messages = [SystemMessage(content=system_prompt)]

    # Add chat history (last 10 exchanges to stay within context)
    for msg in chat_history[-20:]:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))

    messages.append(HumanMessage(content=user_message))
    return messages


def _generate_response_without_report(user_message: str, chat_history: List[ChatMessage]) -> str:
    """Generate a response when no gap report is available."""
    system = (
        "You are an ISO/IEC 42001:2023 compliance consultant. "
        "No gap analysis report has been provided yet. "
        "Explain that you need a gap report to answer specific compliance questions, "
        "and guide the user to run the analysis first. "
        "You can answer general questions about ISO 42001."
    )

    llm = get_llm(temperature=0.0)
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [SystemMessage(content=system), HumanMessage(content=user_message)]
        response = llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        logger.error(f"LLM failed: {exc}")
        return (
            "I'm here to help with ISO/IEC 42001 compliance questions. "
            "Please run the gap analysis first so I can provide specific insights about your organization."
        )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="AIU")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Conversational chat about ISO 42001 gap analysis results.

    If gap_report is provided, it is used as context.
    Chat history is stored to and retrieved from ORG-HISTORY.
    """
    org_id = request.org_id
    if not org_id:
        raise HTTPException(status_code=400, detail="org_id is required")

    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    user_message = request.message.strip()
    chat_history = list(request.chat_history)

    logger.info(f"AIU chat: org_id={org_id}, message_len={len(user_message)}")

    # If no report provided, give general guidance
    if request.gap_report is None:
        assistant_response = _generate_response_without_report(user_message, chat_history)
    else:
        report = request.gap_report

        # Build system prompt with report context; retrieve prior exchanges
        # semantically related to the current question
        history_context = _get_chat_history_from_rag(org_id, user_message)
        report_data = _format_gap_report_for_prompt(report)
        report_data["history_context"] = history_context

        system_prompt = SYSTEM_PROMPT.format(**report_data)

        # Build and invoke LLM
        llm = get_llm(temperature=0.0)
        try:
            messages = _build_conversation_prompt(system_prompt, chat_history, user_message)
            response = llm.invoke(messages)
            assistant_response = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.error(f"LLM failed during chat: {exc}", exc_info=True)
            assistant_response = (
                "I encountered an error processing your request. "
                "Please try again or check the service logs."
            )

        # Save to ORG-HISTORY
        _save_chat_to_history(org_id, user_message, assistant_response)

    # Update chat history
    timestamp = datetime.utcnow().isoformat() + "Z"
    chat_history.append(ChatMessage(role="user", content=user_message, timestamp=timestamp))
    chat_history.append(
        ChatMessage(role="assistant", content=assistant_response, timestamp=timestamp)
    )

    return ChatResponse(
        org_id=org_id,
        message=assistant_response,
        chat_history=chat_history,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005, log_level="info")
