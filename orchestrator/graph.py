"""LangGraph orchestration DAG for ISO/IEC 42001 gap analysis.

Graph topology:
  START → run_agents_parallel → run_aga → run_aiu → END

AS-1, AS-2, AS-3 run concurrently inside run_agents_parallel via
asyncio.gather — LangGraph 0.1.x does not support multiple add_edge
calls from the same source node for fan-out.

State: GraphState TypedDict
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import httpx
from typing_extensions import TypedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import get_settings
from shared.models import (
    AnalyzeRequest,
    ChatMessage,
    ChatRequest,
    ConsolidateRequest,
    DocumentInput,
    EvaluationCard,
    GapReport,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared Graph State
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    org_id: str
    org_docs_path: str
    documents: List[Dict[str, Any]]  # List of DocumentInput dicts
    as1_output: Optional[List[Dict[str, Any]]]
    as2_output: Optional[List[Dict[str, Any]]]
    as3_output: Optional[List[Dict[str, Any]]]
    aga_report: Optional[Dict[str, Any]]
    chat_history: List[Dict[str, Any]]
    execution_metadata: Dict[str, Any]
    failed_agents: List[str]


# ---------------------------------------------------------------------------
# HTTP client helpers
# ---------------------------------------------------------------------------

async def _post_json(
    url: str,
    payload: dict,
    timeout: float,
) -> dict:
    """POST JSON payload to a service URL, return response dict."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def run_as1(state: GraphState) -> Dict[str, Any]:
    """Call AS-1 service to evaluate Clauses 4, 5, 6."""
    settings = get_settings()
    org_id = state["org_id"]
    documents = state.get("documents", [])

    logger.info(f"run_as1: org_id={org_id}, docs={len(documents)}")

    try:
        payload = {
            "org_id": org_id,
            "documents": documents,
        }
        result = await _post_json(
            f"{settings.AS1_URL}/analyze",
            payload,
            timeout=float(settings.AS_TIMEOUT),
        )
        logger.info(f"run_as1 completed: {len(result)} cards")
        return {"as1_output": result}
    except Exception as exc:
        logger.error(f"run_as1 failed: {exc}", exc_info=True)
        return {"as1_output": None, "failed_agents": state.get("failed_agents", []) + ["AS-1"]}


async def run_as2(state: GraphState) -> Dict[str, Any]:
    """Call AS-2 service to evaluate Clauses 7, 8 + Annex A.2-A.6."""
    settings = get_settings()
    org_id = state["org_id"]
    documents = state.get("documents", [])

    logger.info(f"run_as2: org_id={org_id}, docs={len(documents)}")

    try:
        payload = {
            "org_id": org_id,
            "documents": documents,
        }
        result = await _post_json(
            f"{settings.AS2_URL}/analyze",
            payload,
            timeout=float(settings.AS_TIMEOUT),
        )
        logger.info(f"run_as2 completed: {len(result)} cards")
        return {"as2_output": result}
    except Exception as exc:
        logger.error(f"run_as2 failed: {exc}", exc_info=True)
        return {"as2_output": None, "failed_agents": state.get("failed_agents", []) + ["AS-2"]}


async def run_as3(state: GraphState) -> Dict[str, Any]:
    """Call AS-3 service to evaluate Clauses 9, 10 + Annex A.7-A.10."""
    settings = get_settings()
    org_id = state["org_id"]
    documents = state.get("documents", [])

    logger.info(f"run_as3: org_id={org_id}, docs={len(documents)}")

    try:
        payload = {
            "org_id": org_id,
            "documents": documents,
        }
        result = await _post_json(
            f"{settings.AS3_URL}/analyze",
            payload,
            timeout=float(settings.AS_TIMEOUT),
        )
        logger.info(f"run_as3 completed: {len(result)} cards")
        return {"as3_output": result}
    except Exception as exc:
        logger.error(f"run_as3 failed: {exc}", exc_info=True)
        return {"as3_output": None, "failed_agents": state.get("failed_agents", []) + ["AS-3"]}


async def run_agents_parallel(state: GraphState) -> Dict[str, Any]:
    """Run AS-1, AS-2, AS-3 concurrently and merge their outputs."""
    results = await asyncio.gather(
        run_as1(state),
        run_as2(state),
        run_as3(state),
        return_exceptions=True,
    )

    merged: Dict[str, Any] = {}
    failed_agents = list(state.get("failed_agents", []))

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Agent raised exception: {result}", exc_info=result)
        elif isinstance(result, dict):
            # Collect any newly failed agents reported by the sub-calls
            for agent in result.get("failed_agents", []):
                if agent not in failed_agents:
                    failed_agents.append(agent)
            merged.update({k: v for k, v in result.items() if k != "failed_agents"})

    merged["failed_agents"] = failed_agents
    return merged


async def run_aga(state: GraphState) -> Dict[str, Any]:
    """Call AGA service to consolidate AS-1/2/3 outputs into GapReport."""
    settings = get_settings()
    org_id = state["org_id"]

    as1_output = state.get("as1_output")
    as2_output = state.get("as2_output")
    as3_output = state.get("as3_output")
    failed_agents = state.get("failed_agents", [])

    logger.info(
        f"run_aga: org_id={org_id}, "
        f"as1={len(as1_output or [])}, "
        f"as2={len(as2_output or [])}, "
        f"as3={len(as3_output or [])} cards"
    )

    try:
        payload = {
            "org_id": org_id,
            "as1_output": as1_output,
            "as2_output": as2_output,
            "as3_output": as3_output,
            "failed_agents": failed_agents,
        }
        result = await _post_json(
            f"{settings.AGA_URL}/consolidate",
            payload,
            timeout=float(settings.AGA_TIMEOUT),
        )
        logger.info(
            f"run_aga completed: score={result.get('overall_compliance_score', 'N/A')}"
        )
        return {"aga_report": result}
    except Exception as exc:
        logger.error(f"run_aga failed: {exc}", exc_info=True)
        return {
            "aga_report": None,
            "failed_agents": failed_agents + ["AGA"],
        }


async def run_aiu(state: GraphState) -> Dict[str, Any]:
    """Call AIU service for initial analysis summary (no user interaction in this step)."""
    settings = get_settings()
    org_id = state["org_id"]
    aga_report = state.get("aga_report")

    if aga_report is None:
        logger.warning("run_aiu: no AGA report available, skipping")
        return {}

    logger.info(f"run_aiu: org_id={org_id}")

    try:
        # Initial greeting/summary message
        payload = {
            "org_id": org_id,
            "message": "Provide a brief executive summary of the gap analysis results and the top 3 most critical actions.",
            "gap_report": aga_report,
            "chat_history": state.get("chat_history", []),
        }
        result = await _post_json(
            f"{settings.AIU_URL}/chat",
            payload,
            timeout=float(settings.AIU_TIMEOUT),
        )
        logger.info("run_aiu completed")
        return {
            "chat_history": result.get("chat_history", []),
            "execution_metadata": {
                **state.get("execution_metadata", {}),
                "aiu_summary": result.get("message", ""),
            },
        }
    except Exception as exc:
        logger.error(f"run_aiu failed: {exc}", exc_info=True)
        return {"failed_agents": state.get("failed_agents", []) + ["AIU"]}


# ---------------------------------------------------------------------------
# LangGraph construction
# ---------------------------------------------------------------------------

def build_graph():
    """
    Build and compile the LangGraph StateGraph.

    Topology:
      START → run_agents_parallel → run_aga → run_aiu → END

    AS-1/2/3 parallelism is handled inside run_agents_parallel via
    asyncio.gather rather than LangGraph fan-out edges (not supported
    in langgraph 0.1.x with multiple add_edge calls from the same node).
    """
    try:
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(GraphState)

        graph.add_node("run_agents_parallel", run_agents_parallel)
        graph.add_node("run_aga", run_aga)
        graph.add_node("run_aiu", run_aiu)

        graph.add_edge(START, "run_agents_parallel")
        graph.add_edge("run_agents_parallel", "run_aga")
        graph.add_edge("run_aga", "run_aiu")
        graph.add_edge("run_aiu", END)

        compiled = graph.compile()
        logger.info("LangGraph compiled successfully")
        return compiled

    except Exception as exc:
        logger.error(f"Failed to build LangGraph: {exc}", exc_info=True)
        raise


# Module-level compiled graph instance (lazy)
_compiled_graph = None


def get_graph():
    """Get or build the compiled graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def run_analysis_pipeline(
    org_id: str,
    documents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run the full analysis pipeline for the given org and documents.

    Args:
        org_id: Organization identifier.
        documents: List of document dicts with keys: filename, content, metadata.

    Returns:
        Final GraphState as dict.
    """
    graph = get_graph()

    initial_state: GraphState = {
        "org_id": org_id,
        "org_docs_path": "",
        "documents": documents,
        "as1_output": None,
        "as2_output": None,
        "as3_output": None,
        "aga_report": None,
        "chat_history": [],
        "execution_metadata": {"start_time": __import__("datetime").datetime.utcnow().isoformat() + "Z"},
        "failed_agents": [],
    }

    logger.info(f"Starting analysis pipeline: org_id={org_id}, docs={len(documents)}")
    final_state = await graph.ainvoke(initial_state)
    logger.info(f"Analysis pipeline completed: org_id={org_id}")
    return final_state
