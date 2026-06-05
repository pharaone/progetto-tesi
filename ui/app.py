"""Streamlit UI for ISO/IEC 42001 Gap Analysis System.

Features:
- Sidebar: org_id input + multi-file document upload
- "Avvia Analisi" button triggers full pipeline
- Dashboard: compliance score, verdict counts, prioritized gaps table, expandable evaluation cards
- Chat interface with AIU for interactive Q&A
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import streamlit as st

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")
AIU_URL = os.getenv("AIU_URL", "http://localhost:8005")

st.set_page_config(
    page_title="ISO/IEC 42001 Gap Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("ISO/IEC 42001")
    st.subheader("Gap Analysis System")
    st.divider()

    org_id = st.text_input(
        "Organization ID",
        placeholder="e.g. acme-corp-2024",
        help="Unique identifier for your organization",
    )

    uploaded_files = st.file_uploader(
        "Upload Organizational Documents",
        accept_multiple_files=True,
        type=["txt", "pdf", "md"],
        help="Upload policies, procedures, risk assessments, and other relevant documents (PDF or TXT)",
    )

    st.divider()
    st.caption("ISO/IEC 42001:2023 AI Management System")
    st.caption("Coverage: Clauses 4-10 + Annex A")

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) ready")
        for f in uploaded_files:
            st.caption(f"• {f.name}")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("ISO/IEC 42001 Compliance Gap Analysis")

# Initialize session state
if "gap_report" not in st.session_state:
    st.session_state.gap_report = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = False


def _verdict_color(verdict: str) -> str:
    colors = {
        "CONFORME": "green",
        "NON_CONFORME": "red",
        "PARZIALMENTE_CONFORME": "orange",
        "NON_APPLICABILE": "gray",
    }
    return colors.get(verdict, "blue")


def _verdict_badge(verdict: str) -> str:
    colors = {
        "CONFORME": "normal",
        "NON_CONFORME": "off",
        "PARZIALMENTE_CONFORME": "inverse",
        "NON_APPLICABILE": "off",
    }
    return verdict.replace("_", " ")


def run_analysis(org_id: str, files) -> Optional[Dict[str, Any]]:
    """Call orchestrator /analyze endpoint with multipart form."""
    try:
        file_tuples = []
        for f in files:
            file_content = f.read()
            file_tuples.append(("files", (f.name, file_content, "application/octet-stream")))

        with httpx.Client(timeout=600.0) as client:
            response = client.post(
                f"{ORCHESTRATOR_URL}/analyze",
                data={"org_id": org_id},
                files=file_tuples,
            )
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        st.error("Analysis timed out. The pipeline may still be running. Please try again later.")
        return None
    except httpx.HTTPStatusError as exc:
        st.error(f"Analysis failed (HTTP {exc.response.status_code}): {exc.response.text}")
        return None
    except Exception as exc:
        st.error(f"Analysis error: {str(exc)}")
        return None


def send_chat_message(
    org_id: str,
    message: str,
    gap_report: Dict,
    chat_history: List,
) -> Optional[Dict[str, Any]]:
    """Send a chat message to the AIU service."""
    try:
        payload = {
            "org_id": org_id,
            "message": message,
            "gap_report": gap_report,
            "chat_history": chat_history,
        }
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{AIU_URL}/chat", json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        st.error("Chat response timed out. Please try again.")
        return None
    except Exception as exc:
        st.error(f"Chat error: {str(exc)}")
        return None


# ---------------------------------------------------------------------------
# Analysis trigger
# ---------------------------------------------------------------------------

col_btn, col_status = st.columns([1, 3])

with col_btn:
    start_btn = st.button(
        "Avvia Analisi",
        type="primary",
        disabled=not (org_id and uploaded_files),
        use_container_width=True,
    )

with col_status:
    if not org_id:
        st.info("Enter an Organization ID in the sidebar to begin.")
    elif not uploaded_files:
        st.info("Upload at least one document in the sidebar.")
    elif st.session_state.gap_report:
        score = st.session_state.gap_report.get("overall_compliance_score", 0)
        st.success(f"Last analysis completed. Score: {score:.1f}/100")

if start_btn and org_id and uploaded_files:
    with st.spinner("Running ISO 42001 gap analysis... This may take several minutes."):
        # Reset file pointers (Streamlit may have already read them)
        for f in uploaded_files:
            f.seek(0)

        report = run_analysis(org_id, uploaded_files)
        if report:
            st.session_state.gap_report = report
            st.session_state.chat_history = []
            st.success("Analysis completed successfully!")
            st.rerun()

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

if st.session_state.gap_report:
    report = st.session_state.gap_report
    st.divider()
    st.header("Compliance Dashboard")

    # Executive summary (if available)
    exec_summary = report.get("execution_metadata", {}).get("executive_summary", "")
    if exec_summary:
        st.info(exec_summary)

    # Partial coverage warning
    if report.get("partial_coverage"):
        failed = ", ".join(report.get("failed_agents", []))
        st.warning(f"Partial coverage — the following agents failed: {failed}. Results are incomplete.")

    # Compliance score
    st.subheader("Overall Compliance Score")
    score = report.get("overall_compliance_score", 0.0)
    counts = report.get("counts", {})

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Compliance Score", f"{score:.1f}/100")
    col2.metric("Conforme", counts.get("compliant", 0), delta=None)
    col3.metric(
        "Non Conforme",
        counts.get("non_compliant", 0),
        delta=None,
        delta_color="inverse",
    )
    col4.metric("Parzialmente", counts.get("partial", 0), delta=None)
    col5.metric("Non Applicabile", counts.get("not_applicable", 0))

    # Progress bar for score
    st.progress(score / 100.0, text=f"Compliance: {score:.1f}%")

    st.divider()

    # Prioritized gaps table
    prioritized_gaps = report.get("prioritized_gaps", [])
    if prioritized_gaps:
        st.subheader(f"Prioritized Gaps ({len(prioritized_gaps)} total)")

        gaps_data = []
        for gap in prioritized_gaps:
            verdict = gap.get("verdict", "")
            gaps_text = "; ".join(gap.get("gaps", [])[:2]) if gap.get("gaps") else "—"
            ca = gap.get("corrective_action", {})
            action = ca.get("description", "—")[:100] + "..." if len(ca.get("description", "")) > 100 else ca.get("description", "—")
            doc_type = ca.get("expected_document_type", "—")

            severity_label = "CRITICO" if gap.get("severity") == 1 else "MODERATO"

            gaps_data.append({
                "Priorità": gap.get("severity", 0),
                "Severità": severity_label,
                "Requisito": gap.get("requirement_id", ""),
                "Verdict": verdict.replace("_", " "),
                "Gap Identificati": gaps_text,
                "Azione Correttiva": action,
                "Documento Atteso": doc_type,
            })

        df = pd.DataFrame(gaps_data)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Priorità": st.column_config.NumberColumn(width="small"),
                "Severità": st.column_config.TextColumn(width="small"),
                "Requisito": st.column_config.TextColumn(width="small"),
                "Verdict": st.column_config.TextColumn(width="medium"),
                "Gap Identificati": st.column_config.TextColumn(width="large"),
                "Azione Correttiva": st.column_config.TextColumn(width="large"),
                "Documento Atteso": st.column_config.TextColumn(width="medium"),
            },
        )
    else:
        st.success("No compliance gaps identified!")

    st.divider()

    # Evaluation cards by clause
    all_cards = report.get("evaluation_cards", [])
    if all_cards:
        st.subheader("Evaluation Cards by Clause Group")

        # Group cards by clause prefix
        groups: Dict[str, List] = {
            "Clause 4 (Context)": [],
            "Clause 5 (Leadership)": [],
            "Clause 6 (Planning)": [],
            "Clause 7 (Support)": [],
            "Clause 8 (Operations)": [],
            "Clause 9 (Performance)": [],
            "Clause 10 (Improvement)": [],
            "Annex A.2": [],
            "Annex A.3": [],
            "Annex A.4": [],
            "Annex A.5": [],
            "Annex A.6": [],
            "Annex A.7": [],
            "Annex A.8": [],
            "Annex A.9": [],
            "Annex A.10": [],
        }

        for card in all_cards:
            req_id = card.get("requirement_id", "")
            if req_id.startswith("cl-4"):
                groups["Clause 4 (Context)"].append(card)
            elif req_id.startswith("cl-5"):
                groups["Clause 5 (Leadership)"].append(card)
            elif req_id.startswith("cl-6"):
                groups["Clause 6 (Planning)"].append(card)
            elif req_id.startswith("cl-7"):
                groups["Clause 7 (Support)"].append(card)
            elif req_id.startswith("cl-8"):
                groups["Clause 8 (Operations)"].append(card)
            elif req_id.startswith("cl-9"):
                groups["Clause 9 (Performance)"].append(card)
            elif req_id.startswith("cl-10"):
                groups["Clause 10 (Improvement)"].append(card)
            elif req_id.startswith("A.2"):
                groups["Annex A.2"].append(card)
            elif req_id.startswith("A.3"):
                groups["Annex A.3"].append(card)
            elif req_id.startswith("A.4"):
                groups["Annex A.4"].append(card)
            elif req_id.startswith("A.5"):
                groups["Annex A.5"].append(card)
            elif req_id.startswith("A.6"):
                groups["Annex A.6"].append(card)
            elif req_id.startswith("A.7"):
                groups["Annex A.7"].append(card)
            elif req_id.startswith("A.8"):
                groups["Annex A.8"].append(card)
            elif req_id.startswith("A.9"):
                groups["Annex A.9"].append(card)
            elif req_id.startswith("A.10"):
                groups["Annex A.10"].append(card)

        for group_name, cards in groups.items():
            if not cards:
                continue

            # Compute group summary
            group_conforme = sum(1 for c in cards if c.get("verdict") == "CONFORME")
            group_non_conf = sum(1 for c in cards if c.get("verdict") == "NON_CONFORME")
            group_partial = sum(1 for c in cards if c.get("verdict") == "PARZIALMENTE_CONFORME")

            label = f"{group_name} — {len(cards)} req. | C:{group_conforme} NC:{group_non_conf} P:{group_partial}"

            with st.expander(label, expanded=(group_non_conf > 0)):
                for card in cards:
                    verdict = card.get("verdict", "NON_APPLICABILE")
                    req_id = card.get("requirement_id", "")
                    req_text = card.get("requirement_text", "")
                    gaps = card.get("gaps", [])
                    ca = card.get("corrective_action", {})
                    evidences = card.get("evidences", [])

                    # Verdict badge
                    verdict_display = verdict.replace("_", " ")
                    badge_type = {
                        "CONFORME": "success",
                        "NON_CONFORME": "error",
                        "PARZIALMENTE_CONFORME": "warning",
                        "NON_APPLICABILE": "secondary",
                    }.get(verdict, "secondary")

                    st.markdown(f"**{req_id}** — `{verdict_display}`")

                    if req_text:
                        st.caption(req_text[:200] + "..." if len(req_text) > 200 else req_text)

                    if gaps:
                        st.markdown("**Gaps:**")
                        for gap in gaps:
                            st.markdown(f"  - {gap}")

                    if ca and (ca.get("description") or ca.get("expected_document_type")):
                        st.markdown(
                            f"**Azione correttiva:** {ca.get('description', '—')}  \n"
                            f"**Documento atteso:** `{ca.get('expected_document_type', '—')}`"
                        )

                    if evidences:
                        with st.expander(f"Evidenze ({len(evidences)})", expanded=False):
                            for ev in evidences:
                                st.markdown(
                                    f"- **Fonte:** {ev.get('source_doc', 'unknown')}  \n"
                                    f"  *{ev.get('excerpt', '')[:200]}*"
                                )

                    st.divider()

    # ---------------------------------------------------------------------------
    # Chat section
    # ---------------------------------------------------------------------------
    st.divider()
    st.header("Consulente ISO 42001 — Chat")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            with st.chat_message(role):
                st.markdown(content)

    # Chat input
    user_input = st.chat_input(
        "Fai una domanda sull'analisi di conformità...",
        key="chat_input",
    )

    if user_input and org_id:
        # Display user message immediately
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        with st.spinner("Elaborazione risposta..."):
            result = send_chat_message(
                org_id=org_id,
                message=user_input,
                gap_report=report,
                chat_history=st.session_state.chat_history,
            )

        if result:
            assistant_msg = result.get("message", "")
            st.session_state.chat_history = result.get("chat_history", [])

            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(assistant_msg)

            st.rerun()

    elif user_input and not org_id:
        st.warning("Inserisci un Organization ID nella barra laterale prima di chattare.")

else:
    # No report yet — show instructions
    st.markdown(
        """
        ## Come iniziare

        1. **Inserisci l'Organization ID** nella barra laterale (es. `acme-corp-2024`)
        2. **Carica i documenti** organizzativi (policy, procedure, valutazioni del rischio)
        3. Clicca **Avvia Analisi** per avviare l'analisi di conformità ISO 42001

        ### Cosa viene analizzato

        Il sistema valuta la conformità a **tutti i requisiti ISO/IEC 42001:2023**:

        | Agente | Clausole | Requisiti |
        |--------|----------|-----------|
        | AS-1 | 4 (Contesto), 5 (Leadership), 6 (Pianificazione) | 10 |
        | AS-2 | 7 (Supporto), 8 (Operazioni) + Annex A.2-A.6 | 24 |
        | AS-3 | 9 (Valutazione), 10 (Miglioramento) + Annex A.7-A.10 | 19 |

        ### Output

        - **Punteggio di conformità** (0-100)
        - **Schede di valutazione** per ogni requisito
        - **Piano d'azione** prioritizzato
        - **Consulente conversazionale** per approfondimenti
        """
    )
