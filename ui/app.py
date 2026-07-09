"""Streamlit UI for ISO/IEC 42001 Gap Analysis System.

Single-company deployment with two authenticated roles:

- Employee:  uploads documents (sees only their own), starts the analysis,
             views APPROVED reports, chats with the AIU consultant.
- Certifier: reviews pending gap reports and approves/rejects them before
             they become visible to employees.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import streamlit as st

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ISO/IEC 42001 Gap Analysis",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

for key, default in [
    ("token", None),
    ("username", None),
    ("role", None),
    ("chat_histories", {}),  # report_id -> list of messages
    ("analysis_notice", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def api_post(path: str, json: Optional[dict] = None, timeout: float = 60.0, **kwargs) -> Optional[dict]:
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{ORCHESTRATOR_URL}{path}",
                json=json,
                headers=_auth_headers() if st.session_state.token else {},
                **kwargs,
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException:
        st.error("Richiesta scaduta (timeout). Riprova più tardi.")
    except httpx.HTTPStatusError as exc:
        _show_http_error(exc)
    except Exception as exc:
        st.error(f"Errore: {exc}")
    return None


def api_get(path: str, timeout: float = 30.0) -> Optional[Any]:
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{ORCHESTRATOR_URL}{path}", headers=_auth_headers())
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        _show_http_error(exc)
    except Exception as exc:
        st.error(f"Errore: {exc}")
    return None


def api_delete(path: str, timeout: float = 30.0) -> Optional[dict]:
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.delete(f"{ORCHESTRATOR_URL}{path}", headers=_auth_headers())
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        _show_http_error(exc)
    except Exception as exc:
        st.error(f"Errore: {exc}")
    return None


def _show_http_error(exc: httpx.HTTPStatusError) -> None:
    if exc.response.status_code == 401:
        st.session_state.token = None
        st.session_state.username = None
        st.session_state.role = None
        st.error("Sessione scaduta. Effettua di nuovo il login.")
    else:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except Exception:
            detail = exc.response.text
        st.error(f"Errore (HTTP {exc.response.status_code}): {detail}")


def logout() -> None:
    st.session_state.token = None
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.chat_histories = {}
    st.session_state.analysis_notice = None


# ---------------------------------------------------------------------------
# Login page
# ---------------------------------------------------------------------------

def render_login_page() -> None:
    st.title("ISO/IEC 42001 Compliance Gap Analysis")
    st.caption("Accedi per continuare")

    tab_login, tab_register = st.tabs(["Accedi", "Registrati (dipendente)"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Accedi", type="primary", use_container_width=True)

        if submitted:
            if not username or not password:
                st.warning("Inserisci username e password.")
            else:
                result = api_post("/auth/login", json={"username": username, "password": password})
                if result:
                    st.session_state.token = result["token"]
                    st.session_state.username = result["username"]
                    st.session_state.role = result["role"]
                    st.rerun()

    with tab_register:
        st.caption("La registrazione crea un account **dipendente**. L'account del certificatore è configurato dall'amministratore.")
        with st.form("register_form"):
            new_username = st.text_input("Username", key="reg_username")
            new_password = st.text_input("Password (min 6 caratteri)", type="password", key="reg_password")
            new_password2 = st.text_input("Conferma password", type="password", key="reg_password2")
            reg_submitted = st.form_submit_button("Registrati", use_container_width=True)

        if reg_submitted:
            if not new_username or not new_password:
                st.warning("Inserisci username e password.")
            elif new_password != new_password2:
                st.warning("Le password non coincidono.")
            elif len(new_password) < 6:
                st.warning("La password deve avere almeno 6 caratteri.")
            else:
                result = api_post("/auth/register", json={"username": new_username, "password": new_password})
                if result:
                    st.session_state.token = result["token"]
                    st.session_state.username = result["username"]
                    st.session_state.role = result["role"]
                    st.success("Registrazione completata!")
                    st.rerun()


# ---------------------------------------------------------------------------
# Report dashboard rendering (shared by employee and certifier views)
# ---------------------------------------------------------------------------

def render_report_dashboard(report: Dict[str, Any]) -> None:
    """Render the compliance dashboard for a gap report (the inner report JSON)."""
    exec_summary = report.get("execution_metadata", {}).get("executive_summary", "")
    if exec_summary:
        st.info(exec_summary)

    if report.get("partial_coverage"):
        failed = ", ".join(report.get("failed_agents", []))
        st.warning(f"Copertura parziale — agenti falliti: {failed}. Risultati incompleti.")

    st.subheader("Punteggio di conformità")
    score = report.get("overall_compliance_score", 0.0)
    counts = report.get("counts", {})

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Compliance Score", f"{score:.1f}/100")
    col2.metric("Conforme", counts.get("compliant", 0))
    col3.metric("Non Conforme", counts.get("non_compliant", 0))
    col4.metric("Parzialmente", counts.get("partial", 0))
    col5.metric("Non Applicabile", counts.get("not_applicable", 0))

    st.progress(min(score / 100.0, 1.0), text=f"Compliance: {score:.1f}%")
    st.divider()

    prioritized_gaps = report.get("prioritized_gaps", [])
    if prioritized_gaps:
        st.subheader(f"Gap prioritizzati ({len(prioritized_gaps)} totali)")

        gaps_data = []
        for gap in prioritized_gaps:
            verdict = gap.get("verdict", "")
            gaps_text = "; ".join(gap.get("gaps", [])[:2]) if gap.get("gaps") else "—"
            ca = gap.get("corrective_action", {})
            action = ca.get("description", "—")
            if len(action) > 100:
                action = action[:100] + "..."
            gaps_data.append({
                "Priorità": gap.get("severity", 0),
                "Severità": "CRITICO" if gap.get("severity") == 1 else "MODERATO",
                "Requisito": gap.get("requirement_id", ""),
                "Verdict": verdict.replace("_", " "),
                "Gap Identificati": gaps_text,
                "Azione Correttiva": action,
                "Documento Atteso": ca.get("expected_document_type", "—"),
            })

        st.dataframe(pd.DataFrame(gaps_data), use_container_width=True, hide_index=True)
    else:
        st.success("Nessun gap di conformità identificato!")

    st.divider()

    all_cards = report.get("evaluation_cards", [])
    if all_cards:
        st.subheader("Schede di valutazione per gruppo di clausole")

        group_order = [
            ("cl-4", "Clause 4 (Context)"),
            ("cl-5", "Clause 5 (Leadership)"),
            ("cl-6", "Clause 6 (Planning)"),
            ("cl-7", "Clause 7 (Support)"),
            ("cl-8", "Clause 8 (Operations)"),
            ("cl-9", "Clause 9 (Performance)"),
            ("cl-10", "Clause 10 (Improvement)"),
            ("A.2", "Annex A.2"),
            ("A.3", "Annex A.3"),
            ("A.4", "Annex A.4"),
            ("A.5", "Annex A.5"),
            ("A.6", "Annex A.6"),
            ("A.7", "Annex A.7"),
            ("A.8", "Annex A.8"),
            ("A.9", "Annex A.9"),
            ("A.10", "Annex A.10"),
        ]

        groups: Dict[str, List] = {label: [] for _, label in group_order}
        for card in all_cards:
            req_id = card.get("requirement_id", "")
            # Longest prefix first so cl-10 isn't captured by cl-1
            for prefix, label in sorted(group_order, key=lambda x: -len(x[0])):
                if req_id.startswith(prefix):
                    groups[label].append(card)
                    break

        for _, group_name in group_order:
            cards = groups[group_name]
            if not cards:
                continue

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

                    st.markdown(f"**{req_id}** — `{verdict.replace('_', ' ')}`")

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


def render_chat(report_id: int, report: Dict[str, Any]) -> None:
    """Chat with the AIU consultant about a specific report."""
    st.header("Consulente ISO 42001 — Chat")

    histories = st.session_state.chat_histories
    history = histories.get(report_id, [])

    chat_container = st.container()
    with chat_container:
        for msg in history:
            with st.chat_message(msg.get("role", "user")):
                st.markdown(msg.get("content", ""))

    user_input = st.chat_input(
        "Fai una domanda sull'analisi di conformità...",
        key=f"chat_input_{report_id}",
    )

    if user_input:
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        with st.spinner("Elaborazione risposta..."):
            result = api_post(
                "/chat",
                json={
                    "org_id": "",  # fixed server-side (single-company instance)
                    "message": user_input,
                    "gap_report": report,
                    "chat_history": history,
                },
                timeout=180.0,
            )

        if result:
            histories[report_id] = result.get("chat_history", [])
            st.rerun()


def render_report_list_and_detail(
    reports: List[dict],
    key_prefix: str,
    show_chat: bool = True,
    review_controls: bool = False,
) -> None:
    """Shared report browser: selectbox → dashboard (+ optional chat / review)."""
    if not reports:
        st.info("Nessun report disponibile.")
        return

    options = {
        f"Report #{r['id']} — {r['status']} — {r['created_at'][:19]} "
        f"(score: {r.get('overall_compliance_score') or '—'})": r["id"]
        for r in reports
    }
    selected_label = st.selectbox(
        "Seleziona un report",
        list(options.keys()),
        key=f"{key_prefix}_select",
    )
    report_id = options[selected_label]

    detail = api_get(f"/reports/{report_id}")
    if not detail:
        return

    meta_cols = st.columns(4)
    meta_cols[0].markdown(f"**Stato:** `{detail['status']}`")
    meta_cols[1].markdown(f"**Creato da:** {detail['created_by']}")
    if detail.get("reviewed_by"):
        meta_cols[2].markdown(f"**Revisionato da:** {detail['reviewed_by']}")
    if detail.get("review_comment"):
        meta_cols[3].markdown(f"**Commento:** {detail['review_comment']}")

    if review_controls and detail["status"] == "PENDING_REVIEW":
        st.divider()
        st.subheader("Revisione del certificatore")
        st.caption(
            "Verifica i risultati dell'analisi qui sotto. Approvando il report, "
            "questo diventerà visibile ai dipendenti."
        )
        comment = st.text_area("Commento (opzionale)", key=f"{key_prefix}_comment_{report_id}")
        col_a, col_r = st.columns(2)
        with col_a:
            if st.button("✅ Approva report", type="primary", key=f"{key_prefix}_approve_{report_id}", use_container_width=True):
                result = api_post(f"/reports/{report_id}/review", json={"approve": True, "comment": comment})
                if result:
                    st.success(f"Report #{report_id} approvato.")
                    st.rerun()
        with col_r:
            if st.button("❌ Rifiuta report", key=f"{key_prefix}_reject_{report_id}", use_container_width=True):
                result = api_post(f"/reports/{report_id}/review", json={"approve": False, "comment": comment})
                if result:
                    st.warning(f"Report #{report_id} rifiutato.")
                    st.rerun()

    st.divider()
    render_report_dashboard(detail["report"])

    if show_chat:
        st.divider()
        render_chat(report_id, detail["report"])


# ---------------------------------------------------------------------------
# Documents section
# ---------------------------------------------------------------------------

def render_documents_section(can_upload: bool) -> None:
    if can_upload:
        st.subheader("Carica documenti")
        uploaded_files = st.file_uploader(
            "Carica documenti organizzativi (policy, procedure, valutazioni del rischio)",
            accept_multiple_files=True,
            type=["txt", "pdf", "md"],
            key="doc_uploader",
        )
        if uploaded_files and st.button("Salva documenti", type="primary"):
            file_tuples = []
            for f in uploaded_files:
                f.seek(0)
                file_tuples.append(("files", (f.name, f.read(), "application/octet-stream")))
            try:
                with httpx.Client(timeout=120.0) as client:
                    resp = client.post(
                        f"{ORCHESTRATOR_URL}/documents",
                        files=file_tuples,
                        headers=_auth_headers(),
                    )
                    resp.raise_for_status()
                    saved = resp.json().get("uploaded", [])
                    st.success(f"{len(saved)} documento/i caricato/i.")
                    st.rerun()
            except httpx.HTTPStatusError as exc:
                _show_http_error(exc)
            except Exception as exc:
                st.error(f"Errore durante il caricamento: {exc}")

        st.divider()

    if st.session_state.role == "certifier":
        st.subheader("Tutti i documenti aziendali")
    else:
        st.subheader("I miei documenti")

    docs = api_get("/documents")
    if docs is None:
        return
    if not docs:
        st.info("Nessun documento caricato.")
        return

    for doc in docs:
        cols = st.columns([4, 2, 2, 1])
        cols[0].markdown(f"📄 **{doc['filename']}**")
        cols[1].caption(f"Caricato da: {doc['uploader']}")
        cols[2].caption(doc["uploaded_at"][:19])
        if cols[3].button("🗑️", key=f"del_doc_{doc['id']}", help="Elimina documento"):
            if api_delete(f"/documents/{doc['id']}"):
                st.rerun()


# ---------------------------------------------------------------------------
# Employee view
# ---------------------------------------------------------------------------

def render_employee_view() -> None:
    tab_docs, tab_analysis, tab_reports = st.tabs(
        ["📄 Documenti", "🔍 Analisi", "📊 Report approvati"]
    )

    with tab_docs:
        render_documents_section(can_upload=True)

    with tab_analysis:
        st.subheader("Avvia l'analisi di conformità")
        st.caption(
            "L'analisi valuta TUTTI i documenti aziendali caricati (di tutti i dipendenti) "
            "rispetto ai requisiti ISO/IEC 42001. Il report risultante sarà visibile solo "
            "dopo l'approvazione del certificatore."
        )

        if st.session_state.analysis_notice:
            st.info(st.session_state.analysis_notice)

        if st.button("Avvia Analisi", type="primary"):
            with st.spinner("Analisi ISO 42001 in corso... Può richiedere diversi minuti."):
                result = api_post("/analyze", timeout=3600.0)
            if result:
                st.session_state.analysis_notice = result.get(
                    "message", "Analisi completata, in attesa di revisione."
                )
                st.success(
                    f"Analisi completata (report #{result.get('report_id')}). "
                    "Il report è in attesa di verifica da parte del certificatore."
                )

    with tab_reports:
        st.subheader("Report approvati dal certificatore")
        reports = api_get("/reports")
        if reports is not None:
            render_report_list_and_detail(reports, key_prefix="emp", show_chat=True)


# ---------------------------------------------------------------------------
# Certifier view
# ---------------------------------------------------------------------------

def render_certifier_view() -> None:
    tab_pending, tab_all, tab_docs = st.tabs(
        ["🔎 Da revisionare", "📊 Tutti i report", "📄 Documenti"]
    )

    with tab_pending:
        st.subheader("Report in attesa di revisione")
        reports = api_get("/reports")
        if reports is not None:
            pending = [r for r in reports if r["status"] == "PENDING_REVIEW"]
            render_report_list_and_detail(
                pending, key_prefix="cert_pending", show_chat=False, review_controls=True
            )

    with tab_all:
        st.subheader("Storico report")
        reports = api_get("/reports")
        if reports is not None:
            render_report_list_and_detail(
                reports, key_prefix="cert_all", show_chat=True, review_controls=True
            )

    with tab_docs:
        render_documents_section(can_upload=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if not st.session_state.token:
    render_login_page()
else:
    with st.sidebar:
        st.title("ISO/IEC 42001")
        st.subheader("Gap Analysis System")
        st.divider()

        role_label = "Certificatore" if st.session_state.role == "certifier" else "Dipendente"
        st.markdown(f"👤 **{st.session_state.username}**")
        st.caption(f"Ruolo: {role_label}")

        if st.button("Esci", use_container_width=True):
            logout()
            st.rerun()

        st.divider()
        st.caption("ISO/IEC 42001:2023 AI Management System")
        st.caption("Coverage: Clauses 4-10 + Annex A")

    st.title("ISO/IEC 42001 Compliance Gap Analysis")

    if st.session_state.role == "certifier":
        render_certifier_view()
    else:
        render_employee_view()
