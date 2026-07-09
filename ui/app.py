"""Streamlit UI for ISO/IEC 42001 Gap Analysis System.

Single-company deployment with two authenticated roles:

- Employee:  uploads documents (sees only their own), starts the analysis,
             views APPROVED reports, chats with the AIU consultant.
- Certifier: reviews pending gap reports and approves/rejects them before
             they become visible to employees.

Design notes:
- Icons are Google Material Symbols (Apache 2.0), bundled by Streamlit and
  rendered offline via the ":material/name:" markdown directive — no emojis,
  no external CDN.
- Login is a two-panel page: product explanation + compact auth card.
- After login: slim top bar (no sidebar) and pill-styled tabs.
- Chat opens in a modal dialog from the report view.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import streamlit as st

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")

# How long the UI waits for the full analysis pipeline (seconds).
# Should be >= AS_TIMEOUT since the agents run in parallel behind it.
ANALYZE_TIMEOUT = float(os.getenv("UI_ANALYZE_TIMEOUT", "3600"))

st.set_page_config(
    page_title="ISO/IEC 42001 Gap Analysis",
    page_icon=":material/verified:",
    layout="wide",
    initial_sidebar_state="collapsed",
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
    ("chat_report_id", None),
    ("chat_report", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

_BASE_CSS = """
<style>
/* Hide default Streamlit chrome for a cleaner app look */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {height: 0; visibility: hidden;}

/* Tighter vertical rhythm */
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
</style>
"""

_APP_CSS = """
<style>
/* Minimal gap above the sticky top bar */
.block-container {padding-top: 0.4rem;}

/* ---- Modern pill tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    padding: 4px 0 10px 0;
}
.stTabs [data-baseweb="tab"] {
    height: 40px;
    border-radius: 20px;
    padding: 0 22px;
    background-color: #F4F6FA;
    border: 1px solid #E3E8F0;
    font-weight: 500;
    color: #3D4451;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #E8EDF5;
    color: #1A1D23;
}
.stTabs [aria-selected="true"] {
    background-color: #2563EB !important;
    border-color: #2563EB !important;
    color: #FFFFFF !important;
}
/* Remove the default underline/highlight bar of the tab bar */
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* ---- Sticky top bar ---- */
/* st.container(key="topbar") tags the wrapper with the st-key-topbar class */
.st-key-topbar {
    position: sticky;
    top: 0;
    z-index: 999;
    background: #FFFFFF;
    box-shadow: 0 2px 10px rgba(16, 42, 100, 0.06);
}
</style>
"""

_LOGIN_CSS = """
<style>
.block-container {max-width: 1080px; padding-top: 3.2rem;}

/* Auth card */
div[data-testid="stForm"] {
    border: 1px solid #E3E8F0;
    border-radius: 14px;
    padding: 1.5rem 1.5rem 1.1rem 1.5rem;
    box-shadow: 0 8px 24px rgba(16, 42, 100, 0.08);
    background: #FFFFFF;
}

/* Feature rows on the left panel */
.feature-row {
    display: flex;
    align-items: flex-start;
    gap: 0.65rem;
    margin-bottom: 0.9rem;
    line-height: 1.35;
}
.feature-row .material-symbols-rounded {color: #2563EB;}

.login-hero-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.login-hero-sub {
    color: #5A6272;
    font-size: 1.02rem;
    margin-bottom: 1.6rem;
}
.roles-box {
    border: 1px solid #E3E8F0;
    border-radius: 10px;
    background: #F4F6FA;
    padding: 0.9rem 1.1rem;
    font-size: 0.9rem;
    color: #3D4451;
    margin-top: 1.2rem;
}
</style>
"""

st.markdown(_BASE_CSS, unsafe_allow_html=True)


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
    st.session_state.chat_report_id = None
    st.session_state.chat_report = None


# ---------------------------------------------------------------------------
# Login page — two panels: product explanation + auth card
# ---------------------------------------------------------------------------

def render_login_page() -> None:
    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)

    col_info, col_form = st.columns([1.25, 1], gap="large")

    with col_info:
        # NB: the :material/...: directive only works in plain markdown,
        # not inside raw HTML blocks
        st.markdown("## :material/verified: Compliance AI by Design")
        st.markdown(
            '<div class="login-hero-sub">Sistema intelligente multi-agente per la '
            "gap analysis e la verifica dei requisiti dello standard "
            "<b>ISO/IEC 42001:2023</b>, il primo standard internazionale certificabile "
            "per i sistemi di gestione dell'intelligenza artificiale "
            "(AI Management System). Il sistema valuta la documentazione organizzativa "
            "rispetto alle Clausole 4–10 e ai controlli dell'Annex A, producendo "
            "valutazioni tracciabili e ancorate a evidenze documentali.</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            ":material/smart_toy: **Agenti Specializzati (AS-1, AS-2, AS-3)** — tre agenti "
            "analizzano in parallelo le rispettive porzioni della norma: Contesto e Governance "
            "(cl. 4–6), Supporto e Operazioni (cl. 7–8, controlli A.2–A.6), Valutazione e "
            "Controlli Avanzati (cl. 9–10, controlli A.7–A.10)."
        )
        st.markdown(
            ":material/hub: **Agente di Gap Analysis (AGA)** — consolida le schede di "
            "valutazione, verifica la coerenza interna e le interdipendenze tra requisiti, "
            "e produce la prioritizzazione delle lacune con il relativo piano d'azione."
        )
        st.markdown(
            ":material/forum: **Agente di Interfaccia Utente (AIU)** — consulente "
            "conversazionale che rende navigabile il report: approfondisci gap, priorità "
            "e azioni correttive in linguaggio naturale (human-in-the-loop)."
        )
        st.markdown(
            ":material/upload_file: **Documenti riservati per dipendente** — ogni dipendente "
            "carica e gestisce i propri documenti (policy, procedure, registri di rischio, "
            "contratti); l'analisi considera l'intero corpus aziendale, ma la visibilità "
            "dei file resta individuale."
        )
        st.markdown(
            ":material/fact_check: **Verifica del certificatore** — ogni report di gap "
            "analysis è validato da un certificatore umano prima della pubblicazione ai "
            "dipendenti, a garanzia dell'affidabilità delle valutazioni."
        )
        st.markdown(
            ":material/trending_up: **Miglioramento continuo** — i chiarimenti forniti sulle "
            "non conformità diventano documentazione organizzativa e arricchiscono il "
            "contesto delle analisi successive."
        )

        st.markdown(
            '<div class="roles-box"><b>Come accedere</b><br>'
            "<b>Dipendente</b>: crea un account dalla scheda <i>Registrati</i> per caricare "
            "documenti, avviare l'analisi e consultare i report approvati.<br>"
            "<b>Certificatore</b>: usa le credenziali fornite dall'amministratore di sistema "
            "per revisionare e approvare i report.</div>",
            unsafe_allow_html=True,
        )

    with col_form:
        tab_login, tab_register = st.tabs(["Accedi", "Registrati"])

        with tab_login:
            with st.form("login_form"):
                st.markdown("##### Accedi al tuo account")
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
            with st.form("register_form"):
                st.markdown("##### Crea un account dipendente")
                new_username = st.text_input("Username", key="reg_username")
                new_password = st.text_input("Password (min 6 caratteri)", type="password", key="reg_password")
                new_password2 = st.text_input("Conferma password", type="password", key="reg_password2")
                reg_submitted = st.form_submit_button("Registrati", type="primary", use_container_width=True)

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
# Top bar
# ---------------------------------------------------------------------------

def render_top_bar() -> None:
    is_certifier = st.session_state.role == "certifier"
    role_label = "Certificatore" if is_certifier else "Dipendente"
    role_icon = ":material/verified_user:" if is_certifier else ":material/person:"

    with st.container(border=True, key="topbar"):
        col_title, col_user, col_logout = st.columns([6, 3, 1], vertical_alignment="center")
        with col_title:
            st.markdown(
                ":material/verified: **ISO/IEC 42001 — Gap Analysis**  \n"
                "<span style='color: #5A6272; font-size: 0.8rem;'>"
                "Sistema multi-agente per l'AI Management System · Clausole 4–10 + Annex A</span>",
                unsafe_allow_html=True,
            )
        with col_user:
            st.markdown(
                f"{role_icon} **{st.session_state.username}**  \n"
                f"<span style='color: #5A6272; font-size: 0.8rem;'>{role_label}</span>",
                unsafe_allow_html=True,
            )
        with col_logout:
            if st.button("Esci", use_container_width=True):
                logout()
                st.rerun()


# ---------------------------------------------------------------------------
# Chat dialog (popup)
# ---------------------------------------------------------------------------

@st.dialog("Consulente ISO 42001", width="large")
def chat_dialog() -> None:
    report_id = st.session_state.chat_report_id
    report = st.session_state.chat_report
    if report_id is None or report is None:
        st.error("Nessun report selezionato.")
        return

    histories = st.session_state.chat_histories
    history = histories.get(report_id, [])

    # Messages area (filled after handling the form so new replies appear
    # without needing an extra rerun)
    messages_area = st.container(height=420)

    with st.form(key=f"chat_form_{report_id}", clear_on_submit=True):
        col_input, col_send = st.columns([5, 1], vertical_alignment="bottom")
        with col_input:
            message = st.text_input(
                "Messaggio",
                placeholder="Fai una domanda sull'analisi di conformità...",
                label_visibility="collapsed",
            )
        with col_send:
            sent = st.form_submit_button("Invia", type="primary", use_container_width=True)

    if sent and message.strip():
        with st.spinner("Elaborazione risposta..."):
            result = api_post(
                "/chat",
                json={
                    "org_id": "",  # fixed server-side (single-company instance)
                    "message": message.strip(),
                    "gap_report": report,
                    "chat_history": history,
                },
                timeout=300.0,
            )
        if result:
            history = result.get("chat_history", [])
            histories[report_id] = history

    with messages_area:
        if not history:
            st.caption(
                "Stai dialogando con l'Agente di Interfaccia Utente (AIU): traduce le tue "
                "domande in linguaggio naturale in interrogazioni sul report consolidato. "
                "Chiedi spiegazioni dei gap, priorità delle azioni correttive o suggerimenti "
                "sui documenti da produrre."
            )
        for msg in history:
            with st.chat_message(msg.get("role", "user")):
                st.markdown(msg.get("content", ""))


def open_chat_button(report_id: int, report: Dict[str, Any], key: str) -> None:
    if st.button("Chat con il consulente", key=key):
        st.session_state.chat_report_id = report_id
        st.session_state.chat_report = report
        chat_dialog()


# ---------------------------------------------------------------------------
# Report dashboard rendering (shared by employee and certifier views)
# ---------------------------------------------------------------------------

def render_report_dashboard(
    report: Dict[str, Any],
    report_id: Optional[int] = None,
    allow_clarifications: bool = False,
) -> None:
    """Render the compliance dashboard for a gap report (the inner report JSON).

    When allow_clarifications is True (employee view on approved reports),
    NON_CONFORME / PARZIALMENTE_CONFORME cards get a form to submit an
    explanation, which is saved among the company documents and used by
    the next analysis.
    """
    exec_summary = report.get("execution_metadata", {}).get("executive_summary", "")
    if exec_summary:
        st.info(exec_summary)

    if report.get("partial_coverage"):
        failed = ", ".join(report.get("failed_agents", []))
        st.warning(f"Copertura parziale — agenti falliti: {failed}. Risultati incompleti.")

    st.markdown("#### :material/speed: Punteggio di conformità")
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
        st.markdown(f"#### :material/priority_high: Gap prioritizzati ({len(prioritized_gaps)} totali)")

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
        st.markdown("#### :material/checklist: Schede di valutazione per gruppo di clausole")

        group_order = [
            ("cl-4", "Clausola 4 — Contesto dell'organizzazione"),
            ("cl-5", "Clausola 5 — Leadership"),
            ("cl-6", "Clausola 6 — Pianificazione"),
            ("cl-7", "Clausola 7 — Supporto"),
            ("cl-8", "Clausola 8 — Operazioni"),
            ("cl-9", "Clausola 9 — Valutazione delle prestazioni"),
            ("cl-10", "Clausola 10 — Miglioramento"),
            ("A.2", "Annex A.2 — Politiche per l'AI"),
            ("A.3", "Annex A.3 — Organizzazione interna"),
            ("A.4", "Annex A.4 — Risorse per i sistemi AI"),
            ("A.5", "Annex A.5 — Valutazione degli impatti"),
            ("A.6", "Annex A.6 — Ciclo di vita dei sistemi AI"),
            ("A.7", "Annex A.7 — Dati per i sistemi AI"),
            ("A.8", "Annex A.8 — Informazioni per le parti interessate"),
            ("A.9", "Annex A.9 — Uso dei sistemi AI"),
            ("A.10", "Annex A.10 — Rapporti con terze parti"),
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
                        # NB: no nested st.expander here — Streamlit forbids
                        # expanders inside expanders (we're already in the
                        # clause-group one)
                        st.markdown(f"**Evidenze ({len(evidences)}):**")
                        for ev in evidences:
                            st.markdown(
                                f"- **Fonte:** {ev.get('source_doc', 'unknown')}  \n"
                                f"  *{ev.get('excerpt', '')[:200]}*"
                            )

                    if (
                        allow_clarifications
                        and report_id is not None
                        and verdict in ("NON_CONFORME", "PARZIALMENTE_CONFORME")
                    ):
                        with st.form(key=f"clar_form_{report_id}_{req_id}"):
                            clar_text = st.text_area(
                                "Aggiungi un chiarimento (es. pratiche esistenti non documentate, "
                                "contesto mancante, riferimenti a procedure interne)",
                                key=f"clar_text_{report_id}_{req_id}",
                                height=100,
                            )
                            submitted = st.form_submit_button("Invia chiarimento")
                        if submitted:
                            if len(clar_text.strip()) < 10:
                                st.warning("Il chiarimento deve contenere almeno 10 caratteri.")
                            else:
                                result = api_post(
                                    f"/reports/{report_id}/clarifications",
                                    json={"requirement_id": req_id, "text": clar_text.strip()},
                                )
                                if result:
                                    st.success(result.get("message", "Chiarimento salvato."))

                    st.divider()


def render_report_list_and_detail(
    reports: List[dict],
    key_prefix: str,
    show_chat: bool = True,
    review_controls: bool = False,
    allow_clarifications: bool = False,
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

    meta_cols = st.columns([2, 2, 2, 2, 2])
    meta_cols[0].markdown(f"**Stato:** `{detail['status']}`")
    meta_cols[1].markdown(f"**Creato da:** {detail['created_by']}")
    if detail.get("reviewed_by"):
        meta_cols[2].markdown(f"**Revisionato da:** {detail['reviewed_by']}")
    if detail.get("review_comment"):
        meta_cols[3].markdown(f"**Commento:** {detail['review_comment']}")
    if show_chat:
        with meta_cols[4]:
            open_chat_button(report_id, detail["report"], key=f"{key_prefix}_chat_{report_id}")

    if review_controls and detail["status"] == "PENDING_REVIEW":
        st.divider()
        st.markdown("#### :material/rate_review: Revisione del certificatore")
        st.caption(
            "Verifica le schede di valutazione prodotte dagli Agenti Specializzati e la "
            "prioritizzazione delle lacune consolidata dall'AGA. Approvando il report, questo "
            "diventerà visibile ai dipendenti; rifiutandolo resterà accessibile solo a te. "
            "Il commento è riportato insieme al report in entrambi i casi."
        )
        comment = st.text_area("Commento (opzionale)", key=f"{key_prefix}_comment_{report_id}")
        col_a, col_r = st.columns(2)
        with col_a:
            if st.button("Approva report", type="primary", key=f"{key_prefix}_approve_{report_id}", use_container_width=True):
                result = api_post(f"/reports/{report_id}/review", json={"approve": True, "comment": comment})
                if result:
                    st.success(f"Report #{report_id} approvato.")
                    st.rerun()
        with col_r:
            if st.button("Rifiuta report", key=f"{key_prefix}_reject_{report_id}", use_container_width=True):
                result = api_post(f"/reports/{report_id}/review", json={"approve": False, "comment": comment})
                if result:
                    st.warning(f"Report #{report_id} rifiutato.")
                    st.rerun()

    st.divider()
    render_report_dashboard(
        detail["report"],
        report_id=report_id,
        allow_clarifications=allow_clarifications and detail["status"] == "APPROVED",
    )


# ---------------------------------------------------------------------------
# Documents section
# ---------------------------------------------------------------------------

def render_documents_section(can_upload: bool) -> None:
    if can_upload:
        st.markdown("#### :material/upload_file: Carica documenti")
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
        st.markdown("#### :material/folder_shared: Tutti i documenti aziendali")
    else:
        st.markdown("#### :material/folder: I miei documenti")

    docs = api_get("/documents")
    if docs is None:
        return
    if not docs:
        st.info("Nessun documento caricato.")
        return

    for doc in docs:
        cols = st.columns([4, 2, 2, 1], vertical_alignment="center")
        cols[0].markdown(f":material/description: **{doc['filename']}**")
        cols[1].caption(f"Caricato da: {doc['uploader']}")
        cols[2].caption(doc["uploaded_at"][:19])
        if cols[3].button("Elimina", key=f"del_doc_{doc['id']}", help="Elimina documento"):
            if api_delete(f"/documents/{doc['id']}"):
                st.rerun()


# ---------------------------------------------------------------------------
# Employee view
# ---------------------------------------------------------------------------

def render_employee_view() -> None:
    tab_docs, tab_analysis, tab_reports = st.tabs(
        ["Documenti", "Analisi", "Report approvati"]
    )

    with tab_docs:
        render_documents_section(can_upload=True)

    with tab_analysis:
        st.markdown("#### :material/play_circle: Avvia l'analisi di conformità")
        st.markdown(
            "L'analisi valuta **l'intero corpus documentale aziendale** (i documenti di "
            "tutti i dipendenti) rispetto ai requisiti dello standard ISO/IEC 42001:2023, "
            "in tre stadi:"
        )
        st.markdown(
            "1. **Analisi parallela** — gli Agenti Specializzati AS-1, AS-2 e AS-3 valutano "
            "le rispettive porzioni della norma (Clausole 4–10 e controlli Annex A), producendo "
            "una scheda di valutazione per ciascun requisito con verdetto, evidenze documentali, "
            "lacune e proposta correttiva.  \n"
            "2. **Consolidamento** — l'Agente di Gap Analysis (AGA) integra le schede in un "
            "report unico, con prioritizzazione delle lacune e piano d'azione.  \n"
            "3. **Revisione umana** — il report resta in attesa di verifica da parte del "
            "certificatore e diventa visibile solo dopo la sua approvazione."
        )
        st.caption(
            "La durata dipende dal numero di requisiti e dall'hardware di inferenza: "
            "da alcuni minuti a oltre un'ora con inferenza su CPU."
        )

        if st.session_state.analysis_notice:
            st.info(st.session_state.analysis_notice)

        if st.button("Avvia Analisi", type="primary"):
            with st.spinner("Analisi ISO 42001 in corso... Può richiedere diversi minuti."):
                result = api_post("/analyze", timeout=ANALYZE_TIMEOUT)
            if result:
                st.session_state.analysis_notice = result.get(
                    "message", "Analisi completata, in attesa di revisione."
                )
                st.success(
                    f"Analisi completata (report #{result.get('report_id')}). "
                    "Il report è in attesa di verifica da parte del certificatore."
                )

    with tab_reports:
        st.markdown("#### :material/lab_profile: Report approvati dal certificatore")
        st.caption(
            "Per i requisiti NON CONFORME o PARZIALMENTE CONFORME puoi aggiungere "
            "un chiarimento: verrà salvato tra i documenti aziendali e considerato "
            "alla prossima analisi."
        )
        reports = api_get("/reports")
        if reports is not None:
            render_report_list_and_detail(
                reports, key_prefix="emp", show_chat=True, allow_clarifications=True
            )


# ---------------------------------------------------------------------------
# Certifier view
# ---------------------------------------------------------------------------

def render_certifier_view() -> None:
    tab_pending, tab_all, tab_docs = st.tabs(
        ["Da revisionare", "Tutti i report", "Documenti"]
    )

    with tab_pending:
        st.markdown("#### :material/pending_actions: Report in attesa di revisione")
        reports = api_get("/reports")
        if reports is not None:
            pending = [r for r in reports if r["status"] == "PENDING_REVIEW"]
            render_report_list_and_detail(
                pending, key_prefix="cert_pending", show_chat=False, review_controls=True
            )

    with tab_all:
        st.markdown("#### :material/history: Storico report")
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
    st.markdown(_APP_CSS, unsafe_allow_html=True)
    render_top_bar()

    if st.session_state.role == "certifier":
        render_certifier_view()
    else:
        render_employee_view()
