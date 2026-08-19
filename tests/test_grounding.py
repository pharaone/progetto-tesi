"""Unit tests for evidence grounding (shared/grounding.py).

Pins the guarantee that a CONFORME verdict cannot survive without an
evidence that is verifiably present in the organizational documentation
actually retrieved for that requirement.
"""

from __future__ import annotations

from shared.grounding import (
    build_snippets,
    enforce_grounding,
    is_grounded_in,
    is_substantive,
    verify_evidences,
)
from shared.models import (
    CorrectiveAction,
    EvaluationCard,
    Evidence,
    Verdict,
)

ORG_TEXT = (
    "Politica per l'Intelligenza Artificiale di Nexoria Consulting S.p.A. "
    "Il Consiglio di Amministrazione ha approvato in data 12 marzo 2024 la "
    "politica per l'uso responsabile dell'intelligenza artificiale, che "
    "definisce ruoli, responsabilita e criteri di valutazione del rischio."
)

ISO_TEXT = (
    "8.4 AI system impact assessment. The organization shall perform AI system "
    "impact assessments according to 6.1.4 at planned intervals."
)

SNIPPETS = [
    {
        "chunk_id": "org_chunk_17",
        "source_doc": "Nexoria_Consulting_Linee_Guida_IA.pdf",
        "text": ORG_TEXT,
    }
]


def _card(verdict: Verdict, evidences) -> EvaluationCard:
    return EvaluationCard(
        requirement_id="cl-5.2",
        requirement_text="Top management shall establish an AI policy.",
        verdict=verdict,
        evidences=evidences,
        gaps=[],
        corrective_action=CorrectiveAction(
            description="—", expected_document_type="policy_document"
        ),
    )


def _evidence(excerpt: str, source: str = "claimed.pdf") -> Evidence:
    return Evidence(chunk_id="claimed_id", source_doc=source, excerpt=excerpt)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def test_verbatim_excerpt_is_grounded():
    assert is_grounded_in("ha approvato in data 12 marzo 2024 la politica", ORG_TEXT)


def test_cosmetic_differences_are_tolerated():
    assert is_grounded_in(
        "Il Consiglio  di   Amministrazione HA APPROVATO in data 12 marzo 2024", ORG_TEXT
    )


def test_paraphrase_is_not_grounded():
    assert not is_grounded_in(
        "L'azienda dispone di una policy sull'AI approvata dai vertici", ORG_TEXT
    )


def test_too_short_excerpt_is_not_grounded():
    assert not is_grounded_in("politica", ORG_TEXT)


def test_iso_text_is_not_grounded_in_org_documentation():
    assert not is_grounded_in(
        "The organization shall perform AI system impact assessments", ORG_TEXT
    )


# ---------------------------------------------------------------------------
# Provenance is system-assigned, never model-asserted
# ---------------------------------------------------------------------------

def test_verified_evidence_gets_real_provenance():
    cited = _evidence(
        "definisce ruoli, responsabilita e criteri di valutazione del rischio",
        source="documento-inventato.pdf",
    )
    verified, iso_confusions, unverifiable = verify_evidences(
        [cited], SNIPPETS, ISO_TEXT
    )

    assert (iso_confusions, unverifiable) == (0, 0)
    assert verified[0].chunk_id == "org_chunk_17"
    assert verified[0].source_doc == "Nexoria_Consulting_Linee_Guida_IA.pdf"


def test_iso_quote_labelled_as_company_doc_is_flagged():
    """The exact failure observed on cl-8.4: ISO text with the company filename."""
    cited = _evidence(
        "The organization shall perform AI system impact assessments according to 6.1.4",
        source="Nexoria_Consulting_Linee_Guida_IA.pdf",
    )
    verified, iso_confusions, unverifiable = verify_evidences(
        [cited], SNIPPETS, ISO_TEXT
    )

    assert verified == []
    assert iso_confusions == 1
    assert unverifiable == 0


# ---------------------------------------------------------------------------
# Verdict enforcement
# ---------------------------------------------------------------------------

def test_conforme_without_any_evidence_is_downgraded():
    card = enforce_grounding(_card(Verdict.CONFORME, []), SNIPPETS, ISO_TEXT)

    assert card.verdict == Verdict.NON_CONFORME
    assert any("declassato" in gap for gap in card.gaps)


def test_conforme_on_hallucinated_evidence_is_downgraded():
    card = _card(Verdict.CONFORME, [_evidence("La societa ha ottenuto la certificazione ISO 42001 nel 2023")])
    card = enforce_grounding(card, SNIPPETS, ISO_TEXT)

    assert card.verdict == Verdict.NON_CONFORME
    assert card.evidences == []
    assert any("non è stata trovata" in gap for gap in card.gaps)


def test_partially_compliant_without_evidence_is_downgraded_too():
    card = enforce_grounding(_card(Verdict.PARZIALMENTE_CONFORME, []), SNIPPETS, ISO_TEXT)
    assert card.verdict == Verdict.NON_CONFORME


def test_conforme_with_one_real_evidence_survives():
    card = _card(
        Verdict.CONFORME,
        [
            _evidence("ha approvato in data 12 marzo 2024 la politica per l'uso responsabile"),
            _evidence("La societa effettua audit trimestrali certificati"),  # invented
        ],
    )
    card = enforce_grounding(card, SNIPPETS, ISO_TEXT)

    assert card.verdict == Verdict.CONFORME
    assert len(card.evidences) == 1
    assert card.evidences[0].chunk_id == "org_chunk_17"
    assert any("non è stata trovata" in gap for gap in card.gaps)


def test_document_title_is_not_substantive():
    """A real quote that asserts nothing: no predicate, so it proves nothing."""
    assert not is_substantive(
        "Politica per l'Intelligenza Artificiale di Nexoria Consulting S.p.A."
    )
    assert is_substantive(
        "Il Consiglio di Amministrazione ha approvato la politica per l'uso responsabile"
    )


def test_short_quote_is_not_substantive():
    assert not is_substantive("La politica esiste")


def test_conforme_supported_only_by_the_document_title_is_downgraded():
    """The exact failure observed on cl-4.4: the title cited four times."""
    title = "Politica per l'Intelligenza Artificiale di Nexoria Consulting S.p.A."
    card = _card(Verdict.CONFORME, [_evidence(title) for _ in range(4)])
    card = enforce_grounding(card, SNIPPETS, ISO_TEXT)

    assert card.verdict == Verdict.NON_CONFORME
    # The citation is truthful, so it is kept for traceability — deduplicated
    assert len(card.evidences) == 1
    assert any("non sostanziali" in gap for gap in card.gaps)


def test_duplicate_citations_are_collapsed():
    quote = "ha approvato in data 12 marzo 2024 la politica per l'uso responsabile"
    card = _card(Verdict.CONFORME, [_evidence(quote), _evidence(quote), _evidence(quote)])
    card = enforce_grounding(card, SNIPPETS, ISO_TEXT)

    assert card.verdict == Verdict.CONFORME
    assert len(card.evidences) == 1


def test_non_conforme_keeps_its_verdict_but_loses_fake_evidence():
    card = _card(Verdict.NON_CONFORME, [_evidence("Testo totalmente inventato dal modello")])
    card = enforce_grounding(card, SNIPPETS, ISO_TEXT)

    assert card.verdict == Verdict.NON_CONFORME
    assert card.evidences == []


def test_empty_retrieval_downgrades_everything_positive():
    card = _card(Verdict.CONFORME, [_evidence("qualunque cosa il modello abbia scritto qui")])
    card = enforce_grounding(card, [], "")
    assert card.verdict == Verdict.NON_CONFORME


# ---------------------------------------------------------------------------
# Snippet construction mirrors what the prompt showed
# ---------------------------------------------------------------------------

def test_build_snippets_keeps_ids_and_truncation():
    results = {
        "documents": [["x" * 1000]],
        "metadatas": [[{"source": "policy.pdf"}]],
        "ids": [["chunk_1"]],
    }
    snippets = build_snippets(results, max_chars=800)

    assert snippets[0]["chunk_id"] == "chunk_1"
    assert snippets[0]["source_doc"] == "policy.pdf"
    assert len(snippets[0]["text"]) == 800


def test_build_snippets_handles_empty_results():
    assert build_snippets({"ids": [[]], "documents": [[]], "metadatas": [[]]}) == []
