"""Evidence grounding: verify that an agent's citations really exist.

The LLM returns evidences as free-form JSON, so nothing stops it from
inventing an excerpt, or — observed in practice — from quoting the ISO
requirement text and labelling it with the company document's filename.
Either way the verdict looks supported while it is not.

This module is the control that closes that gap, independently of prompt
quality:

1. every excerpt cited by the model is matched against the exact snippets
   that were actually retrieved and shown to it;
2. matched evidences get their chunk_id/source_doc **rewritten from the
   retrieved chunk**, so provenance is assigned by the system and never
   asserted by the model (traceability requirement of the thesis);
3. excerpts that match the ISO standard context instead of the
   organizational documentation are dropped and flagged as source
   confusion;
4. duplicate citations are collapsed and each one is checked for
   substance: a document title, heading or filename is a truthful quote
   yet proves nothing about what the organization does, so it cannot
   support a positive verdict;
5. a CONFORME / PARZIALMENTE_CONFORME verdict left without a single
   verifiable AND substantive evidence is downgraded to NON_CONFORME with
   an explicit AUTO-FLAG gap — controlled degradation rather than a false
   positive.

Matching is verbatim-tolerant, not semantic: exact containment after
whitespace/case normalisation, with an n-gram containment fallback that
survives cosmetic edits (punctuation, ellipsis) while still rejecting
paraphrase. Verified-but-thin citations are kept in the card for
traceability; they simply do not count as support.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from shared.models import EvaluationCard, Evidence, Verdict

logger = logging.getLogger(__name__)

# An excerpt shorter than this cannot be meaningfully verified
MIN_EXCERPT_CHARS = 15
# Word n-gram size and minimum share of n-grams that must appear in the source
NGRAM_SIZE = 4
MIN_CONTAINMENT = 0.6
# A citation shorter than this cannot substantiate an obligation
MIN_EVIDENCE_WORDS = 8

# A statement about what the organization does contains a predicate; a title,
# heading or filename does not. Deliberately small and transparent: over-
# flagging only sends the card to the certifier, it never invents compliance.
_PREDICATE_WORDS = frozenset(
    {
        # Italian
        "è", "sono", "era", "erano", "ha", "hanno", "deve", "devono",
        "viene", "vengono",
        "definisce", "definiscono", "definito", "definita", "definiti", "definite",
        "stabilisce", "stabiliscono", "stabilito", "stabilita",
        "prevede", "prevedono", "previsto", "prevista",
        "adotta", "adottano", "adottato", "adottata",
        "effettua", "effettuano", "effettuato", "effettuata",
        "esegue", "eseguono", "eseguito", "eseguita",
        "garantisce", "garantiscono", "garantito", "garantita",
        "mantiene", "mantengono", "mantenuto", "mantenuta",
        "approva", "approvano", "approvato", "approvata",
        "documenta", "documentano", "documentato", "documentata",
        "assicura", "assicurano", "identifica", "identificano",
        "monitora", "monitorano", "riesamina", "riesaminano",
        "copre", "coprono", "include", "includono", "comprende", "comprendono",
        "utilizza", "utilizzano", "applica", "applicano", "svolge", "svolgono",
        "nominato", "nominata", "assegna", "assegnati", "assegnate",
        # English
        "is", "are", "was", "were", "has", "have", "had", "shall", "must",
        "defines", "define", "defined", "establishes", "establish", "established",
        "maintains", "maintain", "maintained", "documents", "documented",
        "performs", "perform", "performed", "ensures", "ensure", "ensured",
        "reviews", "review", "reviewed", "approves", "approved",
        "implements", "implement", "implemented", "monitors", "monitored",
        "includes", "include", "covers", "cover", "assigns", "assigned",
        "conducts", "conducted", "identifies", "identified",
    }
)

# Phrases marking a gap or a planned action rather than an implemented one.
# Deliberately narrow: prohibitions ("non è consentito") are real controls and
# must NOT match here.
_ABSENCE_MARKERS = (
    # Italian — absence
    "non è ancora", "non è stato ancora", "non è stata ancora", "non sono ancora",
    "non ancora", "non è presente", "non sono presenti", "non è stata definita",
    "non è stato definito", "non esiste", "non sono in essere", "non dispone",
    "non è formalizzat", "senza un mandato", "in via informale", "su base informale",
    "non formalmente",
    # Italian — plans
    "prossimi passi", "azioni prioritarie", "roadmap", "in fase di", "si prevede",
    "è prevista", "sono previste", "intende adottare", "percorso di adeguamento",
    # English
    "not yet", "will be", "is planned", "are planned", "next steps",
    "to be established", "in progress", "we intend",
)


def normalize(text: str) -> str:
    """Collapse whitespace and case so cosmetic differences do not matter."""
    return " ".join((text or "").split()).lower()


def is_grounded_in(excerpt: str, source_text: str) -> bool:
    """True when `excerpt` is verbatim (or near-verbatim) present in `source_text`."""
    needle = normalize(excerpt)
    haystack = normalize(source_text)

    if len(needle) < MIN_EXCERPT_CHARS or not haystack:
        return False

    # Fast path: exact containment after normalisation
    if needle in haystack:
        return True

    # Tolerant path: how many word n-grams of the excerpt occur in the source?
    words = needle.split()
    if len(words) < NGRAM_SIZE:
        return False

    grams = {
        " ".join(words[i : i + NGRAM_SIZE])
        for i in range(len(words) - NGRAM_SIZE + 1)
    }
    if not grams:
        return False

    hits = sum(1 for gram in grams if gram in haystack)
    return (hits / len(grams)) >= MIN_CONTAINMENT


def states_absence_or_plan(excerpt: str) -> bool:
    """True when a citation describes a gap or a future action, not a practice.

    Observed in real reports: "Non è stato ancora istituito un Comitato di
    Governance IA" and roadmap items under "Prossimi passi" were cited as
    evidence *supporting* compliance. Such a quote is real and substantive,
    yet it proves the opposite of what the verdict claims.
    """
    text = normalize(excerpt)
    return any(marker in text for marker in _ABSENCE_MARKERS)


def is_substantive(excerpt: str) -> bool:
    """True when a citation actually asserts something about the organization.

    Document titles, section headings and filenames are truthful quotes that
    prove nothing: they carry no predicate. Requiring a minimum length and a
    predicate word keeps "Company X — AI Guidelines" from justifying CONFORME.
    """
    words = normalize(excerpt).replace("'", " ").replace("’", " ").split()
    if len(words) < MIN_EVIDENCE_WORDS:
        return False
    return any(word.strip(".,;:()[]«»\"") in _PREDICATE_WORDS for word in words)


def find_source_snippet(
    excerpt: str,
    snippets: Sequence[Dict[str, str]],
) -> Optional[Dict[str, str]]:
    """Return the retrieved snippet an excerpt was actually taken from."""
    for snippet in snippets:
        if is_grounded_in(excerpt, snippet.get("text", "")):
            return snippet
    return None


def verify_evidences(
    evidences: Sequence[Evidence],
    org_snippets: Sequence[Dict[str, str]],
    iso_context: str = "",
) -> Tuple[List[Evidence], int, int]:
    """Split cited evidences into verified / from-ISO / unverifiable.

    Returns (verified_evidences, iso_confusions, unverifiable).
    Verified evidences carry the chunk_id and source_doc of the retrieved
    chunk they were found in, not the ones claimed by the model.
    """
    verified: List[Evidence] = []
    iso_confusions = 0
    unverifiable = 0
    seen: set = set()

    for evidence in evidences:
        snippet = find_source_snippet(evidence.excerpt, org_snippets)
        if snippet is not None:
            # The same quote repeated is still a single piece of evidence
            key = normalize(evidence.excerpt)
            if key in seen:
                continue
            seen.add(key)
            verified.append(
                Evidence(
                    chunk_id=snippet.get("chunk_id", "unknown"),
                    source_doc=snippet.get("source_doc", "unknown"),
                    excerpt=evidence.excerpt[:400],
                )
            )
        elif iso_context and is_grounded_in(evidence.excerpt, iso_context):
            iso_confusions += 1
        else:
            unverifiable += 1

    return verified, iso_confusions, unverifiable


def enforce_grounding(
    card: EvaluationCard,
    org_snippets: Sequence[Dict[str, str]],
    iso_context: str = "",
) -> EvaluationCard:
    """Keep only verifiable evidences and downgrade unsupported verdicts."""
    verified, iso_confusions, unverifiable = verify_evidences(
        card.evidences, org_snippets, iso_context
    )

    gaps = list(card.gaps)

    if iso_confusions:
        gaps.append(
            f"AUTO-FLAG: {iso_confusions} evidenza/e citava il testo della norma "
            "ISO anziché la documentazione aziendale ed è stata scartata."
        )
    if unverifiable:
        gaps.append(
            f"AUTO-FLAG: {unverifiable} evidenza/e non è stata trovata nella "
            "documentazione effettivamente recuperata ed è stata scartata."
        )

    # A citation supports compliance only when it is substantive AND describes
    # something already in place
    supporting = [
        ev for ev in verified
        if is_substantive(ev.excerpt) and not states_absence_or_plan(ev.excerpt)
    ]
    gap_citations = [ev for ev in verified if states_absence_or_plan(ev.excerpt)]

    if card.verdict in (Verdict.CONFORME, Verdict.PARZIALMENTE_CONFORME) and not supporting:
        if gap_citations:
            reason = (
                "le evidenze citate descrivono azioni pianificate o l'assenza del "
                "requisito, non una pratica già in essere"
            )
            log_reason = "only forward-looking or absence citations"
        elif verified:
            reason = (
                "le evidenze citate sono verificabili ma non sostanziali "
                "(titoli, intestazioni o frammenti privi di contenuto)"
            )
            log_reason = "only non-substantive citations"
        else:
            reason = "nessuna evidenza verificabile nella documentazione aziendale"
            log_reason = "no verifiable evidence"
        gaps.append(
            f"AUTO-FLAG: verdetto {card.verdict.value} declassato a NON_CONFORME — {reason}."
        )
        logger.warning(
            f"Grounding check: {card.requirement_id} downgraded from "
            f"{card.verdict.value} to NON_CONFORME ({log_reason})"
        )
        card.verdict = Verdict.NON_CONFORME

    card.evidences = verified
    card.gaps = gaps
    return card


def build_snippets(results: Dict, max_chars: int = 800) -> List[Dict[str, str]]:
    """Rebuild what the model was shown, keeping the real chunk provenance.

    `max_chars` must match the truncation used when formatting the RAG
    results into the prompt, otherwise an excerpt quoted from the tail of a
    long chunk would look unverifiable.
    """
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    ids = (results.get("ids") or [[]])[0]

    snippets: List[Dict[str, str]] = []
    for doc, meta, doc_id in zip(docs, metas, ids):
        source = (meta or {}).get("source", "unknown")
        snippets.append(
            {
                "chunk_id": str(doc_id),
                "source_doc": str(source),
                "text": (doc or "")[:max_chars],
            }
        )
    return snippets
