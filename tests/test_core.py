"""Core KPI tests for ISO/IEC 42001 Gap Analysis System.

Metrics tested:
- TCN (Total Coverage Number): All 45 requirements produce evaluation cards
- TA (Test Accuracy): Output schema validation for all cards
- IR (Idempotency/Reproducibility): Same input → same verdicts across 3 runs
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List

import httpx
import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AS1_URL = os.getenv("AS1_URL", "http://localhost:8001")
AS2_URL = os.getenv("AS2_URL", "http://localhost:8002")
AS3_URL = os.getenv("AS3_URL", "http://localhost:8003")
AGA_URL = os.getenv("AGA_URL", "http://localhost:8004")
AIU_URL = os.getenv("AIU_URL", "http://localhost:8005")
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")

TEST_ORG_ID = "test-org-kpi-001"

# Known requirement IDs by agent
AS1_REQUIREMENTS = [
    "cl-4.1", "cl-4.2", "cl-4.3", "cl-4.4",
    "cl-5.1", "cl-5.2", "cl-5.3",
    "cl-6.1", "cl-6.2", "cl-6.3",
]

AS2_REQUIREMENTS = [
    "cl-7.1", "cl-7.2", "cl-7.3", "cl-7.4", "cl-7.5",
    "cl-8.1", "cl-8.2", "cl-8.3", "cl-8.4",
    "A.2.1", "A.2.2", "A.3.1", "A.4.1", "A.4.2",
    "A.5.1", "A.5.2", "A.5.3", "A.5.4", "A.5.5", "A.5.6", "A.5.7",
    "A.6.1.1", "A.6.1.2", "A.6.1.3", "A.6.1.4",
    "A.6.2.1", "A.6.2.2", "A.6.2.3", "A.6.2.4",
    "A.6.2.5", "A.6.2.6", "A.6.2.7", "A.6.2.8",
]

AS3_REQUIREMENTS = [
    "cl-9.1", "cl-9.2", "cl-9.3",
    "cl-10.1", "cl-10.2",
    "A.7.1", "A.7.2", "A.7.3", "A.7.4", "A.7.5",
    "A.8.1", "A.8.2", "A.8.3", "A.8.4",
    "A.9.1", "A.9.2",
    "A.10.1", "A.10.2", "A.10.3",
]

ALL_REQUIREMENTS = AS1_REQUIREMENTS + AS2_REQUIREMENTS + AS3_REQUIREMENTS
TOTAL_REQUIREMENTS = len(ALL_REQUIREMENTS)  # 10 + 33 + 19 = 62

VALID_VERDICTS = {
    "CONFORME",
    "NON_CONFORME",
    "PARZIALMENTE_CONFORME",
    "NON_APPLICABILE",
}

REQUIRED_CARD_FIELDS = {
    "requirement_id",
    "requirement_text",
    "verdict",
    "evidences",
    "gaps",
    "corrective_action",
    "execution_metadata",
}

REQUIRED_CORRECTIVE_ACTION_FIELDS = {"description", "expected_document_type"}
REQUIRED_EXECUTION_METADATA_FIELDS = {"timestamp", "model_version", "input_hash"}


# ---------------------------------------------------------------------------
# Sample test documents
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENT = {
    "filename": "ai_policy.txt",
    "content": (
        "AI Policy — ACME Corporation\n\n"
        "1. Context and Scope\n"
        "This policy applies to all AI systems developed, deployed, or operated by ACME Corporation. "
        "We operate in the healthcare and financial services sectors and are subject to GDPR and ISO 27001.\n\n"
        "2. Leadership Commitment\n"
        "Top management is committed to responsible AI development and has established an AI Governance Board "
        "responsible for overseeing AI management system compliance.\n\n"
        "3. AI Objectives\n"
        "- Maintain AI systems that are transparent, fair, and explainable\n"
        "- Conduct annual AI risk assessments\n"
        "- Implement human oversight for all high-risk AI applications\n\n"
        "4. Risk Management\n"
        "We perform impact assessments for all AI systems before deployment. "
        "Risk registers are maintained and reviewed quarterly.\n\n"
        "5. Data Governance\n"
        "All training data is documented with provenance records. "
        "Data quality checks are performed before model training.\n\n"
        "6. Monitoring and Improvement\n"
        "AI systems are monitored continuously. Internal audits are conducted annually. "
        "Performance metrics are reported to management review.\n"
    ),
    "metadata": {"org_id": TEST_ORG_ID},
}

ANALYZE_PAYLOAD = {
    "org_id": TEST_ORG_ID,
    "documents": [SAMPLE_DOCUMENT],
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _post_analyze(url: str, payload: dict, timeout: float = 300.0) -> List[Dict[str, Any]]:
    """POST to an AS agent /analyze endpoint."""
    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{url}/analyze", json=payload)
        response.raise_for_status()
        return response.json()


def _validate_evaluation_card(card: Dict[str, Any]) -> List[str]:
    """Validate an evaluation card, return list of validation errors."""
    errors = []

    # Required fields
    for field in REQUIRED_CARD_FIELDS:
        if field not in card:
            errors.append(f"Missing field: {field}")

    # Verdict validity
    verdict = card.get("verdict", "")
    if verdict not in VALID_VERDICTS:
        errors.append(f"Invalid verdict: '{verdict}' not in {VALID_VERDICTS}")

    # evidences must be a list
    if not isinstance(card.get("evidences"), list):
        errors.append("'evidences' must be a list")

    # gaps must be a list
    if not isinstance(card.get("gaps"), list):
        errors.append("'gaps' must be a list")

    # corrective_action must be a dict with required fields
    ca = card.get("corrective_action", {})
    if not isinstance(ca, dict):
        errors.append("'corrective_action' must be a dict")
    else:
        for f in REQUIRED_CORRECTIVE_ACTION_FIELDS:
            if f not in ca:
                errors.append(f"corrective_action missing field: {f}")

    # execution_metadata must be a dict with required fields
    meta = card.get("execution_metadata", {})
    if not isinstance(meta, dict):
        errors.append("'execution_metadata' must be a dict")
    else:
        for f in REQUIRED_EXECUTION_METADATA_FIELDS:
            if f not in meta:
                errors.append(f"execution_metadata missing field: {f}")

    # requirement_id and requirement_text must be non-empty strings
    for str_field in ("requirement_id", "requirement_text"):
        val = card.get(str_field, "")
        if not isinstance(val, str) or not val.strip():
            errors.append(f"'{str_field}' must be a non-empty string")

    return errors


# ---------------------------------------------------------------------------
# TCN Tests: Total Coverage Number
# ---------------------------------------------------------------------------

class TestTCN:
    """TCN: Verify system produces evaluation cards for all expected requirements."""

    @pytest.mark.integration
    def test_as1_covers_all_clause_456_requirements(self):
        """AS-1 must return exactly 10 evaluation cards for cl-4.x, cl-5.x, cl-6.x."""
        cards = _post_analyze(AS1_URL, ANALYZE_PAYLOAD)

        returned_ids = {card["requirement_id"] for card in cards}
        missing = set(AS1_REQUIREMENTS) - returned_ids

        assert len(cards) == len(AS1_REQUIREMENTS), (
            f"Expected {len(AS1_REQUIREMENTS)} cards, got {len(cards)}"
        )
        assert not missing, f"Missing requirement IDs in AS-1 output: {missing}"

    @pytest.mark.integration
    def test_as2_covers_all_clause_78_and_annex_a26_requirements(self):
        """AS-2 must return evaluation cards for all cl-7.x, cl-8.x, A.2-A.6 requirements."""
        cards = _post_analyze(AS2_URL, ANALYZE_PAYLOAD)

        returned_ids = {card["requirement_id"] for card in cards}
        missing = set(AS2_REQUIREMENTS) - returned_ids

        assert len(cards) == len(AS2_REQUIREMENTS), (
            f"Expected {len(AS2_REQUIREMENTS)} cards, got {len(cards)}"
        )
        assert not missing, f"Missing requirement IDs in AS-2 output: {missing}"

    @pytest.mark.integration
    def test_as3_covers_all_clause_910_and_annex_a710_requirements(self):
        """AS-3 must return evaluation cards for all cl-9.x, cl-10.x, A.7-A.10 requirements."""
        cards = _post_analyze(AS3_URL, ANALYZE_PAYLOAD)

        returned_ids = {card["requirement_id"] for card in cards}
        missing = set(AS3_REQUIREMENTS) - returned_ids

        assert len(cards) == len(AS3_REQUIREMENTS), (
            f"Expected {len(AS3_REQUIREMENTS)} cards, got {len(cards)}"
        )
        assert not missing, f"Missing requirement IDs in AS-3 output: {missing}"

    @pytest.mark.integration
    def test_full_pipeline_covers_all_requirements(self):
        """Full pipeline (via AGA consolidate) must cover all requirements."""
        as1_cards = _post_analyze(AS1_URL, ANALYZE_PAYLOAD)
        as2_cards = _post_analyze(AS2_URL, ANALYZE_PAYLOAD)
        as3_cards = _post_analyze(AS3_URL, ANALYZE_PAYLOAD)

        consolidate_payload = {
            "org_id": TEST_ORG_ID,
            "as1_output": as1_cards,
            "as2_output": as2_cards,
            "as3_output": as3_cards,
            "failed_agents": [],
        }

        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{AGA_URL}/consolidate", json=consolidate_payload)
            response.raise_for_status()
            report = response.json()

        all_cards = report.get("evaluation_cards", [])
        returned_ids = {card["requirement_id"] for card in all_cards}
        missing = set(ALL_REQUIREMENTS) - returned_ids

        assert len(all_cards) >= TOTAL_REQUIREMENTS, (
            f"Expected at least {TOTAL_REQUIREMENTS} cards, got {len(all_cards)}"
        )
        assert not missing, f"Missing requirement IDs in full pipeline output: {missing}"
        assert report.get("total_requirements") == len(all_cards)

    @pytest.mark.integration
    def test_tcn_no_duplicate_requirement_ids(self):
        """No requirement ID should appear twice in the combined output."""
        as1_cards = _post_analyze(AS1_URL, ANALYZE_PAYLOAD)
        as2_cards = _post_analyze(AS2_URL, ANALYZE_PAYLOAD)
        as3_cards = _post_analyze(AS3_URL, ANALYZE_PAYLOAD)

        all_ids = (
            [c["requirement_id"] for c in as1_cards]
            + [c["requirement_id"] for c in as2_cards]
            + [c["requirement_id"] for c in as3_cards]
        )
        duplicates = [r for r in set(all_ids) if all_ids.count(r) > 1]
        assert not duplicates, f"Duplicate requirement IDs found: {duplicates}"


# ---------------------------------------------------------------------------
# TA Tests: Test Accuracy (schema validation)
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """TA: Verify all evaluation cards conform to the required schema."""

    @pytest.mark.integration
    def test_as1_cards_schema_valid(self):
        """All AS-1 evaluation cards must have the required schema."""
        cards = _post_analyze(AS1_URL, ANALYZE_PAYLOAD)
        assert cards, "AS-1 returned empty list"

        for card in cards:
            errors = _validate_evaluation_card(card)
            assert not errors, (
                f"Schema validation failed for {card.get('requirement_id', 'unknown')}: {errors}"
            )

    @pytest.mark.integration
    def test_as2_cards_schema_valid(self):
        """All AS-2 evaluation cards must have the required schema."""
        cards = _post_analyze(AS2_URL, ANALYZE_PAYLOAD)
        assert cards, "AS-2 returned empty list"

        for card in cards:
            errors = _validate_evaluation_card(card)
            assert not errors, (
                f"Schema validation failed for {card.get('requirement_id', 'unknown')}: {errors}"
            )

    @pytest.mark.integration
    def test_as3_cards_schema_valid(self):
        """All AS-3 evaluation cards must have the required schema."""
        cards = _post_analyze(AS3_URL, ANALYZE_PAYLOAD)
        assert cards, "AS-3 returned empty list"

        for card in cards:
            errors = _validate_evaluation_card(card)
            assert not errors, (
                f"Schema validation failed for {card.get('requirement_id', 'unknown')}: {errors}"
            )

    @pytest.mark.integration
    def test_gap_report_schema_valid(self):
        """GapReport from AGA must have all required fields with correct types."""
        as1_cards = _post_analyze(AS1_URL, ANALYZE_PAYLOAD)
        as2_cards = _post_analyze(AS2_URL, ANALYZE_PAYLOAD)
        as3_cards = _post_analyze(AS3_URL, ANALYZE_PAYLOAD)

        payload = {
            "org_id": TEST_ORG_ID,
            "as1_output": as1_cards,
            "as2_output": as2_cards,
            "as3_output": as3_cards,
            "failed_agents": [],
        }

        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{AGA_URL}/consolidate", json=payload)
            response.raise_for_status()
            report = response.json()

        # Required GapReport fields
        assert "org_id" in report
        assert "overall_compliance_score" in report
        assert 0 <= report["overall_compliance_score"] <= 100
        assert "total_requirements" in report
        assert "counts" in report
        assert "prioritized_gaps" in report
        assert "action_plan" in report
        assert "evaluation_cards" in report
        assert "execution_metadata" in report
        assert isinstance(report["prioritized_gaps"], list)
        assert isinstance(report["action_plan"], list)
        assert isinstance(report["evaluation_cards"], list)

        # Counts
        counts = report["counts"]
        assert "compliant" in counts
        assert "non_compliant" in counts
        assert "partial" in counts
        assert "not_applicable" in counts

        # Total should match sum of counts
        total_from_counts = (
            counts["compliant"]
            + counts["non_compliant"]
            + counts["partial"]
            + counts["not_applicable"]
        )
        assert total_from_counts == report["total_requirements"]

    @pytest.mark.integration
    def test_prioritized_gaps_ordering(self):
        """Prioritized gaps must be ordered: NON_CONFORME (severity=1) before PARZIALMENTE_CONFORME (severity=2)."""
        as1_cards = _post_analyze(AS1_URL, ANALYZE_PAYLOAD)
        as2_cards = _post_analyze(AS2_URL, ANALYZE_PAYLOAD)
        as3_cards = _post_analyze(AS3_URL, ANALYZE_PAYLOAD)

        payload = {
            "org_id": TEST_ORG_ID,
            "as1_output": as1_cards,
            "as2_output": as2_cards,
            "as3_output": as3_cards,
            "failed_agents": [],
        }

        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{AGA_URL}/consolidate", json=payload)
            response.raise_for_status()
            report = response.json()

        gaps = report.get("prioritized_gaps", [])
        severities = [g["severity"] for g in gaps]

        # Severity must be non-decreasing (1s before 2s)
        for i in range(1, len(severities)):
            assert severities[i] >= severities[i - 1], (
                f"Severity ordering violated at position {i}: "
                f"{severities[i - 1]} followed by {severities[i]}"
            )

    @pytest.mark.integration
    def test_input_hash_consistency(self):
        """input_hash in execution_metadata must match SHA-256 of document contents."""
        content = SAMPLE_DOCUMENT["content"]
        expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        cards = _post_analyze(AS1_URL, ANALYZE_PAYLOAD)
        assert cards

        # All cards should have the same input_hash
        for card in cards:
            meta = card.get("execution_metadata", {})
            actual_hash = meta.get("input_hash", "")
            assert actual_hash == expected_hash, (
                f"Input hash mismatch for {card['requirement_id']}: "
                f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
            )


# ---------------------------------------------------------------------------
# IR Tests: Idempotency/Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """IR: Same input 3 times → same verdicts (temperature=0, deterministic)."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_as1_reproducibility(self):
        """Running AS-1 three times on the same input must yield identical verdicts."""
        runs = []
        for i in range(3):
            cards = _post_analyze(AS1_URL, ANALYZE_PAYLOAD)
            run_verdicts = {
                card["requirement_id"]: card["verdict"]
                for card in cards
            }
            runs.append(run_verdicts)

        # All three runs must produce identical verdicts
        assert runs[0] == runs[1], (
            f"AS-1 run 1 vs run 2 differ:\n"
            + "\n".join(
                f"  {k}: {runs[0].get(k)} vs {runs[1].get(k)}"
                for k in runs[0]
                if runs[0].get(k) != runs[1].get(k)
            )
        )
        assert runs[0] == runs[2], (
            f"AS-1 run 1 vs run 3 differ:\n"
            + "\n".join(
                f"  {k}: {runs[0].get(k)} vs {runs[2].get(k)}"
                for k in runs[0]
                if runs[0].get(k) != runs[2].get(k)
            )
        )

    @pytest.mark.integration
    @pytest.mark.slow
    def test_as2_reproducibility(self):
        """Running AS-2 three times on the same input must yield identical verdicts."""
        runs = []
        for _ in range(3):
            cards = _post_analyze(AS2_URL, ANALYZE_PAYLOAD)
            runs.append({c["requirement_id"]: c["verdict"] for c in cards})

        assert runs[0] == runs[1], "AS-2 run 1 vs run 2 differ"
        assert runs[0] == runs[2], "AS-2 run 1 vs run 3 differ"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_as3_reproducibility(self):
        """Running AS-3 three times on the same input must yield identical verdicts."""
        runs = []
        for _ in range(3):
            cards = _post_analyze(AS3_URL, ANALYZE_PAYLOAD)
            runs.append({c["requirement_id"]: c["verdict"] for c in cards})

        assert runs[0] == runs[1], "AS-3 run 1 vs run 2 differ"
        assert runs[0] == runs[2], "AS-3 run 1 vs run 3 differ"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_compliance_score_reproducibility(self):
        """Running full pipeline 3 times must yield same compliance score."""
        scores = []
        for _ in range(3):
            as1 = _post_analyze(AS1_URL, ANALYZE_PAYLOAD)
            as2 = _post_analyze(AS2_URL, ANALYZE_PAYLOAD)
            as3 = _post_analyze(AS3_URL, ANALYZE_PAYLOAD)

            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    f"{AGA_URL}/consolidate",
                    json={
                        "org_id": TEST_ORG_ID,
                        "as1_output": as1,
                        "as2_output": as2,
                        "as3_output": as3,
                        "failed_agents": [],
                    },
                )
                resp.raise_for_status()
                report = resp.json()
                scores.append(report["overall_compliance_score"])

        assert scores[0] == scores[1] == scores[2], (
            f"Compliance scores not reproducible: {scores}"
        )
