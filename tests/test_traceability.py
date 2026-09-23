"""Unit tests for report traceability fields.

- execution_metadata must come from the server, never from the LLM output:
  mistral:7b fills the schema's metadata block with placeholder hashes and
  past dates, which made every card claim a fake input hash.
- the executive summary must quote the score of the current report, not a
  score the model picked up from earlier reports in ORG-HISTORY.

Pure unit tests: no services, no LLM calls.
"""

from __future__ import annotations

import importlib
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("chromadb")
pytest.importorskip("langchain_core")

REQUIREMENT = {"id": "A.2.2", "text": "A.2.2\n\nAI policy\n\nThe organization shall document a policy."}

MODEL_OUTPUT = json.dumps({
    "requirement_id": "A.2.2",
    "verdict": "PARZIALMENTE_CONFORME",
    "evidences": [],
    "gaps": ["gap"],
    "corrective_action": {"description": "fix", "expected_document_type": "policy"},
    "execution_metadata": {
        "timestamp": "2023-03-15T12:00:00Z",
        "model_version": "v1.0",
        "input_hash": "sha256:abcdefghijklmnopqrstuvwxyz123456",
    },
})


@pytest.mark.parametrize("service", ["as1", "as2", "as3"])
def test_execution_metadata_ignores_values_invented_by_the_model(service):
    module = importlib.import_module(f"services.{service}.main")
    card = module._parse_llm_output(
        MODEL_OUTPUT, REQUIREMENT, [], input_hash="a" * 64, model_version="mistral:7b"
    )

    meta = card.execution_metadata
    assert meta.input_hash == "a" * 64
    assert meta.model_version == "mistral:7b"
    assert not meta.timestamp.startswith("2023")


def test_summary_quotes_the_current_score_not_a_previous_one():
    from services.aga.main import _pin_score

    summary = (
        "The nexoria-test organization is significantly non-compliant, "
        "with a score of 33.6/100. Critical areas: risk management."
    )
    assert "39.9/100" in _pin_score(summary, 39.86)
    assert "33.6" not in _pin_score(summary, 39.86)


def test_summary_without_a_score_is_left_untouched():
    from services.aga.main import _pin_score

    summary = "Critical areas include clause 6.1.2 and control A.5.2."
    assert _pin_score(summary, 39.86) == summary
