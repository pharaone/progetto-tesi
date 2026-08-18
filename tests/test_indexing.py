"""Unit tests for requirement-aware ISO indexing (rag/indexer.py).

These tests pin down the property the whole gap analysis depends on:
**one requirement = one segment**. Before requirement-first segmentation the
indexer tagged each 512-character chunk with the first heading it contained,
so short consecutive headings (clause titles, Annex A control table rows)
were swallowed by the preceding requirement and never produced an
evaluation card.

Pure unit tests: no services, no ChromaDB writes.
"""

from __future__ import annotations

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("langchain")

from rag.indexer import (  # noqa: E402
    CHUNK_SIZE,
    _strip_toc_lines,
    segment_by_requirement,
)


def _as_dict(text: str) -> dict:
    """Segment and merge the parts of each requirement, as the loader does."""
    merged: dict = {}
    for requirement_id, segment in segment_by_requirement(text):
        merged[requirement_id] = merged.get(requirement_id, "") + segment
    return merged


# ---------------------------------------------------------------------------
# Regression: a clause title must not merge the next clause into the previous
# ---------------------------------------------------------------------------

CLAUSE_BOUNDARY_SAMPLE = """\
4.4 AI management system

The organization shall establish, implement, maintain and continually improve
an AI management system.

5 Leadership

5.1 Leadership and commitment

Top management shall demonstrate leadership and commitment with respect to the
AI management system.

5.2 AI policy

Top management shall establish an AI policy.
"""


def test_clause_title_does_not_swallow_the_next_requirement():
    segments = _as_dict(CLAUSE_BOUNDARY_SAMPLE)

    assert "cl-4.4" in segments
    assert "cl-5.1" in segments
    assert "cl-5.2" in segments

    # The regression: 5.1 used to live inside the cl-4.4 chunk
    assert "5.1 Leadership and commitment" not in segments["cl-4.4"]
    assert "Top management" not in segments["cl-4.4"]

    assert "Top management shall demonstrate leadership" in segments["cl-5.1"]
    assert "Top management shall establish an AI policy" in segments["cl-5.2"]


# ---------------------------------------------------------------------------
# Regression: consecutive short sub-clauses stay separate
# ---------------------------------------------------------------------------

SHORT_SUBCLAUSES_SAMPLE = """\
8.2 AI risk assessment
The organization shall perform AI risk assessments at planned intervals.
8.3 AI risk treatment
The organization shall implement an AI risk treatment plan.
8.4 AI system impact assessment
The organization shall perform AI system impact assessments.
"""


def test_consecutive_subclauses_are_separate_requirements():
    segments = _as_dict(SHORT_SUBCLAUSES_SAMPLE)

    assert set(segments) >= {"cl-8.2", "cl-8.3", "cl-8.4"}
    assert "8.3 AI risk treatment" not in segments["cl-8.2"]
    assert "risk treatment plan" not in segments["cl-8.2"]
    assert "impact assessment" not in segments["cl-8.3"]


# ---------------------------------------------------------------------------
# Regression: Annex A control table rows (one or two lines each)
# ---------------------------------------------------------------------------

ANNEX_TABLE_SAMPLE = """\
A.4.3 Data resources
The organization shall document information about the data resources used.
A.4.4 Tooling resources
The organization shall document information about the tooling resources used.
A.4.5 System and computing resources
The organization shall document information about the computing resources used.
A.6.2.3 Verification and validation
The organization shall define verification and validation measures.
"""


def test_annex_controls_each_get_their_own_requirement():
    segments = _as_dict(ANNEX_TABLE_SAMPLE)

    for control in ("A.4.3", "A.4.4", "A.4.5", "A.6.2.3"):
        assert control in segments, f"{control} was not indexed as a requirement"

    assert "Tooling resources" not in segments["A.4.3"]
    assert "computing resources" not in segments["A.4.4"]
    # Three-level controls must not be truncated to their parent
    assert "A.6.2" not in segments


# ---------------------------------------------------------------------------
# Generic property: no segment contains another requirement's heading
# ---------------------------------------------------------------------------

def test_no_segment_contains_a_foreign_heading():
    text = CLAUSE_BOUNDARY_SAMPLE + "\n" + SHORT_SUBCLAUSES_SAMPLE + "\n" + ANNEX_TABLE_SAMPLE
    segments = segment_by_requirement(text)
    headings = [rid for rid, _ in segments if rid]

    for requirement_id, segment in segments:
        body = segment.split("\n", 1)[1] if "\n" in segment else ""
        for other in headings:
            if other == requirement_id:
                continue
            plain = other[3:] if other.startswith("cl-") else other
            for line in body.splitlines():
                assert not line.strip().startswith(plain + " "), (
                    f"segment {requirement_id!r} contains heading {other!r}"
                )


# ---------------------------------------------------------------------------
# Long requirements are split but keep a single requirement_id
# ---------------------------------------------------------------------------

def test_long_requirement_keeps_one_id_across_parts():
    body = "The organization shall document the following. " * 40
    text = f"6.1.2 AI risk assessment\n{body}\n7.1 Resources\nShort clause.\n"

    segments = segment_by_requirement(text)
    ids = [rid for rid, _ in segments]

    assert ids.count("cl-6.1.2") == 1  # segmentation yields one segment...
    assert len(dict(segments)["cl-6.1.2"]) > CHUNK_SIZE  # ...longer than a chunk
    assert "cl-7.1" in ids


# ---------------------------------------------------------------------------
# Noise that must NOT create requirements
# ---------------------------------------------------------------------------

def test_page_numbers_and_toc_do_not_create_requirements():
    text = (
        "5.1 Leadership and commitment...................... 12\n"
        "A.4.3 Data resources............................... 34\n"
    )
    segments = _as_dict(_strip_toc_lines(text))
    assert "cl-5.1" not in segments
    assert "A.4.3" not in segments


def test_bare_page_number_is_not_a_boundary():
    text = "4.1 Understanding the organization\nThe organization shall determine.\n\n7\n\nmore text\n"
    segments = _as_dict(text)
    assert list(segments) == ["cl-4.1"]
    assert "more text" in segments["cl-4.1"]


def test_clause_ten_is_not_confused_with_clause_one():
    text = "10.1 Continual improvement\nThe organization shall continually improve.\n"
    segments = _as_dict(text)
    assert "cl-10.1" in segments
