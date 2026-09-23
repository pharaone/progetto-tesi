"""Unit tests for the ISO indexing status that gates the analysis.

AS-1 indexes the standard in a background thread while /health already
answers, so an analysis could start on a partial index and silently evaluate
only part of the standard. The status file records whether indexing
completed; agents and orchestrator refuse to start until it says "ready".

Pure unit tests: embedding and ChromaDB writes are replaced by fakes.
"""

from __future__ import annotations

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("langchain")

import rag.iso_indexing as iso  # noqa: E402


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Isolated CHROMADB_PATH, an ISO docs dir with one file, fake indexing."""
    monkeypatch.setenv("CHROMADB_PATH", str(tmp_path / "chromadb"))
    docs = tmp_path / "iso_docs"
    docs.mkdir()
    (docs / "iso42001_full.txt").write_text("4.1 Understanding the organization\nText.")

    calls = {"indexed": [], "reset": [], "state_during_indexing": None}

    def fake_index(path, collection):
        calls["indexed"].append(collection)
        calls["state_during_indexing"] = iso.iso_index_status()["state"]
        return 10

    monkeypatch.setattr(iso, "index_iso_document", fake_index)
    monkeypatch.setattr(iso, "initialize_collections", lambda: None)
    monkeypatch.setattr(iso, "reset_collection", lambda name: calls["reset"].append(name))
    return docs, calls


def test_status_is_missing_before_any_indexing(workspace):
    assert iso.iso_index_status()["state"] == iso.STATE_MISSING
    assert not iso.iso_index_ready()


def test_status_is_indexing_while_running_and_ready_after(workspace):
    docs, calls = workspace
    iso.index_directory(str(docs))

    assert calls["state_during_indexing"] == iso.STATE_INDEXING
    assert iso.iso_index_ready()


def test_a_file_that_fails_to_index_leaves_the_index_not_ready(workspace, monkeypatch):
    docs, _ = workspace

    def broken(path, collection):
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(iso, "index_iso_document", broken)
    iso.index_directory(str(docs))

    assert iso.iso_index_status()["state"] == iso.STATE_FAILED


def test_an_exception_during_indexing_marks_failed(workspace, monkeypatch):
    docs, _ = workspace

    def crash():
        raise RuntimeError("chromadb unavailable")

    monkeypatch.setattr(iso, "initialize_collections", crash)
    with pytest.raises(RuntimeError):
        iso.index_directory(str(docs))

    assert iso.iso_index_status()["state"] == iso.STATE_FAILED


def test_startup_skips_an_index_already_complete(workspace):
    docs, calls = workspace
    iso.index_directory(str(docs))
    calls["indexed"].clear()

    iso.ensure_iso_indexed(str(docs))

    assert calls["indexed"] == []


def test_startup_rebuilds_an_index_interrupted_by_a_restart(workspace):
    """A status left at "indexing" means the previous run never finished."""
    docs, calls = workspace
    iso._write_status(iso.STATE_INDEXING)

    iso.ensure_iso_indexed(str(docs))

    assert calls["reset"], "an incomplete index must be rebuilt from scratch"
    assert iso.iso_index_ready()


def test_startup_without_iso_files_adopts_an_index_built_by_hand(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMADB_PATH", str(tmp_path / "chromadb"))
    monkeypatch.setattr(iso, "get_collection", lambda name: type("C", (), {"count": lambda self: 120})())

    iso.ensure_iso_indexed(str(tmp_path / "empty"))

    assert iso.iso_index_ready()


def test_startup_without_iso_files_or_index_stays_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMADB_PATH", str(tmp_path / "chromadb"))
    monkeypatch.setattr(iso, "get_collection", lambda name: type("C", (), {"count": lambda self: 0})())

    iso.ensure_iso_indexed(str(tmp_path / "empty"))

    assert iso.iso_index_status()["state"] == iso.STATE_MISSING


@pytest.mark.parametrize("service", ["as1", "as2", "as3"])
def test_agents_refuse_to_analyze_until_the_index_is_ready(service, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    import importlib

    from fastapi.testclient import TestClient

    monkeypatch.setenv("CHROMADB_PATH", str(tmp_path / "chromadb"))
    iso._write_status(iso.STATE_INDEXING)
    app = importlib.import_module(f"services.{service}.main").app

    response = TestClient(app).post("/analyze", json={
        "org_id": "test-org",
        "documents": [{"filename": "policy.txt", "content": "AI policy."}],
    })

    assert response.status_code == 503
    assert "still being indexed" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Orchestrator: documents uploaded, clarified or synced are indexed in the
# background, and an analysis started right after would race that indexing
# ---------------------------------------------------------------------------

@pytest.fixture
def orchestrator(tmp_path, monkeypatch):
    pytest.importorskip("langgraph")
    monkeypatch.setenv("CHROMADB_PATH", str(tmp_path / "chromadb"))
    iso._write_status(iso.STATE_READY)
    import orchestrator.main as orch

    monkeypatch.setattr(orch.db, "is_sync_running", lambda: False)
    monkeypatch.setattr(orch, "_pending_doc_indexing", 0)
    return orch


def _run_scheduled(tasks):
    for task in tasks.tasks:
        task.func(*task.args, **task.kwargs)


def test_analysis_waits_for_documents_being_indexed(orchestrator, monkeypatch):
    from fastapi import BackgroundTasks

    import rag.indexer

    monkeypatch.setattr(rag.indexer, "index_text_as_org_doc", lambda *a: 3)
    assert orchestrator._analysis_blocker() is None

    tasks = BackgroundTasks()
    orchestrator._schedule_doc_indexing(tasks, "AI policy.", "policy.txt", "org")

    # Blocked as soon as the upload returns, before the task even starts
    assert orchestrator._analysis_blocker()["state"] == "indexing_documents"

    _run_scheduled(tasks)
    assert orchestrator._analysis_blocker() is None


def test_a_failed_document_indexing_does_not_block_forever(orchestrator, monkeypatch):
    from fastapi import BackgroundTasks

    import rag.indexer

    def broken(*args):
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(rag.indexer, "index_text_as_org_doc", broken)
    tasks = BackgroundTasks()
    orchestrator._schedule_doc_indexing(tasks, "AI policy.", "policy.txt", "org")
    _run_scheduled(tasks)

    assert orchestrator._analysis_blocker() is None


def test_analysis_waits_for_a_running_sync(orchestrator, monkeypatch):
    monkeypatch.setattr(orchestrator.db, "is_sync_running", lambda: True)
    assert orchestrator._analysis_blocker()["state"] == "indexing_documents"


def test_analysis_waits_for_the_iso_index_first(orchestrator):
    iso._write_status(iso.STATE_INDEXING)
    assert orchestrator._analysis_blocker()["state"] == iso.STATE_INDEXING
