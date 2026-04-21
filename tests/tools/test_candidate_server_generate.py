"""Tests for /generate and /generate/status endpoints."""

import threading
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import app.tools.candidate_server as server_module
from app.tools.candidate_server import create_app


@pytest.fixture(autouse=True)
def reset_job_state():
    """Reset generate job state before every test."""
    server_module._generate_job = {
        "status": "idle",
        "started_at": None,
        "finished_at": None,
        "error": None,
    }
    yield


@pytest.fixture
def album_client(tmp_path):
    shrink_wrapped = tmp_path / "shrink_wrapped"
    shrink_wrapped.mkdir()
    app = create_app(shrink_wrapped_dir=shrink_wrapped)
    return TestClient(app)


def test_generate_status_idle(album_client):
    res = album_client.get("/generate/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "idle"
    assert data["started_at"] is None
    assert data["finished_at"] is None
    assert data["error"] is None


def test_generate_starts_successfully(album_client):
    # Patch the imports inside _run so no real agent executes
    with (
        patch("app.agents.workflow.concept_workflow.run_white_agent_workflow"),
        patch(
            "app.util.shrinkwrap_chain_artifacts.shrinkwrap",
            return_value={"processed": 1},
        ),
    ):
        res = album_client.post("/generate")

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "running"
    assert data["started_at"] is not None


def test_generate_returns_409_while_running(album_client):
    server_module._generate_job["status"] = "running"
    res = album_client.post("/generate")
    assert res.status_code == 409
    assert "already running" in res.json()["detail"]


def test_generate_status_done_after_job(album_client):
    """Status transitions to done after a mocked background job completes."""
    done_event = threading.Event()

    def fake_run():
        server_module._generate_job["status"] = "done"
        server_module._generate_job["finished_at"] = "2026-04-20T12:00:00+00:00"
        done_event.set()

    server_module._generate_job["status"] = "running"
    server_module._generate_job["started_at"] = "2026-04-20T11:59:00+00:00"

    t = threading.Thread(target=fake_run, daemon=True)
    t.start()
    done_event.wait(timeout=2.0)

    res = album_client.get("/generate/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "done"
    assert data["finished_at"] is not None
    assert data["error"] is None
