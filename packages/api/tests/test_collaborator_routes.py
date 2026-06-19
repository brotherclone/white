"""Tests for /collaborators and /production/work-orders API routes."""

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from white_api.candidate_server import create_app


def _client(tmp_path: Path):
    """Create a TestClient in single-song mode with isolated dirs."""
    prod_dir = tmp_path / "production" / "the-song"
    prod_dir.mkdir(parents=True)
    registry_dir = tmp_path / "collaborators"
    registry_dir.mkdir()
    (prod_dir / "song_context.yml").write_text(
        yaml.dump(
            {"title": "Test Song", "bpm": 120, "key": "C major", "time_sig": "4/4"}
        )
    )
    app = create_app(production_dir=prod_dir, registry_dir=registry_dir)
    return TestClient(app), prod_dir


def _collab_payload(collaborator_id: str = "kate-koherence") -> dict:
    return {
        "id": collaborator_id,
        "name": "Kate Koherence",
        "roles": ["vocalist"],
        "email": "kate@example.com",
    }


# ---------------------------------------------------------------------------
# Collaborator CRUD
# ---------------------------------------------------------------------------


def test_list_collaborators_empty(tmp_path):
    client, _ = _client(tmp_path)
    # Point registry to tmp dir

    resp = client.get("/collaborators")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_and_get_collaborator(tmp_path):
    client, prod_dir = _client(tmp_path)
    payload = _collab_payload()

    resp = client.post("/collaborators", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"] == "kate-koherence"
    assert data["name"] == "Kate Koherence"

    resp2 = client.get("/collaborators/kate-koherence")
    assert resp2.status_code == 200
    assert resp2.json()["email"] == "kate@example.com"


def test_create_duplicate_returns_409(tmp_path):
    client, _ = _client(tmp_path)
    payload = _collab_payload()
    client.post("/collaborators", json=payload)
    resp = client.post("/collaborators", json=payload)
    assert resp.status_code == 409


def test_get_unknown_collaborator_returns_404(tmp_path):
    client, _ = _client(tmp_path)
    resp = client.get("/collaborators/nobody")
    assert resp.status_code == 404


def test_update_collaborator(tmp_path):
    client, _ = _client(tmp_path)
    payload = _collab_payload()
    client.post("/collaborators", json=payload)

    updated = {**payload, "notes": "prefers mp3 rough mixes"}
    resp = client.put("/collaborators/kate-koherence", json=updated)
    assert resp.status_code == 200
    assert resp.json()["notes"] == "prefers mp3 rough mixes"


def test_delete_collaborator(tmp_path):
    client, _ = _client(tmp_path)
    payload = _collab_payload()
    client.post("/collaborators", json=payload)

    resp = client.delete("/collaborators/kate-koherence")
    assert resp.status_code == 204

    resp2 = client.get("/collaborators/kate-koherence")
    assert resp2.status_code == 404


def test_delete_unknown_returns_404(tmp_path):
    client, _ = _client(tmp_path)
    resp = client.delete("/collaborators/nobody")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Work orders
# ---------------------------------------------------------------------------


def test_list_work_orders_empty(tmp_path):
    client, _ = _client(tmp_path)
    resp = client.get("/production/work-orders")
    assert resp.status_code == 200
    assert resp.json() == []


def test_generate_work_order(tmp_path):
    client, _ = _client(tmp_path)
    resp = client.post(
        "/production/work-orders/generate",
        json={
            "collaborator_id": "kate-koherence",
            "role": "vocalist",
            "platform": "direct",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["collaborator_id"] == "kate-koherence"
    assert data["role"] == "vocalist"
    assert data["status"] == "draft"


def test_generate_invalid_role_returns_422(tmp_path):
    client, _ = _client(tmp_path)
    resp = client.post(
        "/production/work-orders/generate",
        json={
            "collaborator_id": "kate-koherence",
            "role": "wizard",
            "platform": "direct",
        },
    )
    assert resp.status_code == 422


def test_update_work_order(tmp_path):
    client, _ = _client(tmp_path)
    # First generate
    gen = client.post(
        "/production/work-orders/generate",
        json={"collaborator_id": "kate-koherence", "role": "vocalist"},
    )
    wo = gen.json()
    wo["part_notes"] = "keep it breathy on verse 1"

    resp = client.put("/production/work-orders/kate-koherence", json=wo)
    assert resp.status_code == 200

    resp2 = client.get("/production/work-orders/kate-koherence")
    assert resp2.status_code == 200
    assert resp2.json()["part_notes"] == "keep it breathy on verse 1"


def test_draft_email_no_collaborator_returns_404(tmp_path):
    client, prod_dir = _client(tmp_path)
    # Save a work order without a collaborator in registry
    gen = client.post(
        "/production/work-orders/generate",
        json={"collaborator_id": "ghost", "role": "vocalist"},
    )
    wo = gen.json()
    client.put("/production/work-orders/ghost", json=wo)

    resp = client.post("/production/work-orders/ghost/draft-email")
    assert resp.status_code == 404


def test_draft_email_no_email_returns_422(tmp_path):
    client, _ = _client(tmp_path)
    # Create collaborator without email
    client.post(
        "/collaborators",
        json={"id": "no-email", "name": "No Email", "roles": ["vocalist"]},
    )
    gen = client.post(
        "/production/work-orders/generate",
        json={"collaborator_id": "no-email", "role": "vocalist"},
    )
    wo = gen.json()
    client.put("/production/work-orders/no-email", json=wo)

    resp = client.post("/production/work-orders/no-email/draft-email")
    assert resp.status_code == 422
    assert "no email" in resp.json()["detail"].lower()


def test_draft_email_success(tmp_path):
    client, _ = _client(tmp_path)
    client.post("/collaborators", json=_collab_payload())
    gen = client.post(
        "/production/work-orders/generate",
        json={"collaborator_id": "kate-koherence", "role": "vocalist"},
    )
    wo = gen.json()
    client.put("/production/work-orders/kate-koherence", json=wo)

    resp = client.post("/production/work-orders/kate-koherence/draft-email")
    assert resp.status_code == 200
    data = resp.json()
    assert data["to"] == "kate@example.com"
    assert "subject" in data
    assert "body" in data
