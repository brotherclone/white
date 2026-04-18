"""Tests for app/tools/candidate_server.py — API endpoints via TestClient."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from fastapi.testclient import TestClient

from app.tools.candidate_server import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_review(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def _make_candidate(
    id: str,
    status: str = "pending",
    rank: int = 1,
    composite: float = 0.5,
    section: str = "",
) -> dict:
    return {
        "id": id,
        "midi_file": f"candidates/{id}.mid",
        "rank": rank,
        "section": section,
        "pattern_name": "test_pattern",
        "status": status,
        "scores": {"composite": composite, "theory": {}, "chromatic": {}},
    }


@pytest.fixture
def prod_dir(tmp_path):
    prod = tmp_path / "song_v1"
    _write_review(
        prod / "chords" / "review.yml",
        {
            "candidates": [
                _make_candidate("chord_001", status="pending", rank=1),
                _make_candidate("chord_002", status="approved", rank=2),
            ]
        },
    )
    _write_review(
        prod / "melody" / "review.yml",
        {
            "candidates": [
                _make_candidate("mel_intro_01", status="pending", section="intro"),
                _make_candidate("mel_verse_01", status="rejected", section="verse"),
            ]
        },
    )
    return prod


@pytest.fixture
def client(prod_dir):
    app = create_app(prod_dir)
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /candidates
# ---------------------------------------------------------------------------


class TestListCandidates:
    def test_returns_all_candidates(self, client):
        resp = client.get("/candidates")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 4

    def test_phase_filter(self, client):
        resp = client.get("/candidates?phase=chords")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert all(c["phase"] == "chords" for c in data)

    def test_section_filter(self, client):
        resp = client.get("/candidates?phase=melody&section=intro")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["section"] == "intro"

    def test_candidate_has_expected_fields(self, client):
        resp = client.get("/candidates")
        c = resp.json()[0]
        for field in (
            "id",
            "phase",
            "section",
            "template",
            "status",
            "rank",
            "composite_score",
            "midi_url",
            "scores",
        ):
            assert field in c, f"Missing field: {field}"

    def test_midi_url_format(self, client):
        resp = client.get("/candidates")
        for c in resp.json():
            assert c["midi_url"] == f"/midi/{c['id']}"

    def test_statuses_preserved(self, client):
        resp = client.get("/candidates")
        by_id = {c["id"]: c["status"] for c in resp.json()}
        assert by_id["chord_001"] == "pending"
        assert by_id["chord_002"] == "approved"
        assert by_id["mel_verse_01"] == "rejected"


# ---------------------------------------------------------------------------
# POST /candidates/{id}/approve
# ---------------------------------------------------------------------------


class TestApprove:
    def test_approve_updates_status(self, client, prod_dir):
        resp = client.post("/candidates/chord_001/approve")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "id": "chord_001", "status": "approved"}

    def test_approve_persisted_to_yml(self, client, prod_dir):
        client.post("/candidates/chord_001/approve")
        with open(prod_dir / "chords" / "review.yml") as f:
            data = yaml.safe_load(f)
        statuses = {c["id"]: c["status"] for c in data["candidates"]}
        assert statuses["chord_001"] == "approved"

    def test_approve_unknown_returns_404(self, client):
        resp = client.post("/candidates/does_not_exist/approve")
        assert resp.status_code == 404

    def test_approve_reflected_in_list(self, client):
        client.post("/candidates/chord_001/approve")
        resp = client.get("/candidates?phase=chords")
        by_id = {c["id"]: c["status"] for c in resp.json()}
        assert by_id["chord_001"] == "approved"


# ---------------------------------------------------------------------------
# POST /candidates/{id}/reject
# ---------------------------------------------------------------------------


class TestReject:
    def test_reject_updates_status(self, client, prod_dir):
        resp = client.post("/candidates/chord_001/reject")
        assert resp.status_code == 200
        assert resp.json()["status"] == "rejected"

    def test_reject_persisted_to_yml(self, client, prod_dir):
        client.post("/candidates/chord_001/reject")
        with open(prod_dir / "chords" / "review.yml") as f:
            data = yaml.safe_load(f)
        statuses = {c["id"]: c["status"] for c in data["candidates"]}
        assert statuses["chord_001"] == "rejected"

    def test_reject_unknown_returns_404(self, client):
        resp = client.post("/candidates/nope/reject")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /midi/{id}
# ---------------------------------------------------------------------------


class TestMidi:
    def test_midi_returns_404_when_file_missing(self, client):
        # midi_file paths in fixtures don't actually exist on disk
        resp = client.get("/midi/chord_001")
        assert resp.status_code == 404

    def test_midi_streams_file_when_present(self, client, prod_dir):
        # Write a real (minimal) MIDI file
        midi_path = prod_dir / "chords" / "candidates" / "chord_001.mid"
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        midi_path.write_bytes(b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x01\xe0")
        resp = client.get("/midi/chord_001")
        assert resp.status_code == 200
        assert "midi" in resp.headers["content-type"]

    def test_midi_unknown_candidate_returns_404(self, client):
        resp = client.get("/midi/ghost_candidate")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /production-dir
# ---------------------------------------------------------------------------


class TestProductionDir:
    def test_returns_production_dir(self, client, prod_dir):
        resp = client.get("/production-dir")
        assert resp.status_code == 200
        assert str(prod_dir) in resp.json()["production_dir"]


# ---------------------------------------------------------------------------
# POST /promote
# ---------------------------------------------------------------------------


class TestPromote:
    def test_valid_phase_returns_ok(self, client, prod_dir):
        # Pre-existing file in approved/ should NOT be counted (before/after delta)
        approved_dir = prod_dir / "chords" / "approved"
        approved_dir.mkdir(parents=True, exist_ok=True)
        (approved_dir / "old.mid").write_bytes(b"MThd")  # already there

        def _promote_side_effect(prod, phase, yes=False):
            # Simulate writing one new file during promotion
            (approved_dir / "chord_002.mid").write_bytes(b"MThd")
            return 0

        with patch(
            "app.generators.midi.production.pipeline_runner.cmd_promote",
            side_effect=_promote_side_effect,
        ):
            resp = client.post("/promote", json={"phase": "chords"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["promoted_count"] == 1  # delta, not total

    def test_invalid_phase_returns_400(self, client):
        resp = client.post("/promote", json={"phase": "violins"})
        assert resp.status_code == 400
        assert "violins" in resp.json()["detail"]

    def test_missing_review_yml_returns_404(self, client, prod_dir):
        # drums has no review.yml in the prod_dir fixture
        resp = client.post("/promote", json={"phase": "drums"})
        assert resp.status_code == 404

    def test_promotion_failure_returns_409(self, client, prod_dir):
        # review.yml exists (chords is set up in fixture) but promotion fails
        with patch(
            "app.generators.midi.production.pipeline_runner.cmd_promote", return_value=1
        ):
            resp = client.post("/promote", json={"phase": "chords"})
        assert resp.status_code == 409

    def test_all_valid_phases_accepted(self, client, prod_dir):
        for phase in ("chords", "drums", "bass", "melody", "quartet"):
            # Create review.yml so the 404 guard passes
            review_dir = prod_dir / phase
            review_dir.mkdir(parents=True, exist_ok=True)
            (review_dir / "review.yml").write_text("candidates: []")
            with patch(
                "app.generators.midi.production.pipeline_runner.cmd_promote",
                return_value=0,
            ):
                resp = client.post("/promote", json={"phase": phase})
            assert resp.status_code == 200, f"Phase {phase} should be accepted"


# ---------------------------------------------------------------------------
# POST /evolve
# ---------------------------------------------------------------------------


class TestEvolve:
    def _mock_run_ok(self, prod_dir: Path, phase: str, evolved_count: int = 3):
        """Return a mock subprocess.run that creates evolved_ MIDI files."""

        def _side_effect(cmd, **kwargs):
            candidates_dir = prod_dir / phase / "candidates"
            candidates_dir.mkdir(parents=True, exist_ok=True)
            for i in range(evolved_count):
                (candidates_dir / f"evolved_{phase}_{i:03d}.mid").write_bytes(b"MThd")
            result = MagicMock()
            result.returncode = 0
            result.stderr = ""
            return result

        return _side_effect

    def test_drums_returns_ok(self, client, prod_dir):
        with patch(
            "app.tools.candidate_server.subprocess.run",
            side_effect=self._mock_run_ok(prod_dir, "drums", 3),
        ):
            resp = client.post("/evolve", json={"phase": "drums"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["evolved_count"] == 3

    def test_bass_returns_ok(self, client, prod_dir):
        with patch(
            "app.tools.candidate_server.subprocess.run",
            side_effect=self._mock_run_ok(prod_dir, "bass", 2),
        ):
            resp = client.post("/evolve", json={"phase": "bass"})
        assert resp.status_code == 200
        assert resp.json()["evolved_count"] == 2

    def test_melody_returns_ok(self, client, prod_dir):
        with patch(
            "app.tools.candidate_server.subprocess.run",
            side_effect=self._mock_run_ok(prod_dir, "melody", 5),
        ):
            resp = client.post("/evolve", json={"phase": "melody"})
        assert resp.status_code == 200

    def test_chords_returns_400(self, client):
        resp = client.post("/evolve", json={"phase": "chords"})
        assert resp.status_code == 400
        assert "chords" in resp.json()["detail"]

    def test_quartet_returns_400(self, client):
        resp = client.post("/evolve", json={"phase": "quartet"})
        assert resp.status_code == 400

    def test_invalid_phase_returns_400(self, client):
        resp = client.post("/evolve", json={"phase": "strings"})
        assert resp.status_code == 400

    def test_subprocess_failure_returns_500(self, client):
        fail = MagicMock()
        fail.returncode = 1
        fail.stderr = "ONNX model not found"
        with patch("app.tools.candidate_server.subprocess.run", return_value=fail):
            resp = client.post("/evolve", json={"phase": "drums"})
        assert resp.status_code == 500
        assert "ONNX" in resp.json()["detail"]
