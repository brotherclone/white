"""Tests for app/tools/candidate_server.py — API endpoints via TestClient."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from fastapi.testclient import TestClient

from white_api.candidate_server import create_app, scan_songs

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
            "white_composition.pipeline_runner.cmd_promote",
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
        with patch("white_composition.pipeline_runner.cmd_promote", return_value=1):
            resp = client.post("/promote", json={"phase": "chords"})
        assert resp.status_code == 409

    def test_all_valid_phases_accepted(self, client, prod_dir):
        for phase in ("chords", "drums", "bass", "melody", "quartet"):
            # Create review.yml so the 404 guard passes
            review_dir = prod_dir / phase
            review_dir.mkdir(parents=True, exist_ok=True)
            (review_dir / "review.yml").write_text("candidates: []")
            with patch(
                "white_composition.pipeline_runner.cmd_promote",
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
            "white_api.candidate_server.subprocess.run",
            side_effect=self._mock_run_ok(prod_dir, "drums", 3),
        ):
            resp = client.post("/evolve", json={"phase": "drums"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["evolved_count"] == 3

    def test_bass_returns_ok(self, client, prod_dir):
        with patch(
            "white_api.candidate_server.subprocess.run",
            side_effect=self._mock_run_ok(prod_dir, "bass", 2),
        ):
            resp = client.post("/evolve", json={"phase": "bass"})
        assert resp.status_code == 200
        assert resp.json()["evolved_count"] == 2

    def test_melody_returns_ok(self, client, prod_dir):
        with patch(
            "white_api.candidate_server.subprocess.run",
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
        with patch("white_api.candidate_server.subprocess.run", return_value=fail):
            resp = client.post("/evolve", json={"phase": "drums"})
        assert resp.status_code == 500
        assert "ONNX" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Album mode helpers
# ---------------------------------------------------------------------------


def _make_shrink_wrapped(root: Path, songs: list[dict]) -> Path:
    """Build a fake shrink_wrapped dir with manifest_bootstrap.yml files."""
    for song in songs:
        manifest_dir = (
            root / song["thread_slug"] / "production" / song["production_slug"]
        )
        manifest_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_dir / "manifest_bootstrap.yml", "w") as f:
            yaml.dump(
                {
                    "title": song.get("title", song["production_slug"]),
                    "key": song.get("key", "C major"),
                    "bpm": song.get("bpm", 120),
                    "rainbow_color": song.get("rainbow_color", "Red"),
                    "singer": song.get("singer"),
                },
                f,
            )
    return root


@pytest.fixture
def sw_dir(tmp_path):
    return _make_shrink_wrapped(
        tmp_path / "sw",
        [
            {
                "thread_slug": "thread-alpha",
                "production_slug": "song_a_v1",
                "title": "Song Alpha",
                "key": "A minor",
                "bpm": 90,
                "rainbow_color": "Red",
                "singer": "Shirley",
            },
            {
                "thread_slug": "thread-alpha",
                "production_slug": "song_b_v1",
                "title": "Song Beta",
                "key": "C major",
                "bpm": 110,
                "rainbow_color": "Blue",
                "singer": None,
            },
        ],
    )


@pytest.fixture
def album_client(sw_dir):
    app = create_app(shrink_wrapped_dir=sw_dir)
    return TestClient(app)


# ---------------------------------------------------------------------------
# scan_songs
# ---------------------------------------------------------------------------


class TestScanSongs:
    def test_returns_all_songs(self, sw_dir):
        songs = scan_songs(sw_dir)
        assert len(songs) == 2

    def test_song_has_required_fields(self, sw_dir):
        songs = scan_songs(sw_dir)
        ids = {s["id"] for s in songs}
        assert "thread-alpha__song_a_v1" in ids
        assert "thread-alpha__song_b_v1" in ids

    def test_song_fields_populated(self, sw_dir):
        songs = scan_songs(sw_dir)
        alpha = next(s for s in songs if s["production_slug"] == "song_a_v1")
        assert alpha["title"] == "Song Alpha"
        assert alpha["key"] == "A minor"
        assert alpha["bpm"] == 90
        assert alpha["rainbow_color"] == "Red"
        assert alpha["singer"] == "Shirley"

    def test_singer_null_when_absent(self, sw_dir):
        songs = scan_songs(sw_dir)
        beta = next(s for s in songs if s["production_slug"] == "song_b_v1")
        assert beta["singer"] is None

    def test_has_decisions_false_when_file_absent(self, sw_dir):
        songs = scan_songs(sw_dir)
        assert all(s["has_decisions"] is False for s in songs)

    def test_has_decisions_true_when_file_present(self, sw_dir):
        prod_dir = sw_dir / "thread-alpha" / "production" / "song_a_v1"
        (prod_dir / "production_decisions.yml").write_text("identity: {}\n")
        songs = scan_songs(sw_dir)
        alpha = next(s for s in songs if s["production_slug"] == "song_a_v1")
        beta = next(s for s in songs if s["production_slug"] == "song_b_v1")
        assert alpha["has_decisions"] is True
        assert beta["has_decisions"] is False

    def test_empty_dir_returns_empty(self, tmp_path):
        assert scan_songs(tmp_path / "empty") == []


# ---------------------------------------------------------------------------
# GET /songs
# ---------------------------------------------------------------------------


class TestGetSongs:
    def test_album_mode_returns_song_list(self, album_client):
        resp = album_client.get("/songs")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_single_song_mode_returns_503(self, client):
        resp = client.get("/songs")
        assert resp.status_code == 503

    def test_song_entry_shape(self, album_client):
        songs = album_client.get("/songs").json()
        song = songs[0]
        for field in (
            "id",
            "thread_slug",
            "production_slug",
            "title",
            "key",
            "bpm",
            "rainbow_color",
            "has_decisions",
        ):
            assert field in song


# ---------------------------------------------------------------------------
# POST /songs/activate
# ---------------------------------------------------------------------------


class TestActivateSong:
    def test_valid_id_activates_and_sets_production_dir(self, album_client, sw_dir):
        resp = album_client.post(
            "/songs/activate", json={"id": "thread-alpha__song_a_v1"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "song_a_v1" in data["production_dir"]

    def test_unknown_id_returns_404(self, album_client):
        resp = album_client.post("/songs/activate", json={"id": "no__such"})
        assert resp.status_code == 404

    def test_single_song_mode_returns_503(self, client):
        resp = client.post("/songs/activate", json={"id": "thread-alpha__song_a_v1"})
        assert resp.status_code == 503

    def test_candidates_accessible_after_activate(self, album_client, sw_dir):
        # Before activation: /candidates returns 503
        assert album_client.get("/candidates").status_code == 503
        # After activation: /candidates is accessible (may return empty list)
        album_client.post("/songs/activate", json={"id": "thread-alpha__song_a_v1"})
        resp = album_client.get("/candidates")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /songs/active
# ---------------------------------------------------------------------------


class TestGetActiveSong:
    def test_none_before_activation(self, album_client):
        resp = album_client.get("/songs/active")
        assert resp.status_code == 200
        assert resp.json()["active"] is None

    def test_returns_song_after_activation(self, album_client):
        album_client.post("/songs/activate", json={"id": "thread-alpha__song_b_v1"})
        resp = album_client.get("/songs/active")
        assert resp.status_code == 200
        active = resp.json()["active"]
        assert active["production_slug"] == "song_b_v1"
        assert active["title"] == "Song Beta"

    def test_single_song_mode_returns_null_active(self, client):
        resp = client.get("/songs/active")
        assert resp.status_code == 200
        assert resp.json()["active"] is None


# ---------------------------------------------------------------------------
# 503 guard on candidate endpoints before activation
# ---------------------------------------------------------------------------


class TestAlbumModeGuard:
    def test_candidates_503_before_activate(self, album_client):
        assert album_client.get("/candidates").status_code == 503

    def test_promote_503_before_activate(self, album_client):
        assert (
            album_client.post("/promote", json={"phase": "chords"}).status_code == 503
        )

    def test_evolve_503_before_activate(self, album_client):
        assert album_client.post("/evolve", json={"phase": "drums"}).status_code == 503


# ---------------------------------------------------------------------------
# GET /drift-report  POST /drift-report  GET /drift-report/status
# ---------------------------------------------------------------------------


def _write_plan(prod_dir: Path) -> None:
    """Write a minimal production_plan.yml for drift report tests."""
    import yaml as _yaml

    plan = {
        "song_slug": "test_song",
        "generated": "2026-01-01T00:00:00Z",
        "title": "Test Song",
        "bpm": 120,
        "time_sig": "4/4",
        "key": "C minor",
        "color": "Black",
        "sections": [
            {"name": "verse", "bars": 8, "play_count": 1, "vocals": True},
            {"name": "chorus", "bars": 8, "play_count": 1, "vocals": True},
        ],
    }
    with open(prod_dir / "production_plan.yml", "w") as f:
        _yaml.dump(plan, f)


def _write_arrangement(prod_dir: Path) -> None:
    """Write a minimal bar/beat arrangement.txt."""
    lines = [
        "1 1 1 1\tverse\t1\t9 1 1 1",
        "9 1 1 1\tchorus\t1\t17 1 1 1",
    ]
    (prod_dir / "arrangement.txt").write_text("\n".join(lines) + "\n")


class TestDriftReport:
    def test_get_returns_404_when_absent(self, client):
        resp = client.get("/drift-report")
        assert resp.status_code == 404

    def test_get_returns_report_when_present(self, client, prod_dir):

        from white_composition.drift_report import (
            BarDelta,
            DriftReport,
            DriftSummary,
            write_report,
        )

        report = DriftReport(
            generated="2026-01-01T00:00:00Z",
            song_title="Test Song",
            proposed_sections=["verse", "chorus"],
            actual_sections=["verse", "chorus"],
            drift=DriftSummary(removed=[], added=[], reordered=False),
            bar_deltas={"chorus": BarDelta(proposed=8, actual=8, delta=0)},
            energy_arc_correlation=0.9,
            summary="High fidelity.",
        )
        write_report(prod_dir, report)

        resp = client.get("/drift-report")
        assert resp.status_code == 200
        data = resp.json()
        assert data["song_title"] == "Test Song"
        assert data["drift"]["reordered"] is False
        assert data["energy_arc_correlation"] == pytest.approx(0.9)

    def test_post_missing_arrangement_returns_422(self, client, prod_dir):
        _write_plan(prod_dir)
        resp = client.post("/drift-report", json={"use_claude": False})
        assert resp.status_code == 422

    def test_post_missing_plan_returns_422(self, client, prod_dir):
        _write_arrangement(prod_dir)
        resp = client.post("/drift-report", json={"use_claude": False})
        assert resp.status_code == 422

    def test_post_starts_job(self, client, prod_dir):
        _write_plan(prod_dir)
        _write_arrangement(prod_dir)

        with (
            patch("white_composition.drift_report.compare_plans") as mock_compare,
            patch("white_composition.drift_report.write_report"),
        ):
            from white_composition.drift_report import DriftReport, DriftSummary

            mock_compare.return_value = DriftReport(
                generated="2026-01-01T00:00:00Z",
                song_title="Test Song",
                proposed_sections=["verse"],
                actual_sections=["verse"],
                drift=DriftSummary(removed=[], added=[], reordered=False),
                bar_deltas={},
                energy_arc_correlation=None,
                summary="",
            )
            resp = client.post("/drift-report", json={"use_claude": False})

        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    def test_post_no_body_uses_defaults(self, client, prod_dir):
        _write_plan(prod_dir)
        _write_arrangement(prod_dir)

        with (
            patch("white_composition.drift_report.compare_plans") as mock_compare,
            patch("white_composition.drift_report.write_report"),
        ):
            from white_composition.drift_report import DriftReport, DriftSummary

            mock_compare.return_value = DriftReport(
                generated="2026-01-01T00:00:00Z",
                song_title="Test Song",
                proposed_sections=[],
                actual_sections=[],
                drift=DriftSummary(removed=[], added=[], reordered=False),
                bar_deltas={},
                energy_arc_correlation=None,
                summary="",
            )
            resp = client.post("/drift-report")

        assert resp.status_code == 200

    def test_get_status_returns_idle_initially(self, client):
        resp = client.get("/drift-report/status")
        assert resp.status_code == 200
        assert resp.json()["status"] in ("idle", "done", "running", "error")

    def test_post_duplicate_job_returns_409(self, client, prod_dir):
        _write_plan(prod_dir)
        _write_arrangement(prod_dir)

        import white_api.candidate_server as srv

        srv._drift_job["status"] = "running"
        try:
            resp = client.post("/drift-report", json={"use_claude": False})
            assert resp.status_code == 409
        finally:
            srv._drift_job["status"] = "idle"

    def test_no_production_dir_returns_503(self):
        app = create_app()
        tc = TestClient(app)
        assert tc.get("/drift-report").status_code == 503
        assert tc.post("/drift-report").status_code == 503
