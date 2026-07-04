"""Tests for /sides endpoints (LP-side sequencing)."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
from fastapi.testclient import TestClient

from white_api.candidate_server import create_app


def _write_wav(path: Path, seconds: float, samplerate: int = 8000) -> None:
    samples = np.zeros(int(seconds * samplerate), dtype="float32")
    sf.write(str(path), samples, samplerate)


def _make_song(
    root: Path,
    thread_slug: str,
    production_slug: str,
    *,
    mix_seconds: float | None = None,
) -> str:
    prod_dir = root / thread_slug / "production" / production_slug
    prod_dir.mkdir(parents=True, exist_ok=True)
    with open(prod_dir / "manifest_bootstrap.yml", "w") as f:
        yaml.dump({"title": production_slug, "rainbow_color": "Red"}, f)

    if mix_seconds is not None:
        mix_path = prod_dir / "mix.wav"
        _write_wav(mix_path, mix_seconds)
        with open(prod_dir / "song_context.yml", "w") as f:
            yaml.dump({"mix_file": str(mix_path)}, f)

    return f"{thread_slug}__{production_slug}"


@pytest.fixture
def sw_dir(tmp_path):
    return tmp_path / "sw"


@pytest.fixture
def client(sw_dir):
    sw_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(shrink_wrapped_dir=sw_dir)
    return TestClient(app)


class TestMixInfoDuration:
    def test_duration_present_for_valid_mix(self, sw_dir, tmp_path):
        song_id = _make_song(sw_dir, "thread-a", "song_one", mix_seconds=3.0)
        prod_dir = sw_dir / "thread-a" / "production" / "song_one"
        app = create_app(production_dir=prod_dir, shrink_wrapped_dir=sw_dir)
        client = TestClient(app)
        resp = client.get("/production/mix/info")
        assert resp.status_code == 200
        assert resp.json()["duration_seconds"] == 3.0
        assert song_id  # sanity: fixture returned an id

    def test_duration_null_without_mix(self, sw_dir):
        _make_song(sw_dir, "thread-a", "song_two")
        prod_dir = sw_dir / "thread-a" / "production" / "song_two"
        app = create_app(production_dir=prod_dir, shrink_wrapped_dir=sw_dir)
        client = TestClient(app)
        resp = client.get("/production/mix/info")
        assert resp.status_code == 200
        assert resp.json()["duration_seconds"] is None
        assert resp.json()["has_mix"] is False


class TestListSides:
    def test_empty_sides_default(self, client):
        resp = client.get("/sides")
        assert resp.status_code == 200
        data = resp.json()
        assert set(data["sides"].keys()) == {"A", "B", "C", "D"}
        assert data["side_limit_seconds"] == 1200.0
        assert data["sides"]["A"]["songs"] == []
        assert data["sides"]["A"]["total_seconds"] == 0.0
        assert data["sides"]["A"]["over_limit"] is False


class TestAssignToSide:
    def test_assign_song_with_mix(self, sw_dir, client):
        song_id = _make_song(sw_dir, "thread-a", "song_one", mix_seconds=200.0)
        resp = client.post("/sides/A/assign", json={"song_id": song_id, "position": 0})
        assert resp.status_code == 200
        body = resp.json()
        assert body["songs"][0]["song_id"] == song_id
        assert body["songs"][0]["duration_seconds"] == 200.0

    def test_assign_rejects_song_without_mix(self, sw_dir, client):
        song_id = _make_song(sw_dir, "thread-a", "song_two")
        resp = client.post("/sides/A/assign", json={"song_id": song_id, "position": 0})
        assert resp.status_code == 400

    def test_assign_unknown_song_404(self, client):
        resp = client.post(
            "/sides/A/assign", json={"song_id": "nope__nope", "position": 0}
        )
        assert resp.status_code == 404

    def test_assign_unknown_side_404(self, sw_dir, client):
        song_id = _make_song(sw_dir, "thread-a", "song_one", mix_seconds=200.0)
        resp = client.post("/sides/Z/assign", json={"song_id": song_id, "position": 0})
        assert resp.status_code == 404

    def test_totals_reflect_assignment(self, sw_dir, client):
        song_id = _make_song(sw_dir, "thread-a", "song_one", mix_seconds=700.0)
        client.post("/sides/A/assign", json={"song_id": song_id, "position": 0})
        resp = client.get("/sides")
        assert resp.json()["sides"]["A"]["total_seconds"] == 700.0
        assert resp.json()["sides"]["A"]["over_limit"] is False

    def test_over_limit_flagged(self, sw_dir, client):
        id1 = _make_song(sw_dir, "thread-a", "song_one", mix_seconds=700.0)
        id2 = _make_song(sw_dir, "thread-a", "song_two", mix_seconds=700.0)
        client.post("/sides/A/assign", json={"song_id": id1, "position": 0})
        client.post("/sides/A/assign", json={"song_id": id2, "position": 1})
        resp = client.get("/sides")
        assert resp.json()["sides"]["A"]["over_limit"] is True


class TestMoveBetweenSides:
    def test_move_updates_side(self, sw_dir, client):
        song_id = _make_song(sw_dir, "thread-a", "song_one", mix_seconds=200.0)
        client.post("/sides/A/assign", json={"song_id": song_id, "position": 0})
        resp = client.post(
            "/sides/A/move",
            json={"song_id": song_id, "to_side": "C", "to_position": 0},
        )
        assert resp.status_code == 200
        sides = client.get("/sides").json()["sides"]
        assert sides["A"]["songs"] == []
        assert sides["C"]["songs"][0]["song_id"] == song_id

    def test_move_unassigned_song_404(self, client):
        resp = client.post(
            "/sides/A/move",
            json={"song_id": "ghost__ghost", "to_side": "B", "to_position": 0},
        )
        assert resp.status_code == 404


class TestRemoveFromSide:
    def test_remove_song(self, sw_dir, client):
        song_id = _make_song(sw_dir, "thread-a", "song_one", mix_seconds=200.0)
        client.post("/sides/A/assign", json={"song_id": song_id, "position": 0})
        resp = client.delete(f"/sides/A/songs/{song_id}")
        assert resp.status_code == 200
        assert client.get("/sides").json()["sides"]["A"]["songs"] == []

    def test_remove_song_not_on_side_404(self, sw_dir, client):
        song_id = _make_song(sw_dir, "thread-a", "song_one", mix_seconds=200.0)
        resp = client.delete(f"/sides/A/songs/{song_id}")
        assert resp.status_code == 404
