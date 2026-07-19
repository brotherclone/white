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
    lifecycle_status: str | None = None,
    lp_consideration: str | None = None,
) -> str:
    prod_dir = root / thread_slug / "production" / production_slug
    prod_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"title": production_slug, "rainbow_color": "Red"}
    if lifecycle_status is not None:
        manifest["lifecycle_status"] = lifecycle_status
    if lp_consideration is not None:
        manifest["lp_consideration"] = lp_consideration
    with open(prod_dir / "manifest_bootstrap.yml", "w") as f:
        yaml.dump(manifest, f)

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

    def test_move_wrong_source_side_404(self, sw_dir, client):
        """The {side} path param must match where the song actually is."""
        song_id = _make_song(sw_dir, "thread-a", "song_one", mix_seconds=200.0)
        client.post("/sides/A/assign", json={"song_id": song_id, "position": 0})
        resp = client.post(
            "/sides/B/move",
            json={"song_id": song_id, "to_side": "C", "to_position": 0},
        )
        assert resp.status_code == 404
        # Song must not have moved.
        sides = client.get("/sides").json()["sides"]
        assert sides["A"]["songs"][0]["song_id"] == song_id
        assert sides["C"]["songs"] == []


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


class TestSongMixInfo:
    def test_info_for_song_with_mix(self, sw_dir, client):
        song_id = _make_song(sw_dir, "thread-a", "song_one", mix_seconds=42.0)
        resp = client.get(f"/songs/{song_id}/mix/info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["has_mix"] is True
        assert body["duration_seconds"] == 42.0
        assert body["mix_file"] is not None

    def test_info_for_song_without_mix(self, sw_dir, client):
        song_id = _make_song(sw_dir, "thread-a", "song_two")
        resp = client.get(f"/songs/{song_id}/mix/info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["has_mix"] is False
        assert body["mix_file"] is None
        assert body["duration_seconds"] is None

    def test_info_unknown_song_404(self, client):
        resp = client.get("/songs/nope__nope/mix/info")
        assert resp.status_code == 404


class TestSongMixStream:
    def test_stream_returns_wav_content_type(self, sw_dir, client):
        song_id = _make_song(sw_dir, "thread-a", "song_one", mix_seconds=5.0)
        resp = client.get(f"/songs/{song_id}/mix")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"

    def test_stream_404_without_mix_file_set(self, sw_dir, client):
        song_id = _make_song(sw_dir, "thread-a", "song_two")
        resp = client.get(f"/songs/{song_id}/mix")
        assert resp.status_code == 404

    def test_stream_404_when_file_missing_on_disk(self, sw_dir, client):
        song_id = _make_song(sw_dir, "thread-a", "song_one", mix_seconds=5.0)
        (sw_dir / "thread-a" / "production" / "song_one" / "mix.wav").unlink()
        resp = client.get(f"/songs/{song_id}/mix")
        assert resp.status_code == 404

    def test_stream_unknown_song_404(self, client):
        resp = client.get("/songs/nope__nope/mix")
        assert resp.status_code == 404


def _lp_consideration(client, song_id: str) -> str | None:
    songs = client.get("/songs").json()
    entry = next(s for s in songs if s["id"] == song_id)
    return entry["lp_consideration"]


class TestLpConsiderationDirectEndpoint:
    def test_default_not_considered(self, sw_dir, client):
        song_id = _make_song(sw_dir, "thread-a", "song_one")
        assert _lp_consideration(client, song_id) == "not_considered"

    def test_set_candidate(self, sw_dir, client):
        song_id = _make_song(sw_dir, "thread-a", "song_one")
        resp = client.post(
            f"/songs/{song_id}/lp-consideration", json={"status": "candidate"}
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "status": "candidate"}
        assert _lp_consideration(client, song_id) == "candidate"

    def test_invalid_status_422(self, sw_dir, client):
        song_id = _make_song(sw_dir, "thread-a", "song_one")
        resp = client.post(
            f"/songs/{song_id}/lp-consideration", json={"status": "bogus"}
        )
        assert resp.status_code == 422

    def test_unknown_song_404(self, client):
        resp = client.post(
            "/songs/nope__nope/lp-consideration", json={"status": "candidate"}
        )
        assert resp.status_code == 404


class TestLpConsiderationAutoTransitions:
    def test_assign_sets_placed(self, sw_dir, client):
        song_id = _make_song(sw_dir, "thread-a", "song_one", mix_seconds=200.0)
        client.post("/sides/A/assign", json={"song_id": song_id, "position": 0})
        assert _lp_consideration(client, song_id) == "placed"

    def test_move_keeps_placed(self, sw_dir, client):
        song_id = _make_song(sw_dir, "thread-a", "song_one", mix_seconds=200.0)
        client.post("/sides/A/assign", json={"song_id": song_id, "position": 0})
        client.post(
            "/sides/A/move",
            json={"song_id": song_id, "to_side": "C", "to_position": 0},
        )
        assert _lp_consideration(client, song_id) == "placed"

    def test_remove_reverts_to_candidate_when_mix_still_exists(self, sw_dir, client):
        song_id = _make_song(sw_dir, "thread-a", "song_one", mix_seconds=200.0)
        client.post("/sides/A/assign", json={"song_id": song_id, "position": 0})
        client.delete(f"/sides/A/songs/{song_id}")
        assert _lp_consideration(client, song_id) == "candidate"

    def test_remove_reverts_to_not_considered_when_mix_missing(self, sw_dir, client):
        song_id = _make_song(sw_dir, "thread-a", "song_one", mix_seconds=200.0)
        client.post("/sides/A/assign", json={"song_id": song_id, "position": 0})
        # Simulate the mix file having been deleted/moved after assignment.
        (sw_dir / "thread-a" / "production" / "song_one" / "mix.wav").unlink()
        client.delete(f"/sides/A/songs/{song_id}")
        assert _lp_consideration(client, song_id) == "not_considered"


def _write_chord_candidate(
    prod_dir: Path, candidate_id: str, midi_bytes: bytes
) -> None:
    """Write a minimal chords/review.yml + matching candidate MIDI file.

    Used to reproduce candidate id collisions across songs — every song's
    chord candidates are literally named chord_001..chord_010, so two
    different songs can share the exact same candidate_id.
    """
    review_dir = prod_dir / "chords"
    review_dir.mkdir(parents=True, exist_ok=True)
    with open(review_dir / "review.yml", "w") as f:
        yaml.dump(
            {
                "candidates": [
                    {
                        "id": candidate_id,
                        "midi_file": f"candidates/{candidate_id}.mid",
                        "rank": 1,
                        "status": "pending",
                        "scores": {"composite": 0.5, "theory": {}, "chromatic": {}},
                    }
                ]
            },
            f,
        )
    midi_path = review_dir / "candidates" / f"{candidate_id}.mid"
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    midi_path.write_bytes(midi_bytes)


class TestSongScopedMidi:
    """The unscoped /midi/{candidate_id} route resolves against whichever
    song the server's global `_production_dir` currently points at — since
    candidate ids like chord_001 repeat across every song, that route can't
    disambiguate. /songs/{song_id}/midi/{candidate_id} resolves directly from
    the URL instead.
    """

    def test_distinguishes_same_candidate_id_across_songs(self, sw_dir, client):
        song_a = _make_song(sw_dir, "thread-a", "song_one")
        song_b = _make_song(sw_dir, "thread-b", "song_two")
        _write_chord_candidate(
            sw_dir / "thread-a" / "production" / "song_one", "chord_001", b"AAAA"
        )
        _write_chord_candidate(
            sw_dir / "thread-b" / "production" / "song_two", "chord_001", b"BBBB"
        )

        resp_a = client.get(f"/songs/{song_a}/midi/chord_001")
        resp_b = client.get(f"/songs/{song_b}/midi/chord_001")
        assert resp_a.status_code == 200
        assert resp_b.status_code == 200
        assert resp_a.content == b"AAAA"
        assert resp_b.content == b"BBBB"

    def test_ignores_stale_active_song_global(self, sw_dir, client):
        song_a = _make_song(sw_dir, "thread-a", "song_one")
        song_b = _make_song(sw_dir, "thread-b", "song_two")
        _write_chord_candidate(
            sw_dir / "thread-a" / "production" / "song_one", "chord_001", b"AAAA"
        )
        _write_chord_candidate(
            sw_dir / "thread-b" / "production" / "song_two", "chord_001", b"BBBB"
        )

        # Activate song A globally, then request song B's candidate by URL —
        # the response must reflect the URL, not the stale active-song global.
        assert client.post("/songs/activate", json={"id": song_a}).status_code == 200
        resp = client.get(f"/songs/{song_b}/midi/chord_001")
        assert resp.content == b"BBBB"

    def test_unknown_song_returns_404(self, sw_dir, client):
        resp = client.get("/songs/does-not-exist/midi/chord_001")
        assert resp.status_code == 404

    def test_unknown_candidate_returns_404(self, sw_dir, client):
        song_a = _make_song(sw_dir, "thread-a", "song_one")
        _write_chord_candidate(
            sw_dir / "thread-a" / "production" / "song_one", "chord_001", b"AAAA"
        )
        resp = client.get(f"/songs/{song_a}/midi/ghost_candidate")
        assert resp.status_code == 404

    def test_response_has_no_store_cache_control(self, sw_dir, client):
        song_a = _make_song(sw_dir, "thread-a", "song_one")
        _write_chord_candidate(
            sw_dir / "thread-a" / "production" / "song_one", "chord_001", b"AAAA"
        )
        resp = client.get(f"/songs/{song_a}/midi/chord_001")
        assert resp.headers.get("cache-control") == "no-store"


class TestSongScopedMixSet:
    """POST /production/mix/set writes into whichever production dir the
    server's global `_production_dir` currently points at.
    /songs/{song_id}/mix/set resolves the production dir from the URL
    instead, so it can't attach a mix to the wrong song when the global is
    stale or was never pointed at the right song in the first place.
    """

    def test_writes_only_to_the_targeted_song(self, sw_dir, client):
        _make_song(sw_dir, "thread-a", "song_one", mix_seconds=1.0)
        song_b = _make_song(sw_dir, "thread-b", "song_two", mix_seconds=1.0)

        resp = client.post(f"/songs/{song_b}/mix/set", json={"path": "/tmp/new_b.mp3"})
        assert resp.status_code == 200

        prod_a = sw_dir / "thread-a" / "production" / "song_one"
        prod_b = sw_dir / "thread-b" / "production" / "song_two"
        with open(prod_b / "song_context.yml") as f:
            assert yaml.safe_load(f)["mix_file"] == "/tmp/new_b.mp3"
        with open(prod_a / "song_context.yml") as f:
            assert yaml.safe_load(f)["mix_file"] != "/tmp/new_b.mp3"

    def test_ignores_stale_active_song_global(self, sw_dir, client):
        song_a = _make_song(sw_dir, "thread-a", "song_one", mix_seconds=1.0)
        song_b = _make_song(sw_dir, "thread-b", "song_two", mix_seconds=1.0)

        # Activate song A globally, then write a mix for song B via URL — it
        # must land in song B's context, not the globally-active song A's.
        assert client.post("/songs/activate", json={"id": song_a}).status_code == 200
        client.post(f"/songs/{song_b}/mix/set", json={"path": "/tmp/new_b.mp3"})

        prod_a = sw_dir / "thread-a" / "production" / "song_one"
        prod_b = sw_dir / "thread-b" / "production" / "song_two"
        with open(prod_b / "song_context.yml") as f:
            assert yaml.safe_load(f)["mix_file"] == "/tmp/new_b.mp3"
        with open(prod_a / "song_context.yml") as f:
            assert yaml.safe_load(f)["mix_file"] != "/tmp/new_b.mp3"

    def test_unknown_song_returns_404(self, client):
        resp = client.post("/songs/does-not-exist/mix/set", json={"path": "/tmp/x.mp3"})
        assert resp.status_code == 404


class TestPlaylistEndpoints:
    def test_config_defaults_and_materializes(self, sw_dir, client):
        resp = client.get("/playlists/config")
        assert resp.status_code == 200
        assert "Listening" in resp.json()["output_dir"]
        assert (sw_dir / "playlist_config.yml").exists()

    def test_config_update_persists(self, sw_dir, client, tmp_path):
        new_dir = str(tmp_path / "MyListening")
        resp = client.post("/playlists/config", json={"output_dir": new_dir})
        assert resp.status_code == 200

        resp = client.get("/playlists/config")
        assert resp.json()["output_dir"] == new_dir

    def test_sync_buckets_songs_correctly(self, sw_dir, client, tmp_path):
        _make_song(
            sw_dir,
            "thread-a",
            "rejected_song",
            mix_seconds=10.0,
            lifecycle_status="scrapped",
        )
        _make_song(sw_dir, "thread-a", "review_song", mix_seconds=10.0)
        placed_id = _make_song(
            sw_dir,
            "thread-a",
            "placed_song",
            mix_seconds=10.0,
            lp_consideration="placed",
        )
        _make_song(sw_dir, "thread-a", "no_mix_song")

        output_dir = tmp_path / "Listening"
        client.post("/playlists/config", json={"output_dir": str(output_dir)})
        client.post("/sides/A/assign", json={"song_id": placed_id, "position": 0})

        resp = client.post("/playlists/sync")
        assert resp.status_code == 200
        assert resp.json() == {"rejects": 1, "review": 1, "wip": 1}
        assert (output_dir / "Rejects" / "REJECT_rejected_song.wav").exists()
        assert (output_dir / "Review" / "review_song.wav").exists()
        assert (output_dir / "White Album WiP" / "01_A_placed_song.wav").exists()

    def test_sync_rejects_unsafe_output_dir(self, sw_dir, client):
        client.post("/playlists/config", json={"output_dir": ""})
        resp = client.post("/playlists/sync")
        assert resp.status_code == 400
