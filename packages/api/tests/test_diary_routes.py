from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from white_api.routes.diary import make_diary_router


def _client(tmp_path: Path) -> TestClient:
    """Create a TestClient wired to a temporary entries directory."""
    app = FastAPI()
    app.include_router(make_diary_router(tmp_path))
    return TestClient(app)


def _payload(**kwargs) -> dict:
    defaults = dict(song_slug="my-song", author="claude", body="test entry body")
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def test_create_returns_201_with_generated_id(tmp_path):
    client = _client(tmp_path)
    resp = client.post("/diary/my-song", json=_payload())
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"]
    assert data["author"] == "claude"
    assert data["body"] == "test entry body"


def test_create_sets_song_slug_from_url(tmp_path):
    client = _client(tmp_path)
    resp = client.post("/diary/my-song", json=_payload(song_slug="ignored"))
    assert resp.status_code == 201
    assert resp.json()["song_slug"] == "my-song"


def test_create_before_production_dir_exists(tmp_path):
    # Diary is independent of production — any song slug works immediately
    client = _client(tmp_path)
    resp = client.post(
        "/diary/brand-new-song", json=_payload(song_slug="brand-new-song")
    )
    assert resp.status_code == 201
    assert resp.json()["song_slug"] == "brand-new-song"


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


def test_list_returns_entries_in_created_at_order(tmp_path):
    client = _client(tmp_path)

    early = _payload(body="first", created_at="2026-01-01T10:00:00+00:00")
    late = _payload(body="second", created_at="2026-01-01T12:00:00+00:00")

    client.post("/diary/my-song", json=late)
    client.post("/diary/my-song", json=early)

    resp = client.get("/diary/my-song")
    assert resp.status_code == 200
    bodies = [e["body"] for e in resp.json()]
    assert bodies == ["first", "second"]


def test_list_empty_returns_empty_list(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/diary/my-song")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_song_with_no_entries_returns_empty(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/diary/brand-new-song")
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# Get single
# ---------------------------------------------------------------------------


def test_get_entry(tmp_path):
    client = _client(tmp_path)
    created = client.post("/diary/my-song", json=_payload()).json()
    entry_id = created["id"]

    resp = client.get(f"/diary/my-song/{entry_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == entry_id


def test_get_missing_entry_returns_404(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/diary/my-song/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


def test_put_replaces_entry(tmp_path):
    client = _client(tmp_path)
    created = client.post("/diary/my-song", json=_payload(body="original")).json()
    entry_id = created["id"]

    updated = {**created, "body": "updated body"}
    resp = client.put(f"/diary/my-song/{entry_id}", json=updated)
    assert resp.status_code == 200
    assert resp.json()["body"] == "updated body"

    fetched = client.get(f"/diary/my-song/{entry_id}").json()
    assert fetched["body"] == "updated body"


def test_put_missing_entry_returns_404(tmp_path):
    client = _client(tmp_path)
    resp = client.put(
        "/diary/my-song/00000000-0000-0000-0000-000000000000", json=_payload()
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_returns_204(tmp_path):
    client = _client(tmp_path)
    created = client.post("/diary/my-song", json=_payload()).json()
    entry_id = created["id"]

    resp = client.delete(f"/diary/my-song/{entry_id}")
    assert resp.status_code == 204


def test_delete_then_get_returns_404(tmp_path):
    client = _client(tmp_path)
    created = client.post("/diary/my-song", json=_payload()).json()
    entry_id = created["id"]

    client.delete(f"/diary/my-song/{entry_id}")
    resp = client.get(f"/diary/my-song/{entry_id}")
    assert resp.status_code == 404


def test_delete_missing_entry_returns_404(tmp_path):
    client = _client(tmp_path)
    resp = client.delete("/diary/my-song/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


def test_counts_empty_entries_dir_returns_empty_dict(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/diary/counts")
    assert resp.status_code == 200
    assert resp.json() == {}


def test_counts_reflects_entries_per_song(tmp_path):
    client = _client(tmp_path)
    client.post("/diary/song-a", json=_payload(song_slug="song-a"))
    client.post("/diary/song-a", json=_payload(song_slug="song-a"))
    client.post("/diary/song-b", json=_payload(song_slug="song-b"))

    resp = client.get("/diary/counts")
    assert resp.status_code == 200
    assert resp.json() == {"song-a": 2, "song-b": 1}


def test_counts_omits_songs_with_no_diary_directory(tmp_path):
    # A song that's never had an entry created has no directory at all —
    # it should be absent from the response, not present with a 0.
    client = _client(tmp_path)
    client.post("/diary/song-a", json=_payload(song_slug="song-a"))

    resp = client.get("/diary/counts")
    assert resp.json() == {"song-a": 1}
    assert "song-b" not in resp.json()


def test_counts_route_not_shadowed_by_song_slug_route(tmp_path):
    # "/counts" must resolve to the counts endpoint, not be swallowed as a
    # request for a song literally named "counts".
    client = _client(tmp_path)
    resp = client.get("/diary/counts")
    assert resp.status_code == 200
    assert isinstance(resp.json(), dict)
