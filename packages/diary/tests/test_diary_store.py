from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from white_diary import DiaryEntry, delete_entry, list_entries, load_entry, write_entry


def _entry(tmp_path: Path, **kwargs) -> DiaryEntry:
    defaults = dict(song_slug="test-song", author="claude", body="some body")
    defaults.update(kwargs)
    return DiaryEntry(**defaults)


def test_write_and_load_round_trip(tmp_path):
    entry = _entry(
        tmp_path, title="Birth Story", tags=["proposal"], metadata={"score": 0.9}
    )
    write_entry(entry, tmp_path)
    loaded = load_entry(entry.id, tmp_path)
    assert loaded == entry


def test_write_creates_dir_if_absent(tmp_path):
    diary_dir = tmp_path / "diary" / "my-song"
    entry = _entry(tmp_path)
    assert not diary_dir.exists()
    write_entry(entry, diary_dir)
    assert (diary_dir / f"{entry.id}.yml").exists()


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_entry("no-such-id", tmp_path)


def test_list_sorted_ascending(tmp_path):
    t1 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 1, 1, 14, 0, 0, tzinfo=timezone.utc)

    e3 = _entry(tmp_path, created_at=t3, body="third")
    e1 = _entry(tmp_path, created_at=t1, body="first")
    e2 = _entry(tmp_path, created_at=t2, body="second")

    for e in (e3, e1, e2):
        write_entry(e, tmp_path)

    result = list_entries(tmp_path)
    assert [r.body for r in result] == ["first", "second", "third"]


def test_list_empty_when_dir_absent(tmp_path):
    assert list_entries(tmp_path / "diary" / "nonexistent-song") == []


def test_delete_removes_entry(tmp_path):
    entry = _entry(tmp_path)
    write_entry(entry, tmp_path)
    delete_entry(entry.id, tmp_path)
    with pytest.raises(FileNotFoundError):
        load_entry(entry.id, tmp_path)


def test_delete_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        delete_entry("no-such-id", tmp_path)


def test_full_yaml_round_trip(tmp_path):
    entry = DiaryEntry(
        song_slug="round-trip-song",
        phase="composition",
        author="prism",
        title="Full Entry",
        body="# Heading\n\nsome *markdown*",
        tags=["chords", "decision"],
        metadata={"score": 0.87, "candidates": 50},
    )
    write_entry(entry, tmp_path)
    loaded = load_entry(entry.id, tmp_path)
    assert loaded.id == entry.id
    assert loaded.song_slug == entry.song_slug
    assert loaded.phase == entry.phase
    assert loaded.author == entry.author
    assert loaded.title == entry.title
    assert loaded.body == entry.body
    assert loaded.tags == entry.tags
    assert loaded.metadata == entry.metadata
    assert loaded.created_at.tzinfo is not None
