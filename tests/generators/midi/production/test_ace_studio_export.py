"""Tests for ace_studio_export registry helpers and export function."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import mido
import pytest
import yaml

from app.generators.midi.production.ace_studio_export import (
    export_to_ace_studio,
    load_singer_registry,
    partition_notes_by_section,
    resolve_ace_voice_name,
)

# ---------------------------------------------------------------------------
# load_singer_registry
# ---------------------------------------------------------------------------


def test_load_singer_registry_happy_path(tmp_path):
    yml = tmp_path / "singer_voices.yml"
    yml.write_text(
        textwrap.dedent(
            """\
            singers:
              Shirley:
                ace_studio_voice: Elirah
                voice_type: alto
              Busyayo:
                ace_studio_voice: Golden G
                voice_type: baritone
              Unknown:
                ace_studio_voice: null
            """
        ),
        encoding="utf-8",
    )
    registry = load_singer_registry(yml)
    assert registry["shirley"]["ace_studio_voice"] == "Elirah"
    assert registry["busyayo"]["ace_studio_voice"] == "Golden G"
    assert registry["unknown"]["ace_studio_voice"] is None


def test_load_singer_registry_missing_file(tmp_path):
    registry = load_singer_registry(tmp_path / "nonexistent.yml")
    assert registry == {}


# ---------------------------------------------------------------------------
# resolve_ace_voice_name
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry():
    return {
        "shirley": {"ace_studio_voice": "Elirah"},
        "gabriel": {"ace_studio_voice": "Mangus"},
        "nullvoice": {"ace_studio_voice": None},
    }


def test_resolve_mapped_confirmed(registry):
    assert resolve_ace_voice_name("Shirley", registry) == "Elirah"


def test_resolve_case_insensitive(registry):
    assert resolve_ace_voice_name("shirley", registry) == "Elirah"
    assert resolve_ace_voice_name("SHIRLEY", registry) == "Elirah"


def test_resolve_mapped_null_falls_back_to_white_name(registry):
    result = resolve_ace_voice_name("NullVoice", registry)
    assert result == "NullVoice"


def test_resolve_not_in_registry_falls_back_to_white_name(registry):
    result = resolve_ace_voice_name("LegacyName", registry)
    assert result == "LegacyName"


def test_resolve_empty_singer(registry):
    assert resolve_ace_voice_name("", registry) == ""


# ---------------------------------------------------------------------------
# export_to_ace_studio — integration (mocked AceStudioClient)
# ---------------------------------------------------------------------------


def _make_production_dir(tmp_path: Path, singer: str = "Shirley") -> Path:
    """Scaffold a minimal production directory for export tests."""
    prod = tmp_path / "production" / "test_song_v1"

    # production_plan.yml
    (prod).mkdir(parents=True)
    (prod / "production_plan.yml").write_text(
        "bpm: 120\ntime_sig: 4/4\ntitle: Test Song\n", encoding="utf-8"
    )

    # melody/lyrics.txt
    (prod / "melody").mkdir()
    (prod / "melody" / "lyrics.txt").write_text("hello world\n", encoding="utf-8")
    (prod / "melody" / "review.yml").write_text(f"singer: {singer}\n", encoding="utf-8")

    # assembled/assembled_melody.mid — single note C4
    (prod / "assembled").mkdir()
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=80, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    mid.save(str(prod / "assembled" / "assembled_melody.mid"))

    return prod


def _mock_ace_client():
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.list_tracks.return_value = [{"index": 0, "name": "Track 1"}]
    client.find_available_track.return_value = 0
    client.find_singer.return_value = [{"id": 42, "name": "Elirah"}]
    client.add_section_clips.return_value = []
    client.get_project_info.return_value = {"projectName": "Test Song"}
    return client


def test_export_uses_ace_voice_name_for_shirley(tmp_path):
    prod = _make_production_dir(tmp_path, singer="Shirley")
    mock_client = _mock_ace_client()

    with patch(
        "app.generators.midi.production.ace_studio_export.AceStudioClient",
        return_value=mock_client,
    ):
        result = export_to_ace_studio(prod)

    assert result is not None
    # find_singer must be called with the ACE Studio voice name, not "Shirley"
    mock_client.find_singer.assert_called_once_with("Elirah")
    assert result["singer"] == "Shirley"
    assert result["ace_studio_voice"] == "Elirah"


def test_export_unknown_singer_falls_back_to_white_name(tmp_path):
    prod = _make_production_dir(tmp_path, singer="GhostSinger")
    mock_client = _mock_ace_client()
    mock_client.find_singer.return_value = []

    with patch(
        "app.generators.midi.production.ace_studio_export.AceStudioClient",
        return_value=mock_client,
    ):
        result = export_to_ace_studio(prod)

    assert result is not None
    mock_client.find_singer.assert_called_once_with("GhostSinger")
    assert result["ace_studio_voice"] == "GhostSinger"


# ---------------------------------------------------------------------------
# partition_notes_by_section
# ---------------------------------------------------------------------------


class TestPartitionNotesBySection:
    def test_single_section_gets_all_notes(self):
        notes = [
            {"pos": 0, "pitch": 60, "dur": 480},
            {"pos": 480, "pitch": 62, "dur": 480},
        ]
        sections = [{"name": "verse", "bars": 4, "play_count": 1}]
        result = partition_notes_by_section(notes, sections, bpm=120)
        assert len(result) == 1
        assert result[0]["name"] == "verse"
        assert len(result[0]["notes"]) == 2

    def test_notes_split_across_sections(self):
        tpb = 480
        bar_ticks = tpb * 4
        notes = [
            {"pos": 0, "pitch": 60, "dur": 480},
            {"pos": bar_ticks * 2, "pitch": 62, "dur": 480},  # in chorus
        ]
        sections = [
            {"name": "verse", "bars": 2, "play_count": 1},
            {"name": "chorus", "bars": 2, "play_count": 1},
        ]
        result = partition_notes_by_section(notes, sections, bpm=120, tpb=tpb)
        assert result[0]["name"] == "verse"
        assert len(result[0]["notes"]) == 1
        assert result[0]["notes"][0]["pos"] == 0  # relative to section start

        assert result[1]["name"] == "chorus"
        assert len(result[1]["notes"]) == 1
        assert (
            result[1]["notes"][0]["pos"] == 0
        )  # relative to section start (was bar_ticks*2)


# ---------------------------------------------------------------------------
# song_context ace_studio block
# ---------------------------------------------------------------------------


def _make_production_dir_with_context(tmp_path, singer="Shirley"):
    prod = _make_production_dir(tmp_path, singer=singer)
    # Add a song_context.yml
    (prod / "song_context.yml").write_text(
        yaml.dump({"title": "Test Song", "bpm": 120}), encoding="utf-8"
    )
    return prod


def test_export_writes_ace_studio_block_to_song_context(tmp_path):
    prod = _make_production_dir_with_context(tmp_path, singer="Shirley")
    mock_client = _mock_ace_client()

    with patch(
        "app.generators.midi.production.ace_studio_export.AceStudioClient",
        return_value=mock_client,
    ):
        result = export_to_ace_studio(prod)

    assert result is not None
    ctx = yaml.safe_load((prod / "song_context.yml").read_text(encoding="utf-8"))
    assert "ace_studio" in ctx
    assert ctx["ace_studio"]["track_index"] == 0
    assert ctx["ace_studio"]["singer"] == "Shirley"
    assert ctx["ace_studio"]["render_path"] is None


def test_export_overwrites_existing_ace_studio_block(tmp_path, caplog):
    import logging

    prod = _make_production_dir_with_context(tmp_path, singer="Shirley")
    # Pre-existing block
    ctx = {
        "title": "Test",
        "ace_studio": {"exported_at": "2026-01-01T00:00:00Z", "singer": "Gabriel"},
    }
    (prod / "song_context.yml").write_text(yaml.dump(ctx), encoding="utf-8")

    mock_client = _mock_ace_client()
    with patch(
        "app.generators.midi.production.ace_studio_export.AceStudioClient",
        return_value=mock_client,
    ):
        with caplog.at_level(logging.WARNING):
            result = export_to_ace_studio(prod)

    assert result is not None
    assert "2026-01-01" in caplog.text
    new_ctx = yaml.safe_load((prod / "song_context.yml").read_text(encoding="utf-8"))
    assert new_ctx["ace_studio"]["singer"] == "Shirley"
