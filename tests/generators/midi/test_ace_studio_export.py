"""
Tests for app.generators.midi.production.ace_studio_export
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import mido
import yaml

from app.generators.midi.production.ace_studio_export import (
    export_to_ace_studio,
    flatten_lyrics,
    parse_midi_notes,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PLAN_YAML = {
    "song_slug": "green__last_pollinators_elegy_v1",
    "title": "The Silence Where Abundance Used to Hum",
    "bpm": 60,
    "time_sig": "3/4",
    "key": "G minor",
    "color": "Green",
    "vocals_planned": True,
    "sections": [],
}

MELODY_REVIEW_YAML = {
    "pipeline": "melody-generation",
    "bpm": 60,
    "singer": "Shirley",
    "color": "Green",
}

LYRICS_TXT = """\
# Song Title
# Green — Shirley
# comment

[intro]
she
knew

[— instrumental: break —]

[verse]
all
the
rows
"""


def _make_midi(tmp_path: Path, notes: list[tuple[int, int, int]]) -> Path:
    """Write a minimal MIDI file with (pitch, start_tick, dur_tick) notes."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    events: list[tuple[int, str, int]] = []
    for pitch, start, dur in notes:
        events.append((start, "on", pitch))
        events.append((start + dur, "off", pitch))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "on" else 1))

    prev_tick = 0
    for abs_tick, kind, pitch in events:
        delta = abs_tick - prev_tick
        if kind == "on":
            track.append(mido.Message("note_on", note=pitch, velocity=80, time=delta))
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))
        prev_tick = abs_tick

    path = tmp_path / "assembled_melody.mid"
    mid.save(str(path))
    return path


def _make_production_dir(tmp_path: Path) -> Path:
    """Create a minimal production directory with all required files."""
    prod = tmp_path / "production" / "green__test"
    (prod / "assembled").mkdir(parents=True)
    (prod / "melody").mkdir(parents=True)

    # Write plan
    (prod / "production_plan.yml").write_text(yaml.dump(PLAN_YAML), encoding="utf-8")
    # Write melody review
    (prod / "melody" / "review.yml").write_text(
        yaml.dump(MELODY_REVIEW_YAML), encoding="utf-8"
    )
    # Write lyrics
    (prod / "melody" / "lyrics.txt").write_text(LYRICS_TXT, encoding="utf-8")
    # Write a small assembled MIDI (4 notes)
    _make_midi(
        prod / "assembled",
        [(60, 480, 480), (62, 960, 480), (64, 1440, 480), (65, 1920, 480)],
    )
    return prod


def _mock_ace(singer_id: int = 42, project_name: str = "Test Project"):
    """Build a mock AceStudioClient context manager."""
    ace = MagicMock()
    ace.list_tracks.return_value = [{"trackIndex": 0, "name": "Vocal", "type": "sing"}]
    ace.find_singer.return_value = [{"id": singer_id, "name": "Shirley"}]
    ace.get_project_info.return_value = {"projectName": project_name}
    ace.add_clip.return_value = {}
    ace.open_editor.return_value = {}
    ace.add_notes_with_lyrics.return_value = {}
    ace.set_tempo.return_value = {}
    ace.set_time_signature.return_value = {}
    ace.load_singer.return_value = {}

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=ace)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx, ace


# ---------------------------------------------------------------------------
# parse_midi_notes
# ---------------------------------------------------------------------------


class TestParseMidiNotes:
    def test_returns_list_of_note_dicts(self, tmp_path):
        midi_path = _make_midi(tmp_path, [(60, 0, 480), (62, 480, 480)])
        notes = parse_midi_notes(midi_path)
        assert isinstance(notes, list)
        assert len(notes) == 2

    def test_note_dict_has_required_keys(self, tmp_path):
        midi_path = _make_midi(tmp_path, [(60, 0, 480)])
        note = parse_midi_notes(midi_path)[0]
        assert "pos" in note
        assert "pitch" in note
        assert "dur" in note

    def test_pitch_preserved(self, tmp_path):
        midi_path = _make_midi(tmp_path, [(67, 0, 480)])
        assert parse_midi_notes(midi_path)[0]["pitch"] == 67

    def test_notes_sorted_by_pos(self, tmp_path):
        midi_path = _make_midi(tmp_path, [(60, 960, 480), (62, 0, 480)])
        notes = parse_midi_notes(midi_path)
        assert notes[0]["pos"] < notes[1]["pos"]

    def test_dur_minimum_one(self, tmp_path):
        # Zero-duration notes should not be produced
        midi_path = _make_midi(tmp_path, [(60, 0, 1)])
        note = parse_midi_notes(midi_path)[0]
        assert note["dur"] >= 1

    def test_scales_ticks_when_ppqn_differs(self, tmp_path):
        # MIDI at 960 TPB should be rescaled to 480
        mid = mido.MidiFile(ticks_per_beat=960)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.Message("note_on", note=60, velocity=80, time=0))
        track.append(mido.Message("note_off", note=60, velocity=0, time=960))
        path = tmp_path / "scaled.mid"
        mid.save(str(path))
        notes = parse_midi_notes(path)
        assert notes[0]["dur"] == 480


# ---------------------------------------------------------------------------
# flatten_lyrics
# ---------------------------------------------------------------------------


class TestFlattenLyrics:
    def test_strips_comment_lines(self, tmp_path):
        f = tmp_path / "lyrics.txt"
        f.write_text("# comment\nhello\n")
        assert "comment" not in flatten_lyrics(f)

    def test_strips_section_headers(self, tmp_path):
        f = tmp_path / "lyrics.txt"
        f.write_text("[verse]\nhello\n")
        result = flatten_lyrics(f)
        assert "[verse]" not in result
        assert "hello" in result

    def test_strips_instrumental_markers(self, tmp_path):
        f = tmp_path / "lyrics.txt"
        f.write_text("[— instrumental: break —]\nhello\n")
        result = flatten_lyrics(f)
        assert "instrumental" not in result

    def test_strips_blank_lines(self, tmp_path):
        f = tmp_path / "lyrics.txt"
        f.write_text("\nhello\n\nworld\n")
        assert flatten_lyrics(f) == "hello world"

    def test_joins_with_spaces(self, tmp_path):
        f = tmp_path / "lyrics.txt"
        f.write_text("she\nknew\neach\nrow\n")
        assert flatten_lyrics(f) == "she knew each row"

    def test_full_fixture(self, tmp_path):
        f = tmp_path / "lyrics.txt"
        f.write_text(LYRICS_TXT)
        result = flatten_lyrics(f)
        assert "she" in result
        assert "knew" in result
        assert "all" in result
        # comments and headers stripped
        assert "#" not in result
        assert "[" not in result


# ---------------------------------------------------------------------------
# export_to_ace_studio — happy path
# ---------------------------------------------------------------------------


class TestExportHappyPath:
    def test_returns_dict(self, tmp_path):
        prod = _make_production_dir(tmp_path)
        ctx, ace = _mock_ace()
        with patch(
            "app.generators.midi.production.ace_studio_export.AceStudioClient",
            return_value=ctx,
        ):
            result = export_to_ace_studio(prod)
        assert result is not None
        assert isinstance(result, dict)

    def test_result_has_expected_keys(self, tmp_path):
        prod = _make_production_dir(tmp_path)
        ctx, ace = _mock_ace()
        with patch(
            "app.generators.midi.production.ace_studio_export.AceStudioClient",
            return_value=ctx,
        ):
            result = export_to_ace_studio(prod)
        for key in ("project_id", "track_index", "singer", "note_count", "title"):
            assert key in result

    def test_sets_bpm_from_plan(self, tmp_path):
        prod = _make_production_dir(tmp_path)
        ctx, ace = _mock_ace()
        with patch(
            "app.generators.midi.production.ace_studio_export.AceStudioClient",
            return_value=ctx,
        ):
            export_to_ace_studio(prod)
        ace.set_tempo.assert_called_once_with(60.0)

    def test_sets_time_signature_from_plan(self, tmp_path):
        prod = _make_production_dir(tmp_path)
        ctx, ace = _mock_ace()
        with patch(
            "app.generators.midi.production.ace_studio_export.AceStudioClient",
            return_value=ctx,
        ):
            export_to_ace_studio(prod)
        ace.set_time_signature.assert_called_once_with(3, 4)

    def test_loads_singer_from_melody_review(self, tmp_path):
        prod = _make_production_dir(tmp_path)
        ctx, ace = _mock_ace(singer_id=42)
        with patch(
            "app.generators.midi.production.ace_studio_export.AceStudioClient",
            return_value=ctx,
        ):
            export_to_ace_studio(prod)
        ace.find_singer.assert_called_once_with("Elirah")
        ace.load_singer.assert_called_once_with(0, 42)

    def test_adds_clip_with_correct_track(self, tmp_path):
        prod = _make_production_dir(tmp_path)
        ctx, ace = _mock_ace()
        with patch(
            "app.generators.midi.production.ace_studio_export.AceStudioClient",
            return_value=ctx,
        ):
            export_to_ace_studio(prod)
        call_args = ace.add_clip.call_args
        assert call_args.kwargs["track_index"] == 0
        assert call_args.kwargs["pos"] == 0
        assert call_args.kwargs["dur"] > 0

    def test_opens_editor_before_adding_notes(self, tmp_path):
        prod = _make_production_dir(tmp_path)
        ctx, ace = _mock_ace()
        with patch(
            "app.generators.midi.production.ace_studio_export.AceStudioClient",
            return_value=ctx,
        ):
            export_to_ace_studio(prod)
        # open_editor must be called before add_notes_with_lyrics
        calls = [c[0] for c in ace.method_calls]
        open_idx = next(i for i, c in enumerate(calls) if c == "open_editor")
        notes_idx = next(i for i, c in enumerate(calls) if c == "add_notes_with_lyrics")
        assert open_idx < notes_idx

    def test_note_count_in_result(self, tmp_path):
        prod = _make_production_dir(tmp_path)
        ctx, ace = _mock_ace()
        with patch(
            "app.generators.midi.production.ace_studio_export.AceStudioClient",
            return_value=ctx,
        ):
            result = export_to_ace_studio(prod)
        assert result["note_count"] == 4  # 4 notes in fixture MIDI

    def test_singer_in_result(self, tmp_path):
        prod = _make_production_dir(tmp_path)
        ctx, ace = _mock_ace()
        with patch(
            "app.generators.midi.production.ace_studio_export.AceStudioClient",
            return_value=ctx,
        ):
            result = export_to_ace_studio(prod)
        assert result["singer"] == "Shirley"

    def test_project_id_from_get_project_info(self, tmp_path):
        prod = _make_production_dir(tmp_path)
        ctx, ace = _mock_ace(project_name="My ACE Project")
        with patch(
            "app.generators.midi.production.ace_studio_export.AceStudioClient",
            return_value=ctx,
        ):
            result = export_to_ace_studio(prod)
        assert result["project_id"] == "My ACE Project"


# ---------------------------------------------------------------------------
# export_to_ace_studio — unreachable
# ---------------------------------------------------------------------------


class TestExportReturnsNoneWhenUnreachable:
    def test_returns_none(self, tmp_path):
        prod = _make_production_dir(tmp_path)
        ctx = MagicMock()
        ctx.__enter__.side_effect = ConnectionError("refused")
        ctx.__exit__ = MagicMock(return_value=False)
        with patch(
            "app.generators.midi.production.ace_studio_export.AceStudioClient",
            return_value=ctx,
        ):
            result = export_to_ace_studio(prod)
        assert result is None

    def test_logs_warning(self, tmp_path, caplog):
        import logging

        prod = _make_production_dir(tmp_path)
        ctx = MagicMock()
        ctx.__enter__.side_effect = ConnectionError("refused")
        ctx.__exit__ = MagicMock(return_value=False)
        with patch(
            "app.generators.midi.production.ace_studio_export.AceStudioClient",
            return_value=ctx,
        ):
            with caplog.at_level(logging.WARNING):
                export_to_ace_studio(prod)
        assert any("unreachable" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# export_to_ace_studio — missing files
# ---------------------------------------------------------------------------


class TestExportSkipsWhenFileMissing:
    def test_returns_none_when_no_assembled_midi(self, tmp_path):
        prod = _make_production_dir(tmp_path)
        (prod / "assembled" / "assembled_melody.mid").unlink()
        result = export_to_ace_studio(prod)
        assert result is None

    def test_returns_none_when_no_lyrics(self, tmp_path):
        prod = _make_production_dir(tmp_path)
        (prod / "melody" / "lyrics.txt").unlink()
        result = export_to_ace_studio(prod)
        assert result is None

    def test_returns_none_when_no_plan(self, tmp_path):
        prod = _make_production_dir(tmp_path)
        (prod / "production_plan.yml").unlink()
        result = export_to_ace_studio(prod)
        assert result is None

    def test_logs_missing_file_name(self, tmp_path, caplog):
        import logging

        prod = _make_production_dir(tmp_path)
        (prod / "assembled" / "assembled_melody.mid").unlink()
        with caplog.at_level(logging.WARNING):
            export_to_ace_studio(prod)
        assert any("assembled_melody.mid" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# export_to_ace_studio — no tracks
# ---------------------------------------------------------------------------


class TestExportSkipsWhenNoTracks:
    def test_returns_none_when_no_tracks(self, tmp_path):
        prod = _make_production_dir(tmp_path)
        ctx, ace = _mock_ace()
        ace.list_tracks.return_value = []
        with patch(
            "app.generators.midi.production.ace_studio_export.AceStudioClient",
            return_value=ctx,
        ):
            result = export_to_ace_studio(prod)
        assert result is None


# ---------------------------------------------------------------------------
# export_to_ace_studio — singer not found
# ---------------------------------------------------------------------------


class TestExportContinuesWhenSingerMissing:
    def test_returns_dict_even_without_singer(self, tmp_path):
        prod = _make_production_dir(tmp_path)
        ctx, ace = _mock_ace()
        ace.find_singer.return_value = []  # singer not found
        with patch(
            "app.generators.midi.production.ace_studio_export.AceStudioClient",
            return_value=ctx,
        ):
            result = export_to_ace_studio(prod)
        assert result is not None

    def test_singer_id_none_when_not_found(self, tmp_path):
        prod = _make_production_dir(tmp_path)
        ctx, ace = _mock_ace()
        ace.find_singer.return_value = []
        with patch(
            "app.generators.midi.production.ace_studio_export.AceStudioClient",
            return_value=ctx,
        ):
            result = export_to_ace_studio(prod)
        assert result["singer_id"] is None

    def test_does_not_call_load_singer_when_not_found(self, tmp_path):
        prod = _make_production_dir(tmp_path)
        ctx, ace = _mock_ace()
        ace.find_singer.return_value = []
        with patch(
            "app.generators.midi.production.ace_studio_export.AceStudioClient",
            return_value=ctx,
        ):
            export_to_ace_studio(prod)
        ace.load_singer.assert_not_called()

    def test_proceeds_without_melody_review(self, tmp_path):
        """Export still works if melody/review.yml is absent (no singer loaded)."""
        prod = _make_production_dir(tmp_path)
        (prod / "melody" / "review.yml").unlink()
        ctx, ace = _mock_ace()
        with patch(
            "app.generators.midi.production.ace_studio_export.AceStudioClient",
            return_value=ctx,
        ):
            result = export_to_ace_studio(prod)
        assert result is not None
        ace.load_singer.assert_not_called()
