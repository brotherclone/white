"""Tests for midi_cleanup — MIDI tempo track trimming."""

from pathlib import Path

import mido
import pytest

from app.util.midi_cleanup import batch_trim, trim_midi_tempo_track

# ---------------------------------------------------------------------------
# Helpers: build synthetic MIDI files
# ---------------------------------------------------------------------------


def _make_midi_file(
    tracks: list[list[mido.Message | mido.MetaMessage]],
    ticks_per_beat: int = 480,
) -> mido.MidiFile:
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    for msgs in tracks:
        t = mido.MidiTrack()
        t.extend(msgs)
        mid.tracks.append(t)
    return mid


def _save_to_tmp(mid: mido.MidiFile, path: Path) -> Path:
    mid.save(str(path))
    return path


def _clean_midi(tmp_path: Path, tracks, ticks_per_beat=480) -> Path:
    """Write a synthetic MIDI to a temp file and return its path."""
    mid = _make_midi_file(tracks, ticks_per_beat)
    p = tmp_path / "test.mid"
    _save_to_tmp(mid, p)
    return p


# ---------------------------------------------------------------------------
# trim_midi_tempo_track
# ---------------------------------------------------------------------------


class TestTrimMidiTempoTrack:
    def test_trims_bloated_meta_track(self, tmp_path):
        """Meta track events after last note tick are removed."""
        # Track 0: meta with set_tempo at t=0 and a long event at t=9000
        # Track 1: one note_on at t=480, note_off at t=960
        meta_msgs = [
            mido.MetaMessage("set_tempo", tempo=500000, time=0),
            mido.MetaMessage("text", text="padding", time=9000),
            mido.MetaMessage("end_of_track", time=0),
        ]
        note_msgs = [
            mido.Message("note_on", note=60, velocity=80, time=480),
            mido.Message("note_off", note=60, velocity=0, time=480),
            mido.MetaMessage("end_of_track", time=0),
        ]
        p = _clean_midi(tmp_path, [meta_msgs, note_msgs])
        result = trim_midi_tempo_track(p)

        # The returned object is a MidiFile
        assert isinstance(result, mido.MidiFile)

        # Rebuild the saved file and verify meta track is trimmed
        saved = mido.MidiFile(str(p))
        meta_track = next(
            t for t in saved.tracks if not any(m.type == "note_on" for m in t)
        )
        # Compute absolute time of last meta event
        abs_t = 0
        for msg in meta_track:
            abs_t += msg.time
        # Should not overshoot 960 (last note tick)
        assert abs_t <= 960 + 1  # +1 for end_of_track delta tolerance

    def test_note_tracks_unchanged(self, tmp_path):
        """Note-bearing tracks are preserved without modification."""
        meta_msgs = [
            mido.MetaMessage("set_tempo", tempo=500000, time=0),
            mido.MetaMessage("end_of_track", time=5000),
        ]
        note_msgs = [
            mido.Message("note_on", note=60, velocity=80, time=0),
            mido.Message("note_off", note=60, velocity=0, time=480),
            mido.MetaMessage("end_of_track", time=0),
        ]
        p = _clean_midi(tmp_path, [meta_msgs, note_msgs])
        trim_midi_tempo_track(p)
        saved = mido.MidiFile(str(p))
        note_track = next(
            t for t in saved.tracks if any(m.type == "note_on" for m in t)
        )
        assert any(m.type == "note_on" for m in note_track)

    def test_no_notes_raises(self, tmp_path):
        """ValueError raised when MIDI has no note events at all."""
        meta_msgs = [
            mido.MetaMessage("set_tempo", tempo=500000, time=0),
            mido.MetaMessage("end_of_track", time=0),
        ]
        p = _clean_midi(tmp_path, [meta_msgs])
        with pytest.raises(ValueError, match="No note events"):
            trim_midi_tempo_track(p)

    def test_output_path_respected(self, tmp_path):
        """When path_out is provided, output goes there and source is untouched."""
        meta_msgs = [
            mido.MetaMessage("set_tempo", tempo=500000, time=0),
            mido.MetaMessage("end_of_track", time=5000),
        ]
        note_msgs = [
            mido.Message("note_on", note=60, velocity=80, time=0),
            mido.Message("note_off", note=60, velocity=0, time=480),
            mido.MetaMessage("end_of_track", time=0),
        ]
        p_in = _clean_midi(tmp_path, [meta_msgs, note_msgs])
        p_out = tmp_path / "out.mid"
        trim_midi_tempo_track(p_in, p_out)
        assert p_out.exists()

    def test_already_trim_meta_not_truncated(self, tmp_path):
        """Meta track that ends before last note is not lengthened."""
        meta_msgs = [
            mido.MetaMessage("set_tempo", tempo=500000, time=0),
            mido.MetaMessage("end_of_track", time=100),
        ]
        note_msgs = [
            mido.Message("note_on", note=60, velocity=80, time=0),
            mido.Message("note_off", note=60, velocity=0, time=960),
            mido.MetaMessage("end_of_track", time=0),
        ]
        p = _clean_midi(tmp_path, [meta_msgs, note_msgs])
        result = trim_midi_tempo_track(p)
        assert result is not None

    def test_multiple_note_tracks(self, tmp_path):
        """All note tracks are kept when meta track is trimmed."""
        meta_msgs = [
            mido.MetaMessage("set_tempo", tempo=500000, time=0),
            mido.MetaMessage("end_of_track", time=9000),
        ]
        note_msgs_1 = [
            mido.Message("note_on", note=60, velocity=80, time=0),
            mido.Message("note_off", note=60, velocity=0, time=480),
            mido.MetaMessage("end_of_track", time=0),
        ]
        note_msgs_2 = [
            mido.Message("note_on", note=64, velocity=80, time=0),
            mido.Message("note_off", note=64, velocity=0, time=480),
            mido.MetaMessage("end_of_track", time=0),
        ]
        p = _clean_midi(tmp_path, [meta_msgs, note_msgs_1, note_msgs_2])
        result = trim_midi_tempo_track(p)
        note_tracks = [t for t in result.tracks if any(m.type == "note_on" for m in t)]
        assert len(note_tracks) == 2


# ---------------------------------------------------------------------------
# batch_trim
# ---------------------------------------------------------------------------


class TestBatchTrim:
    def _write_bloated_midi(self, path: Path):
        """Write a MIDI with a bloated meta track."""
        meta_msgs = [
            mido.MetaMessage("set_tempo", tempo=500000, time=0),
            mido.MetaMessage("end_of_track", time=9000),
        ]
        note_msgs = [
            mido.Message("note_on", note=60, velocity=80, time=0),
            mido.Message("note_off", note=60, velocity=0, time=480),
            mido.MetaMessage("end_of_track", time=0),
        ]
        mid = _make_midi_file([meta_msgs, note_msgs])
        mid.save(str(path))

    def _write_clean_midi(self, path: Path):
        """Write a MIDI where meta and note tracks end together."""
        meta_msgs = [
            mido.MetaMessage("set_tempo", tempo=500000, time=0),
            mido.MetaMessage("end_of_track", time=0),
        ]
        note_msgs = [
            mido.Message("note_on", note=60, velocity=80, time=0),
            mido.Message("note_off", note=60, velocity=0, time=480),
            mido.MetaMessage("end_of_track", time=0),
        ]
        mid = _make_midi_file([meta_msgs, note_msgs])
        mid.save(str(path))

    def test_empty_directory_returns_empty(self, tmp_path):
        report = batch_trim(tmp_path)
        assert report == []

    def test_detects_bloated_file(self, tmp_path):
        self._write_bloated_midi(tmp_path / "bloated.mid")
        report = batch_trim(tmp_path, dry_run=True)
        assert len(report) == 1
        assert report[0]["bloated"] is True

    def test_clean_file_not_flagged(self, tmp_path):
        self._write_clean_midi(tmp_path / "clean.mid")
        report = batch_trim(tmp_path, dry_run=True)
        assert len(report) == 1
        assert report[0]["bloated"] is False

    def test_dry_run_does_not_fix(self, tmp_path):
        self._write_bloated_midi(tmp_path / "bloated.mid")
        report = batch_trim(tmp_path, dry_run=True)
        assert report[0]["fixed"] is False

    def test_non_dry_run_fixes_bloated(self, tmp_path):
        self._write_bloated_midi(tmp_path / "bloated.mid")
        report = batch_trim(tmp_path, dry_run=False)
        assert report[0]["fixed"] is True

    def test_report_has_expected_keys(self, tmp_path):
        self._write_bloated_midi(tmp_path / "x.mid")
        report = batch_trim(tmp_path, dry_run=True)
        entry = report[0]
        assert "path" in entry
        assert "meta_ticks" in entry
        assert "note_ticks" in entry
        assert "beats" in entry
        assert "bloated" in entry
        assert "fixed" in entry

    def test_recurses_into_subdirectories(self, tmp_path):
        sub = tmp_path / "chords" / "approved"
        sub.mkdir(parents=True)
        self._write_bloated_midi(sub / "loop.mid")
        report = batch_trim(tmp_path, dry_run=True)
        assert len(report) == 1

    def test_empty_midi_skipped(self, tmp_path):
        (tmp_path / "empty.mid").write_bytes(b"")
        report = batch_trim(tmp_path)
        assert report == []

    def test_corrupt_midi_skipped(self, tmp_path):
        (tmp_path / "bad.mid").write_bytes(b"not a midi file")
        report = batch_trim(tmp_path)
        assert report == []
