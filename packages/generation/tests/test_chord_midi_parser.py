"""Tests for chord_generator/midi_parser.py."""

from __future__ import annotations

from pathlib import Path

import mido

from white_generation.chord_generator.midi_parser import (
    parse_all_chords,
    parse_chord_metadata,
)


def _write_chord_midi(path: Path, notes: list[int]) -> None:
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    for note in notes:
        track.append(mido.Message("note_on", note=note, velocity=80, time=0))
    for note in notes:
        track.append(mido.Message("note_off", note=note, velocity=0, time=480))
    mid.save(str(path))


class TestParseChordMetadataFunction:
    def test_plain_roman_numeral(self):
        path = Path("chords/01 - C Major - A Minor/1 Triads/01 - C Major/I - C Maj.mid")
        meta = parse_chord_metadata(path)
        assert meta["function"] == "I"
        assert meta["chord_name"] == "C Maj"

    def test_flat_borrowed_function(self):
        path = Path(
            "chords/01 - C Major - A Minor/3 Borrowed & Modal Chords/01 - C Major/bII - Db13.mid"
        )
        meta = parse_chord_metadata(path)
        assert meta["function"] == "bII"
        assert meta["chord_name"] == "Db13"

    def test_sharp_borrowed_function(self):
        path = Path(
            "chords/01 - C Major - A Minor/3 Borrowed & Modal Chords/02 - A Minor/#vi - F#dim.mid"
        )
        meta = parse_chord_metadata(path)
        assert meta["function"] == "#vi"
        assert meta["chord_name"] == "F#dim"

    def test_category_reference_file_has_no_function(self):
        path = Path(
            "chords/09 - Ab Major - F Minor/3 Borrowed & Modal Chords/02 - F Minor/"
            "All Borrowed & Modal Chords (F Minor).mid"
        )
        meta = parse_chord_metadata(path)
        assert meta["function"] is None
        assert meta["chord_name"] == "All Borrowed & Modal Chords (F Minor)"


class TestParseAllChordsExcludesCategoryReferenceFiles:
    def test_all_prefixed_files_are_skipped(self, tmp_path):
        key_dir = tmp_path / "01 - C Major - A Minor" / "1 Triads" / "01 - C Major"
        key_dir.mkdir(parents=True)
        _write_chord_midi(key_dir / "I - C Maj.mid", [60, 64, 67])
        _write_chord_midi(key_dir / "All Triads (C Major).mid", list(range(48, 72)))

        chords = parse_all_chords(tmp_path)

        names = [c["chord_name"] for c in chords]
        assert "C Maj" in names
        assert not any(n.startswith("All ") for n in names)
        assert len(chords) == 1
