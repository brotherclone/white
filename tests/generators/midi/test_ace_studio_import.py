"""Tests for ace_studio_import.py."""

from __future__ import annotations

from pathlib import Path

import mido
import pytest

from app.generators.midi.production.ace_studio_import import (
    _beat_to_timestamp,
    export_lrc,
    find_ace_export,
    load_ace_export,
    merge_syllables,
    parse_ace_export,
)

TPB = 480  # ticks per beat used in test MIDIs
TEMPO_60BPM = 1_000_000  # µs/beat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_midi(events: list[tuple], tpb: int = TPB, tempo: int = TEMPO_60BPM) -> Path:
    """Build a minimal MIDI file from (delta_ticks, type, ...) tuples.

    types: ('tempo', value), ('lyric', text), ('note_on', note, vel)
    Returns a MidiFile object (in-memory, not written to disk).
    """
    mid = mido.MidiFile(ticks_per_beat=tpb)
    t0 = mido.MidiTrack()
    mid.tracks.append(t0)
    t0.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    t0.append(mido.MetaMessage("end_of_track", time=0))

    t1 = mido.MidiTrack()
    mid.tracks.append(t1)
    for dt, etype, *args in events:
        if etype == "lyric":
            t1.append(mido.MetaMessage("lyrics", text=args[0], time=dt))
        elif etype == "note_on":
            t1.append(mido.Message("note_on", note=args[0], velocity=args[1], time=dt))
    t1.append(mido.MetaMessage("end_of_track", time=0))
    return mid


def _write_midi(tmp_path: Path, mid: mido.MidiFile, name: str = "test.mid") -> Path:
    p = tmp_path / name
    mid.save(str(p))
    return p


# ---------------------------------------------------------------------------
# merge_syllables
# ---------------------------------------------------------------------------


class TestMergeSyllables:
    def _syl(self, raw, word, frag, tick, pitch=60, vel=100):
        tpb = TPB
        return {
            "raw_text": raw,
            "word": word,
            "frag_index": frag,
            "start_tick": tick,
            "start_beat": tick / tpb,
            "pitch": pitch,
            "velocity": vel,
            "end_tick": tick + tpb,
        }

    def test_single_syllable_word(self):
        syls = [self._syl("rust", "rust", 0, 0)]
        words = merge_syllables(syls)
        assert len(words) == 1
        assert words[0]["word"] == "rust"
        assert words[0]["syllable_count"] == 1

    def test_two_fragment_word(self):
        syls = [
            self._syl("iron#1", "iron", 1, 0),
            self._syl("iron#2", "iron", 2, TPB // 2),
        ]
        words = merge_syllables(syls)
        assert len(words) == 1
        assert words[0]["word"] == "iron"
        assert words[0]["syllable_count"] == 2
        assert words[0]["start_beat"] == 0.0

    def test_three_fragment_word(self):
        syls = [
            self._syl("conveyor#1", "conveyor", 1, 0),
            self._syl("conveyor#2", "conveyor", 2, TPB // 3),
            self._syl("conveyor#3", "conveyor", 3, 2 * TPB // 3),
        ]
        words = merge_syllables(syls)
        assert len(words) == 1
        assert words[0]["syllable_count"] == 3

    def test_mixed_sequence(self):
        syls = [
            self._syl("rust", "rust", 0, 0),
            self._syl("on", "on", 0, TPB),
            self._syl("the", "the", 0, 2 * TPB),
            self._syl("iron#1", "iron", 1, 3 * TPB),
            self._syl("iron#2", "iron", 2, 3 * TPB + TPB // 2),
        ]
        words = merge_syllables(syls)
        assert len(words) == 4
        assert [w["word"] for w in words] == ["rust", "on", "the", "iron"]
        assert words[3]["syllable_count"] == 2

    def test_empty_input(self):
        assert merge_syllables([]) == []

    def test_onset_pitch_from_first_fragment(self):
        syls = [
            self._syl("iron#1", "iron", 1, 0, pitch=64),
            self._syl("iron#2", "iron", 2, TPB // 2, pitch=62),
        ]
        words = merge_syllables(syls)
        assert words[0]["pitch"] == 64  # onset pitch, not second fragment


# ---------------------------------------------------------------------------
# _beat_to_timestamp
# ---------------------------------------------------------------------------


class TestBeatToTimestamp:
    def test_60bpm_beat_36(self):
        # At 60 BPM, beat 36 = 36 seconds = [00:36.00]
        assert _beat_to_timestamp(36.0, TEMPO_60BPM) == "[00:36.00]"

    def test_60bpm_beat_37_5(self):
        # beat 37.5 = 37.5s → [00:37.50]
        assert _beat_to_timestamp(37.5, TEMPO_60BPM) == "[00:37.50]"

    def test_120bpm_beat_24(self):
        # At 120 BPM (500000 µs/beat), beat 24 = 12 seconds = [00:12.00]
        assert _beat_to_timestamp(24.0, 500_000) == "[00:12.00]"

    def test_over_one_minute(self):
        # At 60 BPM, beat 90 = 90 seconds = [01:30.00]
        assert _beat_to_timestamp(90.0, TEMPO_60BPM) == "[01:30.00]"

    def test_fractional_centiseconds(self):
        # beat 36.25 at 60 BPM = 36.25s → [00:36.25]
        assert _beat_to_timestamp(36.25, TEMPO_60BPM) == "[00:36.25]"


# ---------------------------------------------------------------------------
# export_lrc
# ---------------------------------------------------------------------------


class TestExportLrc:
    def _words(self):
        return [
            {"word": "rust", "start_beat": 36.0},
            {"word": "on", "start_beat": 37.0},
            {"word": "the", "start_beat": 37.5},
            {"word": "iron", "start_beat": 38.0},
        ]

    def test_lrc_written(self, tmp_path):
        out = tmp_path / "test.lrc"
        export_lrc(self._words(), TEMPO_60BPM, out)
        assert out.exists()

    def test_lrc_line_format(self, tmp_path):
        out = tmp_path / "test.lrc"
        export_lrc(self._words(), TEMPO_60BPM, out)
        lines = out.read_text().strip().splitlines()
        assert lines[0] == "[00:36.00] rust"
        assert lines[1] == "[00:37.00] on"
        assert lines[2] == "[00:37.50] the"
        assert lines[3] == "[00:38.00] iron"

    def test_lrc_word_count(self, tmp_path):
        out = tmp_path / "test.lrc"
        export_lrc(self._words(), TEMPO_60BPM, out)
        lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
        assert len(lines) == 4

    def test_utf8_encoding(self, tmp_path):
        words = [{"word": "façade", "start_beat": 1.0}]
        out = tmp_path / "test.lrc"
        export_lrc(words, TEMPO_60BPM, out)
        assert "façade" in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# find_ace_export
# ---------------------------------------------------------------------------


class TestFindAceExport:
    def test_finds_single_version(self, tmp_path):
        folder = tmp_path / "VocalSynthv0"
        folder.mkdir()
        midi = folder / "VocalSynthv0_Anderson.mid"
        midi.touch()
        result = find_ace_export(tmp_path)
        assert result == midi

    def test_prefers_highest_version(self, tmp_path):
        (tmp_path / "VocalSynthv0").mkdir()
        (tmp_path / "VocalSynthv0" / "VocalSynthv0_Anderson.mid").touch()
        (tmp_path / "VocalSynthv1").mkdir()
        v1_midi = tmp_path / "VocalSynthv1" / "VocalSynthv1_Anderson.mid"
        v1_midi.touch()
        result = find_ace_export(tmp_path)
        assert result == v1_midi

    def test_returns_none_when_absent(self, tmp_path):
        assert find_ace_export(tmp_path) is None

    def test_returns_none_when_folder_empty(self, tmp_path):
        (tmp_path / "VocalSynthv0").mkdir()
        assert find_ace_export(tmp_path) is None


# ---------------------------------------------------------------------------
# Integration: parse real blue song MIDI
# ---------------------------------------------------------------------------


REAL_MIDI = Path(
    "shrink_wrapped/white-the-breathing-machine-learns-to-sing/"
    "production/blue__rust_signal_memorial_v1/VocalSynthv0_1_Anderson.mid"
)


@pytest.mark.skipif(not REAL_MIDI.exists(), reason="Real ACE export MIDI not present")
class TestIntegrationBlue:
    def test_word_count(self):
        words, _ = parse_ace_export(REAL_MIDI)
        # The lyric data has 285 syllable events; merged word count should be less
        assert len(words) > 100
        assert len(words) < 285

    def test_first_word_is_rust(self):
        words, _ = parse_ace_export(REAL_MIDI)
        assert words[0]["word"] == "rust"

    def test_iron_merged(self):
        words, _ = parse_ace_export(REAL_MIDI)
        iron_words = [w for w in words if w["word"] == "iron"]
        # Most iron occurrences should be merged (syllable_count >= 2);
        # orphaned #2 fragments (one exists in the real MIDI) get syllable_count=1
        assert any(w["syllable_count"] >= 2 for w in iron_words)

    def test_tempo_is_60bpm(self):
        _, tempo_us = parse_ace_export(REAL_MIDI)
        assert tempo_us == 1_000_000

    def test_lrc_generation(self, tmp_path):
        words, tempo_us = parse_ace_export(REAL_MIDI)
        out = tmp_path / "blue.lrc"
        export_lrc(words, tempo_us, out)
        lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
        assert lines[0].startswith("[00:36.00] rust")
        assert len(lines) == len(words)

    def test_load_ace_export_via_production_dir(self):
        prod = REAL_MIDI.parent
        words = load_ace_export(prod)
        assert words is not None
        assert words[0]["word"] == "rust"
