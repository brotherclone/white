"""Tests for drift_report.py."""

from __future__ import annotations

from pathlib import Path

import mido
import pytest

from app.generators.midi.production.drift_report import (
    parse_timecode,
    levenshtein,
    compare_section,
    generate_drift_report,
    segment_ace_export_by_arrangement,
    write_drift_report,
)

BPM = 60
TPB = 480


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_arrangement(tmp_path: Path, rows: list[tuple]) -> Path:
    """Write a minimal arrangement.txt.

    rows: (start_tc, label, track_num, end_tc)
    """
    p = tmp_path / "arrangement.txt"
    lines = [f"{r[0]}\t {r[1]}\t {r[2]}\t {r[3]}" for r in rows]
    p.write_text("\n".join(lines))
    return p


def _make_midi(tmp_path: Path, notes: list[tuple], name: str = "loop.mid") -> Path:
    """Build a minimal MIDI with note_on events.

    notes: (tick, pitch, velocity)
    """
    mid = mido.MidiFile(ticks_per_beat=TPB)
    t = mido.MidiTrack()
    mid.tracks.append(t)
    prev_tick = 0
    for tick, pitch, vel in sorted(notes):
        t.append(
            mido.Message("note_on", note=pitch, velocity=vel, time=tick - prev_tick)
        )
        prev_tick = tick
    t.append(mido.MetaMessage("end_of_track", time=0))
    p = tmp_path / name
    mid.save(str(p))
    return p


def _word(
    word: str, start_beat: float, pitch: int = 60, end_beat: float = None
) -> dict:
    end_beat = end_beat if end_beat is not None else start_beat + 1.0
    return {
        "word": word,
        "start_beat": start_beat,
        "end_beat": end_beat,
        "pitch": pitch,
        "velocity": 100,
        "start_tick": int(start_beat * TPB),
        "end_tick": int(end_beat * TPB),
        "syllable_count": 1,
    }


# ---------------------------------------------------------------------------
# _parse_timecode
# ---------------------------------------------------------------------------


class TestParseTimecode:
    def test_song_start(self):
        assert parse_timecode("01:00:00:00.00") == pytest.approx(0.0)

    def test_36_seconds(self):
        assert parse_timecode("01:00:36:00.00") == pytest.approx(36.0)

    def test_over_one_minute(self):
        assert parse_timecode("01:01:12:00.00") == pytest.approx(72.0)

    def test_frame_offset(self):
        # frame 6 at 30fps = 0.2s, subframe 20/3000 ≈ 0.00667s
        result = parse_timecode("01:03:36:06.20")
        assert result == pytest.approx(216 + 6 / 30 + 20 / 3000, rel=1e-4)


# ---------------------------------------------------------------------------
# segment_ace_export_by_arrangement
# ---------------------------------------------------------------------------


class TestSegmentAceExportByArrangement:
    def _arrangement(self, tmp_path):
        return _make_arrangement(
            tmp_path,
            [
                # track 4 melody sections: 36–54s and 54–72s
                ("01:00:00:00.00", "drums_intro", 2, "01:00:36:00.00"),
                ("01:00:36:00.00", "verse", 4, "01:00:54:00.00"),
                ("01:00:54:00.00", "chorus", 4, "01:01:12:00.00"),
            ],
        )

    def test_words_assigned_to_correct_section(self, tmp_path):
        arr = self._arrangement(tmp_path)
        words = [_word("rust", 36.0), _word("on", 37.0), _word("the", 54.0)]
        result = segment_ace_export_by_arrangement(words, arr, BPM)
        assert "verse" in result
        assert "chorus" in result
        assert len(result["verse"]) == 2
        assert len(result["chorus"]) == 1

    def test_words_before_first_section_omitted(self, tmp_path):
        arr = self._arrangement(tmp_path)
        words = [_word("early", 10.0), _word("rust", 36.0)]
        result = segment_ace_export_by_arrangement(words, arr, BPM)
        # "early" at 10s is before any track-4 section
        all_words = [w for wlist in result.values() for w in wlist]
        assert all(w["word"] != "early" for w in all_words)

    def test_words_in_gap_omitted(self, tmp_path):
        # Gap between sections (only possible if there's a non-contiguous arrangement)
        arr = _make_arrangement(
            tmp_path,
            [
                ("01:00:36:00.00", "verse", 4, "01:00:54:00.00"),
                # gap at 54–60s
                ("01:01:00:00.00", "chorus", 4, "01:01:18:00.00"),
            ],
        )
        words = [_word("gap", 57.0), _word("chorus_word", 60.0)]
        result = segment_ace_export_by_arrangement(words, arr, BPM)
        all_words = [w for wlist in result.values() for w in wlist]
        assert all(w["word"] != "gap" for w in all_words)

    def test_word_straddling_boundary_goes_to_section_it_starts_in(self, tmp_path):
        arr = self._arrangement(tmp_path)
        # Word at exactly 54.0 → chorus starts at 54.0 → goes to chorus
        word = _word("boundary", 54.0, end_beat=55.0)
        result = segment_ace_export_by_arrangement([word], arr, BPM)
        assert "chorus" in result
        assert result["chorus"][0]["word"] == "boundary"
        assert "verse" not in result

    def test_empty_sections_omitted(self, tmp_path):
        arr = self._arrangement(tmp_path)
        words = [_word("rust", 36.0)]
        result = segment_ace_export_by_arrangement(words, arr, BPM)
        assert "chorus" not in result


# ---------------------------------------------------------------------------
# _levenshtein
# ---------------------------------------------------------------------------


class TestLevenshtein:
    def test_identical(self):
        assert levenshtein(["a", "b", "c"], ["a", "b", "c"]) == 0

    def test_one_insertion(self):
        assert levenshtein(["a", "b"], ["a", "x", "b"]) == 1

    def test_one_deletion(self):
        assert levenshtein(["a", "x", "b"], ["a", "b"]) == 1

    def test_one_substitution(self):
        assert levenshtein(["a", "b", "c"], ["a", "x", "c"]) == 1

    def test_empty_inputs(self):
        assert levenshtein([], []) == 0
        assert levenshtein(["a"], []) == 1
        assert levenshtein([], ["a"]) == 1


# ---------------------------------------------------------------------------
# compare_section
# ---------------------------------------------------------------------------


class TestCompareSection:
    def test_identical_section(self, tmp_path):
        # Approved MIDI: notes at ticks 0, 480, 960 with pitches 60, 62, 64
        midi = _make_midi(tmp_path, [(0, 60, 100), (TPB, 62, 100), (2 * TPB, 64, 100)])
        # ACE words at beats 0, 1, 2 with same pitches
        words = [_word("a", 0.0, 60), _word("b", 1.0, 62), _word("c", 2.0, 64)]
        result = compare_section(midi, words)
        assert result["pitch_match_pct"] == 1.0
        assert result["rhythm_drift_beats"] == pytest.approx(0.0)
        assert result["note_count_delta"] == 0

    def test_transposed_section_reduces_pitch_match(self, tmp_path):
        # Approved: 60, 62, 64; ACE: 72, 74, 76 (12 semitones up, outside 2-semitone tolerance)
        midi = _make_midi(tmp_path, [(0, 60, 100), (TPB, 62, 100), (2 * TPB, 64, 100)])
        words = [_word("a", 0.0, 72), _word("b", 1.0, 74), _word("c", 2.0, 76)]
        result = compare_section(midi, words)
        assert result["pitch_match_pct"] == 0.0

    def test_within_tolerance_matches(self, tmp_path):
        # Approved: 60; ACE: 61 (1 semitone → within 2 → match)
        midi = _make_midi(tmp_path, [(0, 60, 100)])
        words = [_word("a", 0.0, 61)]
        result = compare_section(midi, words)
        assert result["pitch_match_pct"] == 1.0

    def test_note_count_delta(self, tmp_path):
        # Approved has 3 notes, ACE has 5 words
        midi = _make_midi(tmp_path, [(0, 60, 100), (TPB, 62, 100), (2 * TPB, 64, 100)])
        words = [_word("w", float(i), 60) for i in range(5)]
        result = compare_section(midi, words)
        assert result["note_count_delta"] == 2  # 5 - 3

    def test_empty_ace_events(self, tmp_path):
        midi = _make_midi(tmp_path, [(0, 60, 100)])
        result = compare_section(midi, [])
        assert result["pitch_match_pct"] is None
        assert result["rhythm_drift_beats"] is None

    def test_lyric_edit_distance_is_none(self, tmp_path):
        midi = _make_midi(tmp_path, [(0, 60, 100)])
        words = [_word("a", 0.0)]
        result = compare_section(midi, words)
        assert result["lyric_edit_distance"] is None


# ---------------------------------------------------------------------------
# generate_drift_report
# ---------------------------------------------------------------------------


class TestGenerateDriftReport:
    def _setup(self, tmp_path):
        """Build a minimal production directory for drift report testing."""
        prod = tmp_path / "prod"
        prod.mkdir()

        # ACE MIDI export (VocalSynthv0_song.mid directly in prod)
        mid = mido.MidiFile(ticks_per_beat=TPB)
        t0 = mido.MidiTrack()
        mid.tracks.append(t0)
        t0.append(mido.MetaMessage("set_tempo", tempo=1_000_000, time=0))
        t0.append(mido.MetaMessage("end_of_track", time=0))
        t1 = mido.MidiTrack()
        mid.tracks.append(t1)
        # Two words: "rust" at beat 36, "on" at beat 37
        t1.append(mido.MetaMessage("lyrics", text="rust", time=36 * TPB))
        t1.append(mido.Message("note_on", note=60, velocity=100, time=0))
        t1.append(mido.MetaMessage("lyrics", text="on", time=TPB))
        t1.append(mido.Message("note_on", note=62, velocity=100, time=0))
        t1.append(mido.MetaMessage("end_of_track", time=0))
        mid.save(str(prod / "VocalSynthv0_test.mid"))

        # arrangement.txt: one melody section 36–54s
        arr_path = prod / "arrangement.txt"
        arr_path.write_text("01:00:36:00.00\t verse\t 4\t 01:00:54:00.00\n")

        # chords/review.yml
        chords_dir = prod / "chords"
        chords_dir.mkdir()
        (chords_dir / "review.yml").write_text("bpm: 60\ntime_sig: '4/4'\n")

        # melody/approved/verse.mid
        melody_dir = prod / "melody" / "approved"
        melody_dir.mkdir(parents=True)
        loop_mid = mido.MidiFile(ticks_per_beat=TPB)
        lt = mido.MidiTrack()
        loop_mid.tracks.append(lt)
        lt.append(mido.Message("note_on", note=60, velocity=100, time=0))
        lt.append(mido.Message("note_on", note=62, velocity=100, time=TPB))
        lt.append(mido.MetaMessage("end_of_track", time=0))
        loop_mid.save(str(melody_dir / "verse.mid"))

        return prod

    def test_report_has_expected_top_level_keys(self, tmp_path):
        prod = self._setup(tmp_path)
        report = generate_drift_report(prod)
        assert "overall_pitch_match" in report
        assert "overall_rhythm_drift" in report
        assert "total_lyric_edits" in report
        assert "total_word_count" in report
        assert "sections" in report

    def test_total_word_count(self, tmp_path):
        prod = self._setup(tmp_path)
        report = generate_drift_report(prod)
        assert report["total_word_count"] == 2

    def test_sections_list_nonempty(self, tmp_path):
        prod = self._setup(tmp_path)
        report = generate_drift_report(prod)
        assert len(report["sections"]) >= 1

    def test_write_drift_report(self, tmp_path):
        prod = self._setup(tmp_path)
        report = generate_drift_report(prod)
        out = write_drift_report(prod, report)
        assert out.exists()
        import yaml

        data = yaml.safe_load(out.read_text())
        assert "overall_pitch_match" in data


# ---------------------------------------------------------------------------
# Integration: real blue song
# ---------------------------------------------------------------------------


BLUE_PROD = Path(
    "shrink_wrapped/white-the-breathing-machine-learns-to-sing/"
    "production/blue__rust_signal_memorial_v1"
)


@pytest.mark.skipif(
    not BLUE_PROD.exists(), reason="Blue song production dir not present"
)
class TestIntegrationBlue:
    def test_drift_report_written(self, tmp_path):
        report = generate_drift_report(BLUE_PROD)
        out = write_drift_report(tmp_path, report)
        assert out.exists()
        assert "overall_pitch_match" in report

    def test_report_has_sections(self):
        report = generate_drift_report(BLUE_PROD)
        assert len(report["sections"]) > 0

    def test_total_word_count_matches_ace_export(self):
        from app.generators.midi.production.ace_studio_import import load_ace_export

        words = load_ace_export(BLUE_PROD)
        report = generate_drift_report(BLUE_PROD)
        assert report["total_word_count"] == len(words)
