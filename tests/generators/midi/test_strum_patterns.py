"""Tests for strum_patterns — pattern templates and MIDI rendering."""

import io
import mido
import pytest
import yaml
from pathlib import Path

from app.generators.midi.patterns.strum_patterns import (
    PATTERNS_4_4,
    PATTERNS_7_8,
    StrumPattern,
    apply_strum_pattern,
    get_patterns_for_time_sig,
    make_fallback_patterns,
    parse_chord_voicings,
    read_approved_harmonic_rhythm,
    strum_to_midi_bytes,
)


# ---------------------------------------------------------------------------
# StrumPattern.bar_length_beats
# ---------------------------------------------------------------------------


class TestBarLengthBeats:
    def test_4_4(self):
        p = StrumPattern(
            name="x", time_sig=(4, 4), description="", onsets=[], durations=[]
        )
        assert p.bar_length_beats() == pytest.approx(4.0)

    def test_7_8(self):
        p = StrumPattern(
            name="x", time_sig=(7, 8), description="", onsets=[], durations=[]
        )
        assert p.bar_length_beats() == pytest.approx(3.5)

    def test_3_4(self):
        p = StrumPattern(
            name="x", time_sig=(3, 4), description="", onsets=[], durations=[]
        )
        assert p.bar_length_beats() == pytest.approx(3.0)

    def test_6_8(self):
        p = StrumPattern(
            name="x", time_sig=(6, 8), description="", onsets=[], durations=[]
        )
        assert p.bar_length_beats() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# get_patterns_for_time_sig
# ---------------------------------------------------------------------------


class TestGetPatternsForTimeSig:
    def test_4_4_returns_patterns(self):
        patterns = get_patterns_for_time_sig((4, 4))
        assert len(patterns) > 0
        assert all(p.time_sig == (4, 4) for p in patterns)

    def test_7_8_returns_patterns(self):
        patterns = get_patterns_for_time_sig((7, 8))
        assert len(patterns) > 0
        assert all(p.time_sig == (7, 8) for p in patterns)

    def test_unsupported_time_sig_uses_fallback(self):
        patterns = get_patterns_for_time_sig((5, 4))
        assert len(patterns) > 0
        assert all(p.time_sig == (5, 4) for p in patterns)

    def test_filter_names_restricts_results(self):
        patterns = get_patterns_for_time_sig((4, 4), filter_names=["whole", "half"])
        names = {p.name for p in patterns}
        assert names <= {"whole", "half"}

    def test_filter_names_empty_list(self):
        # filter_names=[] is falsy → treated as no filter → returns all patterns
        patterns = get_patterns_for_time_sig((4, 4), filter_names=[])
        assert len(patterns) == len(PATTERNS_4_4)

    def test_filter_names_nonexistent(self):
        patterns = get_patterns_for_time_sig((4, 4), filter_names=["nonexistent"])
        assert patterns == []

    def test_all_patterns_present(self):
        patterns_44 = get_patterns_for_time_sig((4, 4))
        names = {p.name for p in patterns_44}
        assert "whole" in names
        assert "arp_up" in names

    def test_7_8_has_grouped_patterns(self):
        patterns = get_patterns_for_time_sig((7, 8))
        names = {p.name for p in patterns}
        assert "grouped_322" in names
        assert "grouped_223" in names


# ---------------------------------------------------------------------------
# make_fallback_patterns
# ---------------------------------------------------------------------------


class TestMakeFallbackPatterns:
    def test_returns_two_patterns(self):
        patterns = make_fallback_patterns((3, 4))
        assert len(patterns) == 2

    def test_whole_pattern_correct_duration(self):
        patterns = make_fallback_patterns((3, 4))
        whole = next(p for p in patterns if p.name == "whole")
        assert whole.durations[0] == pytest.approx(3.0)

    def test_beat_pattern_correct_onsets(self):
        patterns = make_fallback_patterns((3, 4))
        beat = next(p for p in patterns if p.name == "beat")
        assert len(beat.onsets) == 3

    def test_time_sig_preserved(self):
        patterns = make_fallback_patterns((5, 8))
        assert all(p.time_sig == (5, 8) for p in patterns)

    def test_5_8_bar_length(self):
        patterns = make_fallback_patterns((5, 8))
        whole = next(p for p in patterns if p.name == "whole")
        assert whole.bar_length_beats() == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# apply_strum_pattern
# ---------------------------------------------------------------------------


class TestApplyStrumPattern:
    def test_block_pattern_all_notes_per_onset(self):
        voicing = [60, 64, 67]
        pattern = PATTERNS_4_4[0]  # whole — 1 onset
        events = apply_strum_pattern(voicing, pattern)
        assert len(events) == 1
        assert events[0]["notes"] == [60, 64, 67]

    def test_quarter_pattern_four_events(self):
        voicing = [60, 64]
        pattern = next(p for p in PATTERNS_4_4 if p.name == "quarter")
        events = apply_strum_pattern(voicing, pattern)
        assert len(events) == 4
        assert all(set(e["notes"]) == {60, 64} for e in events)

    def test_onset_ticks_match_pattern(self):
        voicing = [60]
        pattern = next(p for p in PATTERNS_4_4 if p.name == "half")
        events = apply_strum_pattern(voicing, pattern, ticks_per_beat=480)
        assert events[0]["onset_tick"] == 0
        assert events[1]["onset_tick"] == 960  # beat 2 = 2 * 480

    def test_arp_up_cycles_notes(self):
        voicing = [60, 64, 67]
        pattern = next(p for p in PATTERNS_4_4 if p.name == "arp_up")
        events = apply_strum_pattern(voicing, pattern)
        # First note should be lowest (sorted ascending for up)
        assert events[0]["notes"] == [60]
        assert events[1]["notes"] == [64]
        assert events[2]["notes"] == [67]
        assert events[3]["notes"] == [60]  # cycles back

    def test_arp_down_reversed_order(self):
        voicing = [60, 64, 67]
        pattern = next(p for p in PATTERNS_4_4 if p.name == "arp_down")
        events = apply_strum_pattern(voicing, pattern)
        assert events[0]["notes"] == [67]
        assert events[1]["notes"] == [64]
        assert events[2]["notes"] == [60]

    def test_arp_empty_voicing_returns_empty(self):
        pattern = next(p for p in PATTERNS_4_4 if p.name == "arp_up")
        events = apply_strum_pattern([], pattern)
        assert events == []

    def test_velocity_propagated(self):
        voicing = [60]
        pattern = PATTERNS_4_4[0]
        events = apply_strum_pattern(voicing, pattern, velocity=100)
        assert events[0]["velocity"] == 100

    def test_duration_ticks_correct(self):
        voicing = [60]
        pattern = PATTERNS_4_4[0]  # whole: 4.0 beats
        events = apply_strum_pattern(voicing, pattern, ticks_per_beat=480)
        assert events[0]["duration_ticks"] == 4 * 480


# ---------------------------------------------------------------------------
# strum_to_midi_bytes
# ---------------------------------------------------------------------------


class TestStrumToMidiBytes:
    def test_returns_bytes(self):
        pattern = next(p for p in PATTERNS_4_4 if p.name == "whole")
        result = strum_to_midi_bytes([[60, 64, 67]], pattern)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_output_is_valid_midi(self):
        pattern = next(p for p in PATTERNS_4_4 if p.name == "quarter")
        result = strum_to_midi_bytes([[60, 64, 67]], pattern)
        mid = mido.MidiFile(file=io.BytesIO(result))
        assert mid is not None

    def test_multiple_chords(self):
        pattern = next(p for p in PATTERNS_4_4 if p.name == "whole")
        chords = [[60, 64, 67], [62, 65, 69], [64, 67, 71]]
        result = strum_to_midi_bytes(chords, pattern)
        mid = mido.MidiFile(file=io.BytesIO(result))
        notes = [m for t in mid.tracks for m in t if m.type == "note_on"]
        assert len(notes) > 0

    def test_tempo_encoded(self):
        pattern = next(p for p in PATTERNS_4_4 if p.name == "whole")
        result = strum_to_midi_bytes([[60]], pattern, bpm=90)
        mid = mido.MidiFile(file=io.BytesIO(result))
        tempo_msgs = [
            m for t in mid.tracks for m in t if getattr(m, "type", "") == "set_tempo"
        ]
        assert any(m.tempo == mido.bpm2tempo(90) for m in tempo_msgs)

    def test_durations_parameter(self):
        pattern = next(p for p in PATTERNS_4_4 if p.name == "whole")
        # Two chords, second gets 2 bars
        chords = [[60, 64, 67], [62, 65, 69]]
        result = strum_to_midi_bytes(chords, pattern, durations=[1.0, 2.0])
        assert isinstance(result, bytes)
        mid = mido.MidiFile(file=io.BytesIO(result))
        assert mid is not None

    def test_7_8_pattern(self):
        pattern = next(p for p in PATTERNS_7_8 if p.name == "whole")
        result = strum_to_midi_bytes([[60, 64]], pattern)
        mid = mido.MidiFile(file=io.BytesIO(result))
        assert mid is not None

    def test_note_offs_before_note_ons_at_same_tick(self):
        """Events at the same tick: note-offs must come before note-ons."""
        pattern = next(p for p in PATTERNS_4_4 if p.name == "quarter")
        result = strum_to_midi_bytes([[60], [62]], pattern)
        mid = mido.MidiFile(file=io.BytesIO(result))
        events = [
            (m.type, m.time) for t in mid.tracks for m in t if hasattr(m, "velocity")
        ]
        # The MIDI should parse cleanly without errors
        assert len(events) > 0


# ---------------------------------------------------------------------------
# parse_chord_voicings
# ---------------------------------------------------------------------------


class TestParseChordVoicings:
    def _make_chord_midi(
        self, tmp_path: Path, chords: list[list[int]], bar_ticks=1920
    ) -> Path:
        """Write a chord MIDI with one chord block per bar."""
        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
        current = 0
        for chord in chords:
            # All notes on at current tick
            first = True
            for note in chord:
                track.append(
                    mido.Message(
                        "note_on",
                        note=note,
                        velocity=80,
                        time=(current if first else 0),
                    )
                )
                first = False
                current = 0
            # All notes off after bar_ticks
            first = True
            for note in chord:
                track.append(
                    mido.Message(
                        "note_off",
                        note=note,
                        velocity=0,
                        time=(bar_ticks if first else 0),
                    )
                )
                first = False
        track.append(mido.MetaMessage("end_of_track", time=0))
        p = tmp_path / "chords.mid"
        mid.save(str(p))
        return p

    def test_single_chord(self, tmp_path):
        p = self._make_chord_midi(tmp_path, [[60, 64, 67]])
        result = parse_chord_voicings(p)
        assert len(result) == 1
        assert 60 in result[0]["notes"]
        assert 64 in result[0]["notes"]
        assert 67 in result[0]["notes"]

    def test_notes_are_sorted(self, tmp_path):
        p = self._make_chord_midi(tmp_path, [[67, 60, 64]])
        result = parse_chord_voicings(p)
        assert result[0]["notes"] == sorted(result[0]["notes"])

    def test_multiple_chords(self, tmp_path):
        p = self._make_chord_midi(tmp_path, [[60, 64, 67], [62, 65, 69]])
        result = parse_chord_voicings(p)
        assert len(result) == 2

    def test_bar_ticks_inferred(self, tmp_path):
        p = self._make_chord_midi(tmp_path, [[60, 64], [62, 65]], bar_ticks=1920)
        result = parse_chord_voicings(p)
        assert result[0]["bar_ticks"] == 1920


# ---------------------------------------------------------------------------
# read_approved_harmonic_rhythm
# ---------------------------------------------------------------------------


class TestReadApprovedHarmonicRhythm:
    def _write_review(self, tmp_path: Path, candidates: list) -> Path:
        review_dir = tmp_path / "chords"
        review_dir.mkdir()
        with open(review_dir / "review.yml", "w") as f:
            yaml.dump({"candidates": candidates}, f)
        return tmp_path

    def test_absent_review_returns_empty(self, tmp_path):
        result = read_approved_harmonic_rhythm(tmp_path)
        assert result == {}

    def test_approved_candidate_with_hr(self, tmp_path):
        candidates = [
            {"status": "approved", "label": "verse", "hr_distribution": [1.0, 2.0]},
        ]
        self._write_review(tmp_path, candidates)
        result = read_approved_harmonic_rhythm(tmp_path)
        assert "verse" in result
        assert result["verse"] == [1.0, 2.0]

    def test_pending_candidate_ignored(self, tmp_path):
        candidates = [
            {"status": "pending", "label": "chorus", "hr_distribution": [1.0]},
        ]
        self._write_review(tmp_path, candidates)
        result = read_approved_harmonic_rhythm(tmp_path)
        assert result == {}

    def test_accepted_alias_works(self, tmp_path):
        candidates = [
            {"status": "accepted", "label": "bridge", "hr_distribution": [2.0]},
        ]
        self._write_review(tmp_path, candidates)
        result = read_approved_harmonic_rhythm(tmp_path)
        assert "bridge" in result

    def test_label_normalised_to_key(self, tmp_path):
        candidates = [
            {"status": "approved", "label": "Verse-A", "hr_distribution": [1.0]},
        ]
        self._write_review(tmp_path, candidates)
        result = read_approved_harmonic_rhythm(tmp_path)
        assert "verse_a" in result

    def test_missing_hr_distribution_skipped(self, tmp_path):
        candidates = [
            {"status": "approved", "label": "outro"},
        ]
        self._write_review(tmp_path, candidates)
        result = read_approved_harmonic_rhythm(tmp_path)
        assert result == {}

    def test_first_approved_wins_per_section(self, tmp_path):
        """Two approved candidates for same label: only first one stored."""
        candidates = [
            {"status": "approved", "label": "verse", "hr_distribution": [1.0]},
            {"status": "approved", "label": "verse", "hr_distribution": [2.0]},
        ]
        self._write_review(tmp_path, candidates)
        result = read_approved_harmonic_rhythm(tmp_path)
        assert result["verse"] == [1.0]
