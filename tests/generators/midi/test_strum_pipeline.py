"""Tests for the strum/rhythm generation pipeline."""

import io

import mido
import pytest
import yaml


# ---------------------------------------------------------------------------
# 1. Pattern template validation
# ---------------------------------------------------------------------------


class TestStrumPatterns:

    def test_all_patterns_have_matching_onsets_durations(self):
        from app.generators.midi.strum_patterns import ALL_PATTERNS

        for p in ALL_PATTERNS:
            assert len(p.onsets) == len(
                p.durations
            ), f"{p.name} ({p.time_sig}): onsets ({len(p.onsets)}) != durations ({len(p.durations)})"

    def test_all_patterns_onsets_within_bar(self):
        from app.generators.midi.strum_patterns import ALL_PATTERNS

        for p in ALL_PATTERNS:
            bar_length = p.bar_length_beats()
            for onset in p.onsets:
                assert (
                    0 <= onset < bar_length
                ), f"{p.name} ({p.time_sig}): onset {onset} outside bar length {bar_length}"

    def test_all_patterns_durations_positive(self):
        from app.generators.midi.strum_patterns import ALL_PATTERNS

        for p in ALL_PATTERNS:
            for dur in p.durations:
                assert dur > 0, f"{p.name}: duration {dur} not positive"

    def test_non_arpeggio_durations_fill_bar(self):
        from app.generators.midi.strum_patterns import ALL_PATTERNS

        for p in ALL_PATTERNS:
            if not p.is_arpeggio:
                total = sum(p.durations)
                bar_length = p.bar_length_beats()
                assert (
                    abs(total - bar_length) < 0.01
                ), f"{p.name} ({p.time_sig}): durations sum to {total}, bar is {bar_length}"

    def test_4_4_patterns_exist(self):
        from app.generators.midi.strum_patterns import ALL_PATTERNS

        names_4_4 = {p.name for p in ALL_PATTERNS if p.time_sig == (4, 4)}
        for expected in [
            "whole",
            "half",
            "quarter",
            "eighth",
            "push",
            "arp_up",
            "arp_down",
        ]:
            assert expected in names_4_4, f"Missing 4/4 pattern: {expected}"

    def test_7_8_patterns_exist(self):
        from app.generators.midi.strum_patterns import ALL_PATTERNS

        names_7_8 = {p.name for p in ALL_PATTERNS if p.time_sig == (7, 8)}
        for expected in ["whole", "grouped_322", "grouped_223", "eighth"]:
            assert expected in names_7_8, f"Missing 7/8 pattern: {expected}"

    def test_fallback_patterns(self):
        from app.generators.midi.strum_patterns import make_fallback_patterns

        patterns = make_fallback_patterns((5, 4))
        assert len(patterns) >= 2
        names = {p.name for p in patterns}
        assert "whole" in names
        assert "beat" in names
        # Beat pattern should have 5 onsets
        beat = [p for p in patterns if p.name == "beat"][0]
        assert len(beat.onsets) == 5

    def test_get_patterns_for_time_sig(self):
        from app.generators.midi.strum_patterns import get_patterns_for_time_sig

        patterns_4_4 = get_patterns_for_time_sig((4, 4))
        assert len(patterns_4_4) >= 7

        patterns_7_8 = get_patterns_for_time_sig((7, 8))
        assert len(patterns_7_8) >= 4

        # Unknown time sig should get fallback
        patterns_11_8 = get_patterns_for_time_sig((11, 8))
        assert len(patterns_11_8) >= 2

    def test_get_patterns_with_filter(self):
        from app.generators.midi.strum_patterns import get_patterns_for_time_sig

        patterns = get_patterns_for_time_sig((4, 4), filter_names=["quarter", "eighth"])
        names = {p.name for p in patterns}
        assert names == {"quarter", "eighth"}

    def test_bar_length_beats(self):
        from app.generators.midi.strum_patterns import StrumPattern

        p = StrumPattern(name="t", time_sig=(4, 4), description="t")
        assert p.bar_length_beats() == 4.0

        p = StrumPattern(name="t", time_sig=(7, 8), description="t")
        assert p.bar_length_beats() == 3.5


# ---------------------------------------------------------------------------
# 2. MIDI parsing
# ---------------------------------------------------------------------------


class TestMidiParsing:

    def _make_chord_midi(self, chords, bpm=120, ticks_per_beat=480):
        """Create a chord MIDI file in the same format as chord_pipeline."""
        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        track = mido.MidiTrack()
        mid.tracks.append(track)

        tempo = mido.bpm2tempo(bpm)
        track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

        bar_ticks = ticks_per_beat * 4

        for notes in chords:
            for note in notes:
                track.append(mido.Message("note_on", note=note, velocity=80, time=0))
            for i, note in enumerate(notes):
                track.append(
                    mido.Message(
                        "note_off",
                        note=note,
                        velocity=0,
                        time=bar_ticks if i == 0 else 0,
                    )
                )

        track.append(mido.MetaMessage("end_of_track", time=0))

        buf = io.BytesIO()
        mid.save(file=buf)
        return buf.getvalue()

    def test_parse_single_chord(self, tmp_path):
        from app.generators.midi.strum_pipeline import parse_chord_voicings

        midi_bytes = self._make_chord_midi([[60, 64, 67]])
        path = tmp_path / "test.mid"
        path.write_bytes(midi_bytes)

        voicings = parse_chord_voicings(path)
        assert len(voicings) == 1
        assert sorted(voicings[0]["notes"]) == [60, 64, 67]

    def test_parse_multiple_chords(self, tmp_path):
        from app.generators.midi.strum_pipeline import parse_chord_voicings

        midi_bytes = self._make_chord_midi([[60, 64, 67], [65, 69, 72], [67, 71, 74]])
        path = tmp_path / "test.mid"
        path.write_bytes(midi_bytes)

        voicings = parse_chord_voicings(path)
        assert len(voicings) == 3
        assert sorted(voicings[0]["notes"]) == [60, 64, 67]
        assert sorted(voicings[1]["notes"]) == [65, 69, 72]
        assert sorted(voicings[2]["notes"]) == [67, 71, 74]

    def test_parse_preserves_velocity(self, tmp_path):
        from app.generators.midi.strum_pipeline import parse_chord_voicings

        midi_bytes = self._make_chord_midi([[60, 64, 67]])
        path = tmp_path / "test.mid"
        path.write_bytes(midi_bytes)

        voicings = parse_chord_voicings(path)
        assert voicings[0]["velocity"] == 80


# ---------------------------------------------------------------------------
# 3. Pattern application
# ---------------------------------------------------------------------------


class TestPatternApplication:

    def test_block_pattern_all_notes(self):
        from app.generators.midi.strum_patterns import StrumPattern
        from app.generators.midi.strum_pipeline import apply_strum_pattern

        pattern = StrumPattern(
            name="quarter",
            time_sig=(4, 4),
            description="test",
            onsets=[0, 1, 2, 3],
            durations=[1, 1, 1, 1],
        )
        events = apply_strum_pattern([60, 64, 67], pattern)
        assert len(events) == 4
        for ev in events:
            assert sorted(ev["notes"]) == [60, 64, 67]

    def test_arpeggio_distributes_notes(self):
        from app.generators.midi.strum_patterns import StrumPattern
        from app.generators.midi.strum_pipeline import apply_strum_pattern

        pattern = StrumPattern(
            name="arp_up",
            time_sig=(4, 4),
            description="test",
            is_arpeggio=True,
            arp_direction="up",
            onsets=[0, 0.5, 1, 1.5],
            durations=[0.5, 0.5, 0.5, 0.5],
        )
        events = apply_strum_pattern([60, 64, 67], pattern)
        assert len(events) == 4
        # Each event should have exactly 1 note
        for ev in events:
            assert len(ev["notes"]) == 1
        # Up direction: should cycle low→high
        notes = [ev["notes"][0] for ev in events]
        assert notes == [60, 64, 67, 60]  # cycles back

    def test_arpeggio_down(self):
        from app.generators.midi.strum_patterns import StrumPattern
        from app.generators.midi.strum_pipeline import apply_strum_pattern

        pattern = StrumPattern(
            name="arp_down",
            time_sig=(4, 4),
            description="test",
            is_arpeggio=True,
            arp_direction="down",
            onsets=[0, 0.5, 1],
            durations=[0.5, 0.5, 0.5],
        )
        events = apply_strum_pattern([60, 64, 67], pattern)
        notes = [ev["notes"][0] for ev in events]
        assert notes == [67, 64, 60]  # high→low

    def test_strum_to_midi_bytes_valid(self):
        from app.generators.midi.strum_patterns import PATTERNS_4_4
        from app.generators.midi.strum_pipeline import strum_to_midi_bytes

        pattern = PATTERNS_4_4[2]  # quarter
        chords = [[60, 64, 67], [65, 69, 72]]
        midi_bytes = strum_to_midi_bytes(chords, pattern, bpm=120)

        assert len(midi_bytes) > 0
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
        assert len(mid.tracks) >= 1

    def test_strum_correct_tempo(self):
        from app.generators.midi.strum_patterns import PATTERNS_4_4
        from app.generators.midi.strum_pipeline import strum_to_midi_bytes

        pattern = PATTERNS_4_4[0]  # whole
        midi_bytes = strum_to_midi_bytes([[60, 64, 67]], pattern, bpm=84)
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

        tempo_msgs = [msg for msg in mid.tracks[0] if msg.type == "set_tempo"]
        assert len(tempo_msgs) == 1
        assert mido.tempo2bpm(tempo_msgs[0].tempo) == pytest.approx(84, abs=0.1)

    def test_quarter_pattern_note_count(self):
        from app.generators.midi.strum_patterns import PATTERNS_4_4
        from app.generators.midi.strum_pipeline import strum_to_midi_bytes

        quarter = [p for p in PATTERNS_4_4 if p.name == "quarter"][0]
        chord = [60, 64, 67]  # 3 notes
        midi_bytes = strum_to_midi_bytes([chord], quarter, bpm=120)
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

        note_ons = [
            msg for msg in mid.tracks[0] if msg.type == "note_on" and msg.velocity > 0
        ]
        # Quarter pattern: 4 onsets × 3 notes = 12 note-ons
        assert len(note_ons) == 12

    def test_arpeggio_pattern_single_notes(self):
        from app.generators.midi.strum_patterns import PATTERNS_4_4
        from app.generators.midi.strum_pipeline import strum_to_midi_bytes

        arp_up = [p for p in PATTERNS_4_4 if p.name == "arp_up"][0]
        chord = [60, 64, 67]  # 3 notes
        midi_bytes = strum_to_midi_bytes([chord], arp_up, bpm=120)
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

        note_ons = [
            msg for msg in mid.tracks[0] if msg.type == "note_on" and msg.velocity > 0
        ]
        # Arp: 16 onsets × 1 note each = 16
        assert len(note_ons) == 16


# ---------------------------------------------------------------------------
# 4. Integration test
# ---------------------------------------------------------------------------


class TestStrumPipelineIntegration:

    def _setup_production_dir(self, tmp_path):
        """Create a mock production directory with approved chords."""
        # Create chord review
        chords_dir = tmp_path / "chords"
        chords_dir.mkdir(parents=True)
        approved_dir = chords_dir / "approved"
        approved_dir.mkdir()

        review = {
            "bpm": 120,
            "time_sig": "4/4",
            "candidates": [
                {
                    "id": "chord_001",
                    "label": "Verse",
                    "status": "approved",
                    "chords": [
                        {"notes": ["C3", "E3", "G3"]},
                        {"notes": ["F3", "A3", "C4"]},
                    ],
                },
            ],
        }
        with open(chords_dir / "review.yml", "w") as f:
            yaml.dump(review, f)

        # Create approved chord MIDI
        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))

        for notes in [[60, 64, 67], [65, 69, 72]]:
            for n in notes:
                track.append(mido.Message("note_on", note=n, velocity=80, time=0))
            for i, n in enumerate(notes):
                track.append(
                    mido.Message(
                        "note_off", note=n, velocity=0, time=1920 if i == 0 else 0
                    )
                )

        track.append(mido.MetaMessage("end_of_track", time=0))
        mid.save(str(approved_dir / "verse.mid"))

        return tmp_path

    def test_per_chord_mode(self, tmp_path):
        from app.generators.midi.strum_pipeline import run_strum_pipeline

        prod_dir = self._setup_production_dir(tmp_path)
        result = run_strum_pipeline(str(prod_dir), mode="per-chord")

        assert len(result) > 0
        # Should have one candidate per pattern for "verse"
        assert all(c["source_chord"] == "verse" for c in result)

        # Verify files written
        strums_dir = prod_dir / "strums"
        assert strums_dir.exists()
        assert (strums_dir / "review.yml").exists()

        # Verify MIDI files
        for cand in result:
            path = strums_dir / "candidates" / f"{cand['id']}.mid"
            assert path.exists(), f"Missing: {path}"

    def test_progression_mode(self, tmp_path):
        from app.generators.midi.strum_pipeline import run_strum_pipeline

        prod_dir = self._setup_production_dir(tmp_path)
        result = run_strum_pipeline(str(prod_dir), mode="progression")

        assert len(result) > 0
        assert all(c["mode"] == "progression" for c in result)
        assert all(c["id"].startswith("progression_") for c in result)

    def test_review_yaml_structure(self, tmp_path):
        from app.generators.midi.strum_pipeline import run_strum_pipeline

        prod_dir = self._setup_production_dir(tmp_path)
        run_strum_pipeline(str(prod_dir), mode="per-chord")

        with open(prod_dir / "strums" / "review.yml") as f:
            review = yaml.safe_load(f)

        assert review["pipeline"] == "strum-generation"
        assert review["bpm"] == 120
        assert len(review["candidates"]) > 0

        for cand in review["candidates"]:
            assert "id" in cand
            assert "midi_file" in cand
            assert "source_chord" in cand
            assert "pattern" in cand
            assert cand["status"] == "pending"
            assert cand["label"] is None

    def test_filter_patterns(self, tmp_path):
        from app.generators.midi.strum_pipeline import run_strum_pipeline

        prod_dir = self._setup_production_dir(tmp_path)
        result = run_strum_pipeline(
            str(prod_dir), mode="per-chord", filter_patterns=["quarter", "eighth"]
        )

        patterns_used = {c["pattern_name"] for c in result}
        assert patterns_used == {"quarter", "eighth"}
