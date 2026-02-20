"""Tests for harmonic rhythm generation pipeline."""

import io

import mido
import pytest

from app.generators.midi.harmonic_rhythm import (
    GRID_UNIT,
    MAX_CANDIDATES,
    accents_to_halfbar_mask,
    default_accent_mask,
    distribution_to_midi_bytes,
    drum_alignment_score,
    enumerate_distributions,
    extract_drum_accents,
)


# ===========================================================================
# Distribution enumeration
# ===========================================================================


class TestDistributionEnumeration:
    def test_empty_chords(self):
        assert enumerate_distributions(0) == []

    def test_single_chord(self):
        dists = enumerate_distributions(1)
        assert len(dists) > 0
        # All should be multiples of 0.5
        for dist in dists:
            assert len(dist) == 1
            assert dist[0] % GRID_UNIT == 0
            assert dist[0] >= 0.5

    def test_uniform_always_included(self):
        for n in [2, 3, 4]:
            dists = enumerate_distributions(n)
            uniform = [1.0] * n
            assert uniform in dists, f"Uniform baseline missing for n={n}"

    def test_minimum_duration(self):
        dists = enumerate_distributions(4)
        for dist in dists:
            for d in dist:
                assert d >= 0.5, f"Chord duration {d} below minimum 0.5"

    def test_total_within_bounds(self):
        n = 4
        dists = enumerate_distributions(n)
        min_total = n * 0.5
        max_total = n * 2.0
        for dist in dists:
            total = sum(dist)
            assert total >= min_total - 0.01, f"Total {total} below min {min_total}"
            assert total <= max_total + 0.01, f"Total {total} above max {max_total}"

    def test_grid_alignment(self):
        dists = enumerate_distributions(3)
        for dist in dists:
            for d in dist:
                # Should be a multiple of 0.5
                assert abs(d % GRID_UNIT) < 0.01, f"Duration {d} not on half-bar grid"

    def test_cap_at_max_candidates(self):
        # With many chords, should be capped
        dists = enumerate_distributions(4, seed=42)
        assert len(dists) <= MAX_CANDIDATES

    def test_deterministic_with_seed(self):
        d1 = enumerate_distributions(4, seed=123)
        d2 = enumerate_distributions(4, seed=123)
        assert d1 == d2

    def test_seed_has_no_effect(self):
        # All distributions are fully enumerated; seed is a no-op
        d1 = enumerate_distributions(4, seed=1)
        d2 = enumerate_distributions(4, seed=999)
        assert d1 == d2

    def test_variety_exists(self):
        dists = enumerate_distributions(4)
        # Should have more than just the uniform
        assert len(dists) > 1
        # Should have different total lengths
        totals = set(round(sum(d), 1) for d in dists)
        assert len(totals) > 1, "All distributions have the same total length"


# ===========================================================================
# Drum accent extraction
# ===========================================================================


def _make_drum_midi(hits: list[tuple[int, int]], tpb: int = 480) -> str:
    """Create a temporary drum MIDI file and return its path.

    hits: list of (tick_offset, velocity) for kick drum (note 36).
    """
    import tempfile

    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))

    prev = 0
    for tick, vel in sorted(hits):
        delta = tick - prev
        track.append(
            mido.Message("note_on", note=36, velocity=vel, time=delta, channel=9)
        )
        track.append(mido.Message("note_off", note=36, velocity=0, time=60, channel=9))
        prev = tick + 60

    track.append(mido.MetaMessage("end_of_track", time=0))

    f = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    mid.save(f.name)
    return f.name


class TestDrumAccentExtraction:
    def test_accent_above_threshold(self):
        # Kick at beat 0 (tick 0) with accent velocity
        path = _make_drum_midi([(0, 120)])
        accents = extract_drum_accents(path, time_sig=(4, 4))
        assert 0.0 in accents

    def test_below_threshold_excluded(self):
        # Only ghost hit at beat 1 (tick 480) — velocity 45 < threshold 80
        path = _make_drum_midi([(480, 45)])
        accents = extract_drum_accents(path, time_sig=(4, 4))
        # Should still have 0.0 (always included) but not 1.0
        assert 0.0 in accents
        assert 1.0 not in accents

    def test_multiple_accents(self):
        # Kick on beat 0 and beat 2 (ticks 0, 960)
        path = _make_drum_midi([(0, 120), (960, 110)])
        accents = extract_drum_accents(path, time_sig=(4, 4))
        assert 0.0 in accents
        assert 2.0 in accents

    def test_always_includes_zero(self):
        # No accented hits at all
        path = _make_drum_midi([(480, 45)])
        accents = extract_drum_accents(path, time_sig=(4, 4))
        assert 0.0 in accents

    def test_7_8_accents(self):
        # 7/8: bar = 3.5 beats = 1680 ticks at 480 tpb
        # Accent at beat 0 and beat 1.5 (tick 720)
        path = _make_drum_midi([(0, 120), (720, 110)])
        accents = extract_drum_accents(path, time_sig=(7, 8))
        assert 0.0 in accents
        assert 1.5 in accents


class TestAccentToHalfbarMask:
    def test_4_4_both_halves_strong(self):
        # Accents at beat 0 and beat 2 — both half-bar boundaries
        accents = [0.0, 2.0]
        mask = accents_to_halfbar_mask(accents, time_sig=(4, 4))
        assert 0.0 in mask
        assert 2.0 in mask

    def test_4_4_only_downbeat(self):
        # Accent only at beat 0
        accents = [0.0]
        mask = accents_to_halfbar_mask(accents, time_sig=(4, 4))
        assert 0.0 in mask
        assert 2.0 not in mask

    def test_tolerance(self):
        # Accent at 1.9 — within 0.25 of half-bar boundary at 2.0
        accents = [0.0, 1.9]
        mask = accents_to_halfbar_mask(accents, time_sig=(4, 4))
        assert 2.0 in mask

    def test_7_8_mask(self):
        # 7/8: bar = 3.5 beats, half-bar = 1.75
        # Accent at 0.0 and 1.5 (within tolerance of 1.75)
        accents = [0.0, 1.5]
        mask = accents_to_halfbar_mask(accents, time_sig=(7, 8))
        assert 0.0 in mask
        assert 1.75 in mask

    def test_default_mask(self):
        mask = default_accent_mask((4, 4))
        assert mask == [0.0]


# ===========================================================================
# Drum alignment scoring
# ===========================================================================


class TestDrumAlignmentScoring:
    def test_all_aligned(self):
        # 4 chords, each 1 bar, accents on every half-bar
        dist = [1.0, 1.0, 1.0, 1.0]
        mask = [0.0, 2.0]  # both half-bar positions strong
        score = drum_alignment_score(dist, mask, time_sig=(4, 4))
        assert score == 1.0

    def test_only_first_aligned(self):
        # 4 chords at half-bar boundaries, but mask only has 0.0
        dist = [0.5, 0.5, 0.5, 0.5]
        mask = [0.0]
        score = drum_alignment_score(dist, mask, time_sig=(4, 4))
        # Onsets at beats 0, 2, 4, 6 — only 0 aligns (wraps: 0, 2, 0, 2)
        # Actually: 0.5 bars = 2 beats, so onsets at 0, 2, 4, 6
        # pos_in_bar: 0, 2, 0, 2 — mask only has 0.0
        # So 2 out of 4 align (the ones at pos 0)
        assert score == pytest.approx(0.5)

    def test_empty_distribution(self):
        assert drum_alignment_score([], [0.0], time_sig=(4, 4)) == 0.0

    def test_tiling_across_bars(self):
        # 2 chords, each 2 bars — onsets at beat 0 and beat 8
        # pos_in_bar: 0 and 0 — both align
        dist = [2.0, 2.0]
        mask = [0.0]
        score = drum_alignment_score(dist, mask, time_sig=(4, 4))
        assert score == 1.0

    def test_misaligned_onset(self):
        # 2 chords: 1.5 bars + 0.5 bars
        # In 4/4: onsets at beat 0 and beat 6 (1.5 * 4)
        # pos_in_bar of 6: 6 % 4 = 2.0
        # mask = [0.0] — beat 2 not in mask
        dist = [1.5, 0.5]
        mask = [0.0]
        score = drum_alignment_score(dist, mask, time_sig=(4, 4))
        assert score == pytest.approx(0.5)  # only first chord at 0 aligns


# ===========================================================================
# MIDI generation
# ===========================================================================


class TestDistributionMidi:
    def test_valid_midi(self):
        chords = [[60, 64, 67], [65, 69, 72]]
        dist = [1.0, 1.5]
        midi_bytes = distribution_to_midi_bytes(chords, dist, bpm=120)
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
        assert len(mid.tracks) == 1

    def test_correct_tempo(self):
        chords = [[60]]
        dist = [1.0]
        midi_bytes = distribution_to_midi_bytes(chords, dist, bpm=84)
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
        tempo_msgs = [m for m in mid.tracks[0] if m.type == "set_tempo"]
        assert len(tempo_msgs) == 1
        assert tempo_msgs[0].tempo == mido.bpm2tempo(84)

    def test_notes_preserved(self):
        chords = [[60, 64, 67]]
        dist = [1.0]
        midi_bytes = distribution_to_midi_bytes(chords, dist, bpm=120)
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
        note_ons = [m for m in mid.tracks[0] if m.type == "note_on"]
        played_notes = sorted(m.note for m in note_ons)
        assert played_notes == [60, 64, 67]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            distribution_to_midi_bytes([[60]], [1.0, 2.0])

    def test_variable_durations(self):
        chords = [[60], [67]]
        dist = [1.5, 0.5]
        midi_bytes = distribution_to_midi_bytes(
            chords, dist, bpm=120, ticks_per_beat=480, time_sig=(4, 4)
        )
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
        # First chord at tick 0, second at tick 1.5 * 4 * 480 = 2880
        # Compute absolute ticks
        abs_tick = 0
        onsets = []
        for msg in mid.tracks[0]:
            abs_tick += msg.time
            if msg.type == "note_on":
                onsets.append((msg.note, abs_tick))
        assert onsets[0] == (60, 0)
        assert onsets[1] == (67, 2880)

    def test_7_8_duration(self):
        chords = [[60]]
        dist = [1.0]
        midi_bytes = distribution_to_midi_bytes(
            chords, dist, bpm=84, ticks_per_beat=480, time_sig=(7, 8)
        )
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
        # 7/8 bar = 3.5 beats = 1680 ticks
        # Note-off should be at tick 1680
        abs_tick = 0
        for msg in mid.tracks[0]:
            abs_tick += msg.time
            if msg.type == "note_off":
                assert abs_tick == 1680
                break


# TestHarmonicRhythmIntegration removed — harmonic_rhythm_pipeline.py deleted in
# collapse-chord-primitive-phases refactor. HR + strum are now baked into chord candidates.
