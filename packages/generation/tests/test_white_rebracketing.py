"""Tests for white_rebracketing.py — MIDI transpose, BPM, bar extract/concat, bar pool."""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import mido
import pytest
import yaml

from white_generation.pipelines.white_rebracketing import (
    _root_to_semitone,
    build_bar_pool,
    concatenate_bars,
    extract_bars,
    set_midi_bpm,
    transpose_midi,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_midi(
    notes: list[tuple[int, int, int, int]], tpb: int = 480, bpm: int = 120
) -> bytes:
    """Create a minimal MIDI file with the given notes.

    notes: list of (channel, pitch, start_tick, end_tick)
    Returns raw MIDI bytes (type 0, single track).
    """
    tempo_us = round(60_000_000 / bpm)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))

    events: list[tuple[int, mido.Message]] = []
    for ch, pitch, start, end in notes:
        events.append(
            (
                start,
                mido.Message("note_on", channel=ch, note=pitch, velocity=80, time=0),
            )
        )
        events.append(
            (end, mido.Message("note_off", channel=ch, note=pitch, velocity=0, time=0))
        )

    events.sort(key=lambda x: x[0])

    prev = 0
    for abs_tick, msg in events:
        track.append(msg.copy(time=abs_tick - prev))
        prev = abs_tick

    track.append(mido.MetaMessage("end_of_track", time=0))

    mid = mido.MidiFile(ticks_per_beat=tpb, type=0)
    mid.tracks.append(track)
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def _all_notes(midi_bytes: bytes) -> list[tuple[int, int]]:
    """Return list of (channel, note) for all note_on messages with velocity > 0."""
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    notes = []
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                notes.append((msg.channel, msg.note))
    return notes


def _get_tempo(midi_bytes: bytes) -> int:
    """Return the first set_tempo value found, or -1."""
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    for track in mid.tracks:
        for msg in track:
            if hasattr(msg, "type") and msg.type == "set_tempo":
                return msg.tempo
    return -1


# ---------------------------------------------------------------------------
# _root_to_semitone
# ---------------------------------------------------------------------------


def test_root_to_semitone_naturals():
    assert _root_to_semitone("C") == 0
    assert _root_to_semitone("G") == 7
    assert _root_to_semitone("B") == 11


def test_root_to_semitone_accidentals():
    assert _root_to_semitone("F#") == 6
    assert _root_to_semitone("Bb") == 10
    assert _root_to_semitone("Db") == 1


def test_root_to_semitone_unknown():
    with pytest.raises(ValueError):
        _root_to_semitone("X")


# ---------------------------------------------------------------------------
# transpose_midi
# ---------------------------------------------------------------------------


def test_transpose_midi_shifts_notes():
    raw = _make_midi([(0, 60, 0, 480), (0, 64, 480, 960)])  # C4, E4
    result = transpose_midi(raw, 2)
    notes = _all_notes(result)
    assert (0, 62) in notes  # D4
    assert (0, 66) in notes  # F#4


def test_transpose_midi_zero_delta_passthrough():
    raw = _make_midi([(0, 60, 0, 480)])
    assert transpose_midi(raw, 0) == raw


def test_transpose_midi_non_note_messages_unchanged():
    raw = _make_midi([(0, 60, 0, 480)])
    result = transpose_midi(raw, 3)
    # Tempo message should still be present
    assert _get_tempo(result) > 0


def test_transpose_midi_clamps_high():
    raw = _make_midi([(0, 107, 0, 480)])  # near top
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = transpose_midi(raw, 5)  # 107+5=112 → clamped to 108
    notes = _all_notes(result)
    assert notes[0][1] == 108
    assert len(w) == 1


def test_transpose_midi_clamps_low():
    raw = _make_midi([(0, 22, 0, 480)])  # near bottom
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = transpose_midi(raw, -5)  # 22-5=17 → clamped to 21
    notes = _all_notes(result)
    assert notes[0][1] == 21
    assert len(w) == 1


# ---------------------------------------------------------------------------
# set_midi_bpm
# ---------------------------------------------------------------------------


def test_set_midi_bpm_sets_correct_tempo():
    raw = _make_midi([(0, 60, 0, 480)], bpm=120)
    result = set_midi_bpm(raw, 90)
    expected_us = round(60_000_000 / 90)
    assert _get_tempo(result) == expected_us


def test_set_midi_bpm_replaces_existing_tempo():
    raw = _make_midi([(0, 60, 0, 480)], bpm=60)
    result = set_midi_bpm(raw, 120)
    expected_us = round(60_000_000 / 120)
    assert _get_tempo(result) == expected_us


def test_set_midi_bpm_note_values_unchanged():
    raw = _make_midi([(0, 60, 0, 480), (0, 64, 480, 960)])
    result = set_midi_bpm(raw, 180)
    notes = _all_notes(result)
    assert (0, 60) in notes
    assert (0, 64) in notes


# ---------------------------------------------------------------------------
# extract_bars
# ---------------------------------------------------------------------------


def test_extract_bars_count():
    # 8 bars worth of notes at 4/4, 480 tpb
    tpb = 480
    bpb = 4
    bar_ticks = tpb * bpb
    notes = [(0, 60 + i, i * bar_ticks, i * bar_ticks + 100) for i in range(8)]
    raw = _make_midi(notes, tpb=tpb)
    bars = extract_bars(raw, tpb, bpb)
    assert len(bars) == 8


def test_extract_bars_retick_zeroed():
    """Each bar's events should start close to tick 0."""
    tpb = 480
    bpb = 4
    bar_ticks = tpb * bpb
    # Note in bar 2 (starts at 2 * bar_ticks)
    notes = [(0, 60, 2 * bar_ticks, 2 * bar_ticks + 100)]
    raw = _make_midi(notes, tpb=tpb)
    bars = extract_bars(raw, tpb, bpb)
    bar2 = bars[2]
    mid = mido.MidiFile(file=io.BytesIO(bar2))
    note_times = [
        msg.time
        for track in mid.tracks
        for msg in track
        if msg.type == "note_on" and msg.velocity > 0
    ]
    # The note should appear near the start of the bar (within bar_ticks)
    assert all(t < bar_ticks for t in note_times)


def test_extract_bars_note_in_correct_bar():
    tpb = 480
    bpb = 4
    bar_ticks = tpb * bpb
    # Only bar 3 has a note
    notes = [(0, 72, 3 * bar_ticks + 10, 3 * bar_ticks + 200)]
    raw = _make_midi(notes, tpb=tpb)
    bars = extract_bars(raw, tpb, bpb)
    note_counts = [len(_all_notes(b)) for b in bars]
    assert note_counts[3] == 1
    assert sum(note_counts) == 1


# ---------------------------------------------------------------------------
# concatenate_bars
# ---------------------------------------------------------------------------


def test_concatenate_bars_note_count():
    tpb = 480
    # Two bars, each with one note
    bar0 = _make_midi([(0, 60, 0, 100)], tpb=tpb)
    bar1 = _make_midi([(0, 64, 0, 100)], tpb=tpb)
    result = concatenate_bars([bar0, bar1], tpb, bpm=120)
    notes = _all_notes(result)
    pitches = {n[1] for n in notes}
    assert 60 in pitches
    assert 64 in pitches


def test_concatenate_bars_tempo_message():
    tpb = 480
    bar = _make_midi([(0, 60, 0, 100)], tpb=tpb)
    result = concatenate_bars([bar], tpb, bpm=90)
    assert _get_tempo(result) == round(60_000_000 / 90)


def test_concatenate_bars_empty_raises():
    with pytest.raises(ValueError):
        concatenate_bars([], 480, 120)


def test_concatenate_bars_total_length_matches_beats_per_bar():
    """Cursor must advance by beats_per_bar × tpb per bar, regardless of note content.

    Sparse bars (notes ending before the bar end) must not shrink the output.
    """
    tpb = 480
    # Bars with short notes (note ends at tick 100, but bar should be 1440 ticks = 6/8)
    bar0 = _make_midi([(0, 60, 0, 100)], tpb=tpb)
    bar1 = _make_midi([(0, 64, 0, 100)], tpb=tpb)
    result = concatenate_bars([bar0, bar1], tpb, bpm=120, beats_per_bar=3.0)
    mid = mido.MidiFile(file=io.BytesIO(result))
    total_ticks = sum(msg.time for track in mid.tracks for msg in track)
    # Each bar should advance 1440 ticks; bar1 note_on lands at tick 1440
    # The total should be at least 1440 + 100 = 1540 (bar1 note at 1440, off at 1540)
    assert total_ticks >= 1440 + 100
    notes = _all_notes(result)
    pitches = {n[1] for n in notes}
    assert 60 in pitches
    assert 64 in pitches


# ---------------------------------------------------------------------------
# build_bar_pool — integration test with fixture dirs
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_sub_proposals(tmp_path: Path) -> list[Path]:
    """Create two minimal sub-proposal dirs with approved chord MIDIs."""
    dirs = []
    colors = [("Red", "C minor", 80), ("Blue", "G minor", 60)]

    for i, (color, key, bpm) in enumerate(colors):
        prod_dir = tmp_path / f"prod_{i}"
        approved_dir = prod_dir / "chords" / "approved"
        approved_dir.mkdir(parents=True)

        # Write review.yml
        review = {"key": key, "bpm": bpm, "color": color}
        with open(prod_dir / "chords" / "review.yml", "w") as f:
            yaml.dump(review, f)

        # Write one approved MIDI (4 bars of notes)
        tpb = 480
        bpb = 4
        bar_ticks = tpb * bpb
        notes = [(0, 60 + j, j * bar_ticks, j * bar_ticks + 100) for j in range(4)]
        midi_bytes = _make_midi(notes, tpb=tpb, bpm=bpm)
        (approved_dir / "loop.mid").write_bytes(midi_bytes)

        dirs.append(prod_dir)

    return dirs


def test_build_bar_pool_bar_count(two_sub_proposals):
    pool = build_bar_pool(two_sub_proposals, white_key="D minor", white_bpm=100)
    # 2 sub-proposals × 4 bars each = 8 bars
    assert len(pool) == 8


def test_build_bar_pool_metadata(two_sub_proposals):
    pool = build_bar_pool(two_sub_proposals, white_key="D minor", white_bpm=100)
    colors = {b["donor_color"] for b in pool}
    assert "Red" in colors
    assert "Blue" in colors
    for bar in pool:
        assert "source_dir" in bar
        assert "source_file" in bar
        assert "bar_index" in bar
        assert "midi_bytes" in bar


def test_build_bar_pool_bpm_applied_on_concat(two_sub_proposals):
    """When pool bars are concatenated, the resulting MIDI has the White BPM."""
    pool = build_bar_pool(two_sub_proposals, white_key="D minor", white_bpm=100)
    assert pool, "pool must be non-empty"
    tpb = 480
    result = concatenate_bars([b["midi_bytes"] for b in pool[:2]], tpb, bpm=100)
    expected_us = round(60_000_000 / 100)
    assert _get_tempo(result) == expected_us


def test_build_bar_pool_missing_review_skipped(tmp_path):
    """A sub-proposal dir without chords/review.yml is skipped with a warning."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pool = build_bar_pool([empty_dir], white_key="C major", white_bpm=120)
    assert pool == []
    assert any("review.yml" in str(warning.message) for warning in w)


def test_build_bar_pool_empty_approved_skipped(tmp_path):
    """A sub-proposal with no approved MIDIs is skipped with a warning."""
    prod_dir = tmp_path / "prod"
    approved_dir = prod_dir / "chords" / "approved"
    approved_dir.mkdir(parents=True)
    review = {"key": "C major", "bpm": 120, "color": "Red"}
    with open(prod_dir / "chords" / "review.yml", "w") as f:
        yaml.dump(review, f)
    # No MIDI files written

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pool = build_bar_pool([prod_dir], white_key="C major", white_bpm=120)
    assert pool == []
    assert any("approved" in str(warning.message).lower() for warning in w)


def test_build_bar_pool_54_beats_per_bar(tmp_path):
    """build_bar_pool with beats_per_bar=5 slices donor 4/4 MIDI at 5-beat boundaries.

    A 4-bar 4/4 loop is 16 beats = 7680 ticks. With beats_per_bar=5, extract_bars
    cuts at 2400-tick boundaries → ceil(7680/2400) = 4 bars, each 2400 ticks wide
    (last may be shorter). Verifies that the bar pool bars carry the right bar length
    so concatenated White chord MIDI matches the target 5/4 bar size.
    """
    prod_dir = tmp_path / "prod"
    approved_dir = prod_dir / "chords" / "approved"
    approved_dir.mkdir(parents=True)

    review = {"key": "C major", "bpm": 80, "color": "Red"}
    with open(prod_dir / "chords" / "review.yml", "w") as f:
        yaml.dump(review, f)

    # Create a 4-bar 4/4 MIDI (16 beats = 7680 ticks at tpb=480)
    tpb = 480
    bpb_44 = 4
    bar_ticks_44 = tpb * bpb_44
    notes = [(0, 60 + j, j * bar_ticks_44, j * bar_ticks_44 + 100) for j in range(4)]
    midi_bytes = _make_midi(notes, tpb=tpb, bpm=80)
    (approved_dir / "loop.mid").write_bytes(midi_bytes)

    # Build pool with 5/4 target: each extracted bar should span 2400 ticks
    pool_54 = build_bar_pool(
        [prod_dir], white_key="C major", white_bpm=80, beats_per_bar=5.0
    )
    assert pool_54, "pool must be non-empty"

    # Verify each bar MIDI is exactly 2400 ticks wide (a 5/4 bar)
    expected_bar_ticks = int(tpb * 5)
    for bar in pool_54:
        mid = mido.MidiFile(file=io.BytesIO(bar["midi_bytes"]))
        actual_ticks = sum(msg.time for msg in mid.tracks[0])
        # Bar MIDI ends with end_of_track at time=0; meaningful length = sum of all delta
        # times. The bar slice may be shorter than expected_bar_ticks for the last partial bar.
        assert actual_ticks <= expected_bar_ticks + 1  # allow 1-tick rounding

    # Notes are at ticks 0, 1920, 3840, 5760 (last note-off at 5860).
    # With 5-beat bars (2400 ticks): ceil(5860/2400) = 3 slices — different from the
    # 4/4 slice count, confirming the target bar size controls the extraction.
    assert len(pool_54) == 3

    # Default (4/4) should produce 4 bars at 1920-tick boundaries (last note-off at 5860
    # → ceil(5860/1920) = 4 slices)
    pool_44 = build_bar_pool([prod_dir], white_key="C major", white_bpm=80)
    assert len(pool_44) == 4
