"""Tests for melody_auto_split module."""

import io
from pathlib import Path

import mido

from white_generation.pipelines.melody_auto_split import (
    Note,
    assign_syllables_to_notes,
    auto_split_melody,
    split_note,
    syllabify,
)

TICKS = 480


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_melody_midi(
    notes: list[tuple[int, int, int]],
    ticks_per_beat: int = TICKS,
) -> bytes:
    """Build a MIDI file from (start_tick, pitch, duration_ticks) tuples."""
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))

    events: list[tuple[int, int, int, bool]] = []
    for start, pitch, dur in notes:
        events.append((start, pitch, 80, True))
        events.append((start + dur, pitch, 0, False))
    events.sort(key=lambda e: (e[0], not e[3]))

    prev = 0
    for abs_tick, pitch, vel, is_on in events:
        msg_type = "note_on" if is_on else "note_off"
        track.append(
            mido.Message(msg_type, note=pitch, velocity=vel, time=abs_tick - prev)
        )
        prev = abs_tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def count_note_ons(midi_bytes: bytes) -> int:
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    return sum(
        1
        for track in mid.tracks
        for msg in track
        if msg.type == "note_on" and msg.velocity > 0
    )


def pitches_from_midi(midi_bytes: bytes) -> list[int]:
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    return [
        msg.note
        for track in mid.tracks
        for msg in track
        if msg.type == "note_on" and msg.velocity > 0
    ]


# ---------------------------------------------------------------------------
# syllabify
# ---------------------------------------------------------------------------


def test_syllabify_single_syllable():
    assert syllabify("cat") == ["cat"]


def test_syllabify_multi_syllable():
    result = syllabify("beautiful")
    assert len(result) > 1
    assert "".join(result).lower() == "beautiful"


def test_syllabify_strips_punctuation():
    result = syllabify("hello,")
    assert all("," not in s for s in result)


def test_syllabify_fallback_non_alpha():
    result = syllabify("42")
    assert result == ["42"]


def test_syllabify_empty():
    assert syllabify("") == []


# ---------------------------------------------------------------------------
# assign_syllables_to_notes
# ---------------------------------------------------------------------------

NOTE = Note(start_tick=0, pitch=60, velocity=80, duration_ticks=TICKS, channel=0)


def _notes(n: int) -> list[Note]:
    return [
        Note(
            start_tick=i * TICKS, pitch=60, velocity=80, duration_ticks=TICKS, channel=0
        )
        for i in range(n)
    ]


def test_assign_equal_count():
    pairs = assign_syllables_to_notes(_notes(3), ["one", "two", "three"])
    assert [s for _, s in pairs] == ["one", "two", "three"]


def test_assign_more_notes_than_syllables():
    pairs = assign_syllables_to_notes(_notes(4), ["a", "b"])
    syllables = [s for _, s in pairs]
    assert syllables[:2] == ["a", "b"]
    assert syllables[2] == ""
    assert syllables[3] == ""


def test_assign_more_syllables_than_notes():
    pairs = assign_syllables_to_notes(_notes(2), ["a", "b", "c", "d"])
    assert [s for _, s in pairs] == ["a", "b"]


def test_assign_empty_notes():
    assert assign_syllables_to_notes([], ["a"]) == []


# ---------------------------------------------------------------------------
# split_note
# ---------------------------------------------------------------------------


def test_split_even():
    note = Note(start_tick=0, pitch=60, velocity=80, duration_ticks=960, channel=0)
    parts = split_note(note, 2, TICKS)
    assert len(parts) == 2
    assert parts[0].duration_ticks == 480
    assert parts[1].duration_ticks == 480
    assert parts[0].start_tick == 0
    assert parts[1].start_tick == 480


def test_split_odd_remainder_absorbed_in_last():
    note = Note(start_tick=0, pitch=60, velocity=80, duration_ticks=10, channel=0)
    parts = split_note(note, 3, TICKS)
    assert len(parts) == 3
    assert parts[0].duration_ticks == 3
    assert parts[1].duration_ticks == 3
    assert parts[2].duration_ticks == 4  # 10 - 3 - 3 = 4


def test_split_preserves_pitch_velocity():
    note = Note(start_tick=100, pitch=72, velocity=100, duration_ticks=960, channel=1)
    for part in split_note(note, 4, TICKS):
        assert part.pitch == 72
        assert part.velocity == 100
        assert part.channel == 1


def test_split_n1_is_noop():
    note = Note(start_tick=0, pitch=60, velocity=80, duration_ticks=480, channel=0)
    assert split_note(note, 1, TICKS) == [note]


# ---------------------------------------------------------------------------
# auto_split_melody — integration test
# ---------------------------------------------------------------------------


def test_auto_split_integration(tmp_path: Path):
    """4-bar melody: 4 notes, lyrics with multi-syllable words → output has >= 4 notes."""
    # Build a 4-note melody: each note is 2 beats long (enough to split)
    note_data = [(i * TICKS * 2, 60 + i, TICKS * 2) for i in range(4)]
    midi_bytes = make_melody_midi(note_data)
    midi_path = tmp_path / "verse.mid"
    midi_path.write_bytes(midi_bytes)

    # Lyrics: 4 lines, each a multi-syllable word
    lyrics = "[verse]\nbeautiful\ncataloging\neverything\nbelonging\n"
    lyrics_path = tmp_path / "lyrics.txt"
    lyrics_path.write_text(lyrics)

    output_path, alignment = auto_split_melody(
        midi_path=midi_path,
        lyrics_path=lyrics_path,
        section="verse",
        min_split_ticks=TICKS,
    )

    output_bytes = output_path.read_bytes()

    # Note count must not decrease
    assert count_note_ons(output_bytes) >= 4

    # Pitches of first note in each phrase must be preserved
    original_pitches = {pitch for _, pitch, _ in note_data}
    output_pitches = set(pitches_from_midi(output_bytes))
    assert original_pitches.issubset(output_pitches)

    # Alignment report has one entry per phrase
    assert len(alignment) == 4
    assert all(a["notes_in"] == 1 for a in alignment)


def test_auto_split_output_path_convention(tmp_path: Path):
    """Output file is named <stem>_split.mid alongside source."""
    midi_bytes = make_melody_midi([(0, 60, TICKS * 4)])
    midi_path = tmp_path / "chorus.mid"
    midi_path.write_bytes(midi_bytes)

    lyrics_path = tmp_path / "lyrics.txt"
    lyrics_path.write_text("[chorus]\nhello\n")

    output_path, _ = auto_split_melody(midi_path=midi_path, lyrics_path=lyrics_path)
    assert output_path == tmp_path / "chorus_split.mid"
    assert output_path.exists()
