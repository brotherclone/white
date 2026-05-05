#!/usr/bin/env python3
"""Melody auto-split: subdivide notes so syllable count matches note count.

When a generated melody has more notes than a lyric line has syllables, ACE Studio
requires manual note subdivision before syllables can be placed. This module
produces a *_split.mid alongside the source MIDI — the source is never modified.

Splitting is driven by pyphen hyphenation (en_US). Each note is assigned one word;
if that word has N > 1 syllables and the note duration >= min_split_ticks, the note
is split into N equal sub-notes at the same pitch and velocity.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mido
import pyphen

from white_generation.pipelines.lyric_pipeline import (
    Phrase,
    _parse_sections,
    extract_phrases,
)

_DIC = pyphen.Pyphen(lang="en_US")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Note:
    start_tick: int
    pitch: int
    velocity: int
    duration_ticks: int
    channel: int


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def syllabify(word: str) -> list[str]:
    """Split word into syllables using pyphen en_US dictionary.

    Strips punctuation before lookup; falls back to [word] if pyphen has no split.
    """
    clean = re.sub(r"[^a-zA-Z'-]", "", word)
    if not clean:
        return [word] if word else []
    inserted = _DIC.inserted(clean)
    parts = inserted.split("-") if "-" in inserted else [clean]
    return parts


def assign_syllables_to_notes(
    notes: list[Note], syllables: list[str]
) -> list[tuple[Note, str]]:
    """Assign one syllable per note, greedy left-to-right.

    Notes beyond the syllable list receive an empty string (melisma continuation).
    """
    return [
        (note, syllables[i] if i < len(syllables) else "")
        for i, note in enumerate(notes)
    ]


def split_note(note: Note, n: int, ticks_per_beat: int) -> list[Note]:
    """Divide note into n equal-duration sub-notes at the same pitch and velocity.

    The last sub-note absorbs the tick remainder from integer division.
    ticks_per_beat is accepted for API symmetry but division is tick-based.
    """
    if n <= 1:
        return [note]
    base = note.duration_ticks // n
    remainder = note.duration_ticks % n
    parts = []
    for i in range(n):
        dur = base + (remainder if i == n - 1 else 0)
        parts.append(
            Note(
                start_tick=note.start_tick + i * base,
                pitch=note.pitch,
                velocity=note.velocity,
                duration_ticks=dur,
                channel=note.channel,
            )
        )
    return parts


# ---------------------------------------------------------------------------
# MIDI I/O helpers
# ---------------------------------------------------------------------------


def _parse_midi_notes(midi_path: Path) -> tuple[list[Note], int]:
    """Parse MIDI file into Note objects with absolute tick positions."""
    mid = mido.MidiFile(str(midi_path))
    ticks_per_beat = mid.ticks_per_beat or 480

    pending: dict[tuple[int, int], tuple[int, int]] = {}
    notes: list[Note] = []

    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = (abs_tick, msg.velocity)
            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                key = (msg.channel, msg.note)
                if key in pending:
                    start, vel = pending.pop(key)
                    notes.append(
                        Note(
                            start_tick=start,
                            pitch=msg.note,
                            velocity=vel,
                            duration_ticks=abs_tick - start,
                            channel=msg.channel,
                        )
                    )

    notes.sort(key=lambda n: n.start_tick)
    return notes, ticks_per_beat


def _write_midi_notes(notes: list[Note], source_midi: Path, output_path: Path) -> None:
    """Write notes to a new MIDI file, copying tempo from source."""
    src = mido.MidiFile(str(source_midi))
    mid = mido.MidiFile(ticks_per_beat=src.ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = mido.bpm2tempo(120)
    for src_track in src.tracks:
        for msg in src_track:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break

    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    events: list[tuple[int, int, int, int, bool]] = []
    for note in notes:
        events.append((note.start_tick, note.pitch, note.velocity, note.channel, True))
        events.append(
            (note.start_tick + note.duration_ticks, note.pitch, 0, note.channel, False)
        )

    events.sort(key=lambda e: (e[0], not e[4]))

    prev_tick = 0
    for abs_tick, pitch, velocity, channel, is_on in events:
        delta = abs_tick - prev_tick
        msg_type = "note_on" if is_on else "note_off"
        track.append(
            mido.Message(
                msg_type, note=pitch, velocity=velocity, time=delta, channel=channel
            )
        )
        prev_tick = abs_tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.save(str(output_path))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def auto_split_melody(
    midi_path: Path,
    lyrics_path: Path,
    section: Optional[str] = None,
    min_split_ticks: int = 480,
    output_path: Optional[Path] = None,
) -> tuple[Path, list[dict]]:
    """Auto-split a melody MIDI to accommodate lyric syllables.

    Args:
        midi_path: Approved melody MIDI file.
        lyrics_path: lyrics.txt with [section] headers.
        section: Section label to use for lyric lines. If None, all lines are used.
        min_split_ticks: Minimum note duration (in ticks) required for splitting.
        output_path: Destination path. Defaults to <stem>_split.mid alongside source.

    Returns:
        Tuple of (output_path, alignment_report).
        alignment_report is a list of per-phrase dicts with note/syllable counts.
    """
    midi_path = Path(midi_path)
    lyrics_path = Path(lyrics_path)

    if output_path is None:
        output_path = midi_path.parent / f"{midi_path.stem}_split.mid"
    output_path = Path(output_path)

    notes, ticks_per_beat = _parse_midi_notes(midi_path)
    phrases: list[Phrase] = extract_phrases(midi_path)

    text = lyrics_path.read_text(encoding="utf-8")
    sections = _parse_sections(text)

    if section and section in sections:
        raw_lines = sections[section].splitlines()
    else:
        raw_lines = []
        for block in sections.values():
            raw_lines.extend(block.splitlines())

    lyric_lines = [ln.strip() for ln in raw_lines if ln.strip()]

    output_notes: list[Note] = []
    alignment: list[dict] = []

    for phrase_idx, phrase in enumerate(phrases):
        phrase_notes = [
            n for n in notes if phrase.start_tick <= n.start_tick <= phrase.end_tick
        ]

        if phrase_idx >= len(lyric_lines) or not phrase_notes:
            output_notes.extend(phrase_notes)
            if phrase_notes:
                alignment.append(
                    {
                        "phrase": phrase_idx,
                        "lyric_line": None,
                        "notes_in": len(phrase_notes),
                        "notes_out": len(phrase_notes),
                        "syllables": 0,
                        "assignments": [],
                    }
                )
            continue

        line = lyric_lines[phrase_idx]
        words = line.split()

        phrase_output: list[Note] = []
        for note_idx, note in enumerate(phrase_notes):
            if note_idx < len(words):
                word_sylls = syllabify(words[note_idx])
                n = len(word_sylls)
                if n > 1 and note.duration_ticks >= min_split_ticks:
                    phrase_output.extend(split_note(note, n, ticks_per_beat))
                else:
                    phrase_output.append(note)
            else:
                phrase_output.append(note)

        all_sylls: list[str] = []
        for word in words:
            all_sylls.extend(syllabify(word))

        assignments = assign_syllables_to_notes(phrase_output, all_sylls)
        alignment.append(
            {
                "phrase": phrase_idx,
                "lyric_line": line,
                "notes_in": len(phrase_notes),
                "notes_out": len(phrase_output),
                "syllables": len(all_sylls),
                "assignments": [syl or "(melisma)" for _, syl in assignments],
            }
        )

        output_notes.extend(phrase_output)

    _write_midi_notes(output_notes, midi_path, output_path)
    return output_path, alignment
