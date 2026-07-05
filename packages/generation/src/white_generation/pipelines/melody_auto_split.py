#!/usr/bin/env python3
"""Melody auto-split: subdivide notes to accommodate lyric syllable count.

When a melody has fewer notes than a lyric line has syllables (one note per word,
but words have multiple syllables), ACE Studio requires manual note subdivision
before syllables can be placed. This module increases note count by splitting:
each note is assigned one word; if that word has N > 1 syllables and the note
duration >= min_split_ticks, the note is subdivided into N equal sub-notes at the
same pitch and velocity.

Outputs a *_split.mid alongside the source MIDI — the source is never modified.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mido
import pyphen

from white_generation.pipelines.lyric_pipeline import (
    Phrase,
    _detect_melody_channel,
    _parse_sections,
    extract_phrases,
    parse_arrangement,
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
    n is capped to note.duration_ticks so sub-notes always have duration >= 1.
    ticks_per_beat is accepted for API symmetry but division is tick-based.
    """
    if n <= 1:
        return [note]
    n = min(n, max(1, note.duration_ticks))
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
    """Parse MIDI file into Note objects with absolute tick positions.

    Uses a per-(channel, pitch) stack so retriggered notes are handled correctly.
    """
    mid = mido.MidiFile(str(midi_path))
    ticks_per_beat = mid.ticks_per_beat or 480

    pending: dict[tuple[int, int], list[tuple[int, int]]] = {}
    notes: list[Note] = []

    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                key = (msg.channel, msg.note)
                pending.setdefault(key, []).append((abs_tick, msg.velocity))
            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                key = (msg.channel, msg.note)
                if pending.get(key):
                    start, vel = pending[key].pop(0)
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

    events.sort(key=lambda e: (e[0], e[4]))

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

    normalized_section = section.strip().lower().replace(" ", "_") if section else None
    if normalized_section and normalized_section in sections:
        raw_lines = sections[normalized_section].splitlines()
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

        if not phrase_notes:
            continue

        if phrase_idx >= len(lyric_lines):
            # More melody phrases than lyric lines — usually means the approved MIDI
            # was regenerated/re-approved after lyrics.txt was written. The notes are
            # passed through unsplit so nothing is silently dropped from the MIDI, but
            # this is flagged clearly since it means real notes have no lyric coverage.
            output_notes.extend(phrase_notes)
            alignment.append(
                {
                    "phrase": phrase_idx,
                    "lyric_line": None,
                    "notes_in": len(phrase_notes),
                    "notes_out": len(phrase_notes),
                    "syllables": 0,
                    "assignments": [],
                    "uncovered": True,
                    "warning": (
                        f"No lyric line for phrase {phrase_idx} — lyrics.txt has only "
                        f"{len(lyric_lines)} line(s) for this section but the approved "
                        "MIDI has more phrases. The MIDI may have changed since lyrics "
                        "were generated; consider re-running the lyric pipeline."
                    ),
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
                "uncovered": False,
            }
        )

        output_notes.extend(phrase_output)

    _write_midi_notes(output_notes, midi_path, output_path)
    return output_path, alignment


def auto_split_all_instances(
    lyrics_path: Path,
    approved_dir: Path,
    min_split_ticks: int = 480,
) -> list[dict]:
    """Generate one split MIDI per lyric section instance.

    Iterates sections found in lyrics.txt (e.g. verse, verse_2, verse_3).
    For each, strips the _N suffix to locate the base approved MIDI and writes
    <section_key>_split.mid alongside it.  Sections with no matching MIDI are
    reported as skipped rather than raising an error.
    """
    text = lyrics_path.read_text(encoding="utf-8")
    sections = _parse_sections(text)

    results = []
    for section_key in sections:
        base_label = re.sub(r"_\d+$", "", section_key)
        base_midi = approved_dir / f"{base_label}.mid"
        if not base_midi.exists():
            results.append(
                {
                    "section": section_key,
                    "skipped": True,
                    "reason": f"no approved MIDI for {base_label}",
                }
            )
            continue

        output_path = approved_dir / f"{section_key}_split.mid"
        out, alignment = auto_split_melody(
            midi_path=base_midi,
            lyrics_path=lyrics_path,
            section=section_key,
            min_split_ticks=min_split_ticks,
            output_path=output_path,
        )
        uncovered_count = sum(1 for a in alignment if a.get("uncovered"))
        result_entry = {
            "section": section_key,
            "skipped": False,
            "split_midi": str(out),
            "alignment": alignment,
            "uncovered_phrase_count": uncovered_count,
        }
        if uncovered_count:
            result_entry["warning"] = (
                f"{section_key}: {uncovered_count} phrase(s) have no lyric coverage — "
                "the approved MIDI may have changed since lyrics were generated for "
                "this section."
            )
        results.append(result_entry)

    return results


def assemble_melody_midi(
    arrangement_path: Path,
    approved_dir: Path,
    bpm: int,
    time_sig_str: str,
    output_path: Optional[Path] = None,
    melody_channel: int = 4,
    ticks_per_beat: int = 480,
) -> Path:
    """Assemble a full-length melody MIDI from arrangement clips.

    For each melody clip instance in arrangement order, looks for
    <instance_key>_split.mid, then <base_label>_split.mid, then <base_label>.mid.
    Places each clip at its absolute bar position so the output can be imported
    at bar 1 in Logic without further offsetting.

    Returns the path to the written assembled MIDI.
    """
    clips = parse_arrangement(arrangement_path)
    resolved_channel = _detect_melody_channel(clips, fallback=melody_channel)
    melody_clips = [c for c in clips if c["channel"] == resolved_channel]

    parts = str(time_sig_str).split("/")
    numerator = int(parts[0])
    denominator = int(parts[1])
    # ticks per bar accounts for denominator: a 7/8 bar is 7 eighth-notes, not 7 quarter-notes
    ticks_per_bar = int(numerator * (4 / denominator) * ticks_per_beat)

    if output_path is None:
        output_path = approved_dir.parent / "assembled_melody.mid"
    output_path = Path(output_path)

    label_seen: dict[str, int] = {}
    all_events: list[tuple[int, mido.Message]] = []

    for clip in melody_clips:
        label = clip["clip_name"]
        label_seen[label] = label_seen.get(label, 0) + 1
        n = label_seen[label]
        instance_key = label if n == 1 else f"{label}_{n}"

        # Resolve the best available MIDI for this instance
        candidates = [
            approved_dir / f"{instance_key}_split.mid",
            approved_dir / f"{re.sub(r'_\\d+$', '', label)}_split.mid",
            approved_dir / f"{label}.mid",
        ]
        midi_path: Optional[Path] = next((p for p in candidates if p.exists()), None)
        if midi_path is None:
            continue

        # Start position: prefer bar/beat (start_bars), fall back to timecode
        start_bars_val = clip.get("start_bars")
        if start_bars_val is not None:
            start_tick = (start_bars_val - 1) * ticks_per_bar
        else:
            start_tick = round(clip["timecode_secs"] * bpm / 60.0 * ticks_per_beat)

        src = mido.MidiFile(str(midi_path))
        scale = ticks_per_beat / (src.ticks_per_beat or 480)

        # Find the earliest note_on tick so we can normalise the clip to start
        # at tick 0.  Logic exports MIDIs with notes at their absolute timeline
        # position, so without normalisation the assembler would double-count
        # the bar offset.
        clip_origin = None
        for track in src.tracks:
            abs_src = 0
            for msg in track:
                abs_src += msg.time
                if msg.type in ("note_on", "note_off"):
                    if clip_origin is None or abs_src < clip_origin:
                        clip_origin = abs_src
        if clip_origin is None:
            clip_origin = 0

        for track in src.tracks:
            abs_src = 0
            for msg in track:
                abs_src += msg.time
                if msg.type in ("note_on", "note_off"):
                    dest_tick = start_tick + round((abs_src - clip_origin) * scale)
                    all_events.append((dest_tick, msg.copy(time=0)))

    if not all_events:
        raise ValueError("No melody clips resolved — run auto-split first")

    # note_off before note_on at the same tick to avoid stuck notes
    all_events.sort(key=lambda e: (e[0], 0 if e[1].type == "note_off" else 1))

    out_mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    out_mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))

    prev_tick = 0
    for abs_tick, msg in all_events:
        delta = abs_tick - prev_tick
        track.append(msg.copy(time=delta))
        prev_tick = abs_tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    out_mid.save(str(output_path))
    return output_path


def assemble_lyrics_text(
    arrangement_path: Path,
    lyrics_path: Path,
    melody_channel: int = 4,
) -> str:
    """Build lyric text matching assembled_melody.mid's real instance order.

    `lyrics.txt` only contains one block for EXACT-repeat sections (e.g. a chorus
    that plays 4 times has a single `[chorus]` block, meant to be reused verbatim).
    `assemble_melody_midi` concatenates the *real* repeated notes, so comparing the
    single lyric block against the assembled MIDI's note count looks like a large
    mismatch. This walks the same arrangement-order clip list `assemble_melody_midi`
    uses and repeats each instance's lyric block — reusing the instance's own block
    if `lyrics.txt` has one (`verse_2`, `verse_3`, ...), otherwise falling back to the
    base label's block (`chorus` reused for `chorus_2`, `chorus_3`, `chorus_4`).
    """
    clips = parse_arrangement(arrangement_path)
    resolved_channel = _detect_melody_channel(clips, fallback=melody_channel)
    melody_clips = [c for c in clips if c["channel"] == resolved_channel]

    text = lyrics_path.read_text(encoding="utf-8")
    sections = _parse_sections(text)

    label_seen: dict[str, int] = {}
    blocks: list[str] = []
    for clip in melody_clips:
        label = clip["clip_name"]
        label_seen[label] = label_seen.get(label, 0) + 1
        n = label_seen[label]
        instance_key = label if n == 1 else f"{label}_{n}"

        block = sections.get(instance_key)
        if block is None:
            block = sections.get(label)

        if block is None:
            blocks.append(
                f"[{instance_key}]\n"
                f"# NO LYRIC BLOCK FOUND for '{label}' "
                f"(checked '{instance_key}' and '{label}')"
            )
        else:
            blocks.append(f"[{instance_key}]\n{block}")

    return "\n\n".join(blocks) + "\n"


def write_assembled_lyrics(
    arrangement_path: Path,
    lyrics_path: Path,
    output_path: Optional[Path] = None,
    melody_channel: int = 4,
) -> Path:
    """Write assemble_lyrics_text()'s output to <melody_dir>/assembled_lyrics.txt."""
    if output_path is None:
        output_path = lyrics_path.parent / "assembled_lyrics.txt"
    output_path = Path(output_path)
    output_path.write_text(
        assemble_lyrics_text(
            arrangement_path, lyrics_path, melody_channel=melody_channel
        ),
        encoding="utf-8",
    )
    return output_path
