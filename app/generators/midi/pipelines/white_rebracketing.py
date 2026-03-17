#!/usr/bin/env python3
"""
White Synthesis Pipeline — MIDI rebracketing utilities.

WORKFLOW
--------
The White song is the chromatic synthesis of all color songs.  Instead of
generating chord progressions from scratch via Markov chains, the White chord
pipeline reads approved MIDI files from the sub-proposal production directories,
transposes them into the White key, adjusts the BPM, slices them into individual
bars, then generates candidates by randomly drawing and shuffling bars (cut-up).

Public API
----------
    transpose_midi(midi_bytes, semitone_delta) -> bytes
    set_midi_bpm(midi_bytes, bpm) -> bytes
    extract_bars(midi_bytes, ticks_per_beat, beats_per_bar) -> list[bytes]
    concatenate_bars(bars, ticks_per_beat, bpm) -> bytes
    build_bar_pool(sub_proposal_dirs, white_key, white_bpm) -> list[dict]
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import mido
import yaml

from app.structures.music.core.enharmonic import flat_to_sharp

# ---------------------------------------------------------------------------
# Root → semitone offset (chromatic scale, C = 0)
# ---------------------------------------------------------------------------

_ROOT_TO_SEMITONE: dict[str, int] = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}

_MIDI_NOTE_MIN = 21
_MIDI_NOTE_MAX = 108


def _root_to_semitone(root: str) -> int:
    """Return the chromatic pitch class (0–11) for a root name like 'F#' or 'Bb'."""
    # Normalise unicode accidentals
    root = root.replace("♭", "b").replace("♯", "#")
    # Resolve enharmonic sharps that aren't in the table (e.g. A# → Bb)
    root = flat_to_sharp.get(
        root, root
    )  # sharp_to_flat inverse — use flat_to_sharp to get sharp
    # flat_to_sharp maps flats → sharps; _ROOT_TO_SEMITONE has both, so just look up directly
    val = _ROOT_TO_SEMITONE.get(root)
    if val is None:
        raise ValueError(f"Unknown root note: {root!r}")
    return val


def _parse_key_root(key_str: str) -> str:
    """Extract the root name from a key string like 'G minor' or 'F# Major'."""
    parts = key_str.strip().split()
    return parts[0] if parts else "C"


# ---------------------------------------------------------------------------
# Core MIDI transforms
# ---------------------------------------------------------------------------


def transpose_midi(midi_bytes: bytes, semitone_delta: int) -> bytes:
    """Transpose all note-on / note-off messages by semitone_delta.

    Notes shifted outside [21, 108] are clamped with a warning.
    All other messages (control change, tempo, etc.) are passed through unchanged.
    """
    if semitone_delta == 0:
        return midi_bytes

    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    clamped = 0

    for track in mid.tracks:
        for msg in track:
            if msg.type in ("note_on", "note_off"):
                new_note = msg.note + semitone_delta
                if new_note < _MIDI_NOTE_MIN or new_note > _MIDI_NOTE_MAX:
                    clamped += 1
                    new_note = max(_MIDI_NOTE_MIN, min(_MIDI_NOTE_MAX, new_note))
                msg.note = new_note

    if clamped:
        warnings.warn(
            f"transpose_midi: {clamped} note(s) clamped to [{_MIDI_NOTE_MIN}, {_MIDI_NOTE_MAX}]"
        )

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def set_midi_bpm(midi_bytes: bytes, bpm: int) -> bytes:
    """Replace (or insert) the MIDI tempo meta message to match bpm.

    The tempo message is placed at tick 0 on track 0.  Any existing tempo
    messages on track 0 are removed.  Tick values are not stretched — only
    the playback speed interpretation changes.
    """
    tempo_us = round(60_000_000 / bpm)
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

    if not mid.tracks:
        buf = io.BytesIO()
        mid.save(file=buf)
        return buf.getvalue()

    # Remove existing tempo messages from track 0
    track0 = mid.tracks[0]
    new_track0 = mido.MidiTrack()
    for msg in track0:
        if not (hasattr(msg, "type") and msg.type == "set_tempo"):
            new_track0.append(msg)

    # Insert new tempo at position 0 (time=0)
    tempo_msg = mido.MetaMessage("set_tempo", tempo=tempo_us, time=0)
    new_track0.insert(0, tempo_msg)
    mid.tracks[0] = new_track0

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def extract_bars(
    midi_bytes: bytes,
    ticks_per_beat: int,
    beats_per_bar: int,
) -> list[bytes]:
    """Slice a MIDI file into individual bar chunks.

    Each bar spans `ticks_per_beat * beats_per_bar` ticks.  Notes that start
    within a bar but would extend past its end are truncated.  Tick offsets
    within each bar are re-zeroed so bar 0 starts at tick 0.

    Returns a list of MIDI byte strings, one per bar.
    """
    bar_ticks = ticks_per_beat * beats_per_bar
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

    # Convert all tracks to absolute tick events, merge into one stream
    events: list[tuple[int, mido.Message]] = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            events.append((abs_tick, msg.copy(time=abs_tick)))

    if not events:
        return []

    total_ticks = max(t for t, _ in events)
    n_bars = max(1, (total_ticks + bar_ticks - 1) // bar_ticks)

    bars: list[bytes] = []
    for bar_idx in range(n_bars):
        bar_start = bar_idx * bar_ticks
        bar_end = bar_start + bar_ticks

        bar_track = mido.MidiTrack()
        prev_abs = bar_start
        pending_note_offs: dict[tuple[int, int], int] = {}  # (channel, note) → end_tick

        for abs_tick, msg in sorted(events, key=lambda x: x[0]):
            if msg.is_meta:
                continue
            if abs_tick < bar_start or abs_tick >= bar_end:
                continue

            if msg.type == "note_on" and msg.velocity > 0:
                # Track pending note-off within this bar
                pending_note_offs[(msg.channel, msg.note)] = bar_end

            rel_tick = abs_tick - bar_start
            delta = rel_tick - (prev_abs - bar_start)
            bar_track.append(msg.copy(time=delta))
            prev_abs = abs_tick

        # Truncate any open notes at bar end
        for (ch, note), end in pending_note_offs.items():
            off_rel = end - bar_start
            delta = off_rel - (prev_abs - bar_start)
            if delta >= 0:
                bar_track.append(
                    mido.Message(
                        "note_off", channel=ch, note=note, velocity=0, time=delta
                    )
                )
                prev_abs = end

        bar_track.append(mido.MetaMessage("end_of_track", time=0))

        bar_mid = mido.MidiFile(ticks_per_beat=ticks_per_beat, type=0)
        bar_mid.tracks.append(bar_track)

        buf = io.BytesIO()
        bar_mid.save(file=buf)
        bars.append(buf.getvalue())

    return bars


def concatenate_bars(
    bars: list[bytes],
    ticks_per_beat: int,
    bpm: int,
) -> bytes:
    """Join a list of bar MIDI byte strings into a single MIDI file.

    The output file has a single track with a tempo message followed by all
    bar events in order.  Tick offsets are adjusted so bars flow continuously.
    """
    if not bars:
        raise ValueError("concatenate_bars: bars list is empty")

    tempo_us = round(60_000_000 / bpm)
    merged_track = mido.MidiTrack()
    merged_track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))

    tick_cursor = 0
    for bar_bytes in bars:
        bar_mid = mido.MidiFile(file=io.BytesIO(bar_bytes))
        # Convert to absolute ticks within the bar
        abs_events: list[tuple[int, mido.Message]] = []
        for track in bar_mid.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                if not msg.is_meta:
                    abs_events.append((abs_tick, msg))

        abs_events.sort(key=lambda x: x[0])

        prev_abs_global = tick_cursor
        for abs_tick, msg in abs_events:
            global_tick = tick_cursor + abs_tick
            delta = global_tick - prev_abs_global
            merged_track.append(msg.copy(time=delta))
            prev_abs_global = global_tick

        # Advance cursor by one bar's worth of ticks (infer from ticks_per_beat)
        if bar_mid.tracks:
            bar_len = sum(msg.time for track in bar_mid.tracks for msg in track)
            tick_cursor += bar_len if bar_len > 0 else ticks_per_beat * 4

    merged_track.append(mido.MetaMessage("end_of_track", time=0))

    out = mido.MidiFile(ticks_per_beat=ticks_per_beat, type=0)
    out.tracks.append(merged_track)
    buf = io.BytesIO()
    out.save(file=buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Bar pool builder
# ---------------------------------------------------------------------------


def build_bar_pool(
    sub_proposal_dirs: list[Path],
    white_key: str,
    white_bpm: int,
) -> list[dict]:
    """Build a pool of transposed, BPM-normalised bars from sub-proposal dirs.

    For each sub-proposal directory:
      1. Read chords/review.yml → donor key and BPM.
      2. Load all chords/approved/*.mid files.
      3. Transpose notes to White key; replace tempo to White BPM.
      4. Extract individual bars.

    Returns a flat list of bar dicts:
      {midi_bytes, source_dir, source_file, donor_color, donor_key, bar_index}
    """

    white_root = _parse_key_root(white_key)
    try:
        white_semitone = _root_to_semitone(white_root)
    except ValueError:
        white_semitone = 0

    pool: list[dict] = []

    for sub_dir in sub_proposal_dirs:
        sub_dir = Path(sub_dir)
        review_path = sub_dir / "chords" / "review.yml"
        approved_dir = sub_dir / "chords" / "approved"

        if not review_path.exists():
            warnings.warn(
                f"build_bar_pool: no chords/review.yml in {sub_dir} — skipping"
            )
            continue

        with open(review_path) as f:
            review = yaml.safe_load(f) or {}

        donor_key = str(review.get("key", "C major"))
        donor_color = str(review.get("color", ""))

        donor_root = _parse_key_root(donor_key)
        try:
            donor_semitone = _root_to_semitone(donor_root)
        except ValueError:
            donor_semitone = 0

        semitone_delta = (white_semitone - donor_semitone) % 12

        midi_files = sorted(approved_dir.glob("*.mid")) if approved_dir.exists() else []
        if not midi_files:
            warnings.warn(
                f"build_bar_pool: no approved MIDIs in {approved_dir} — skipping"
            )
            continue

        for midi_path in midi_files:
            raw = midi_path.read_bytes()
            if not raw:
                warnings.warn(f"build_bar_pool: skipping empty file {midi_path}")
                continue
            transposed = transpose_midi(raw, semitone_delta)
            rescaled = set_midi_bpm(transposed, white_bpm)

            mid = mido.MidiFile(file=io.BytesIO(rescaled))
            tpb = mid.ticks_per_beat or 480
            beats_per_bar = _beats_per_bar_from_review(review)

            bars = extract_bars(rescaled, tpb, beats_per_bar)
            for bar_idx, bar_bytes in enumerate(bars):
                pool.append(
                    {
                        "midi_bytes": bar_bytes,
                        "source_dir": str(sub_dir),
                        "source_file": midi_path.name,
                        "donor_color": donor_color,
                        "donor_key": donor_key,
                        "bar_index": bar_idx,
                    }
                )

    return pool


def _beats_per_bar_from_review(review: dict) -> int:
    """Derive beats_per_bar from a chord review.yml dict."""
    # chords/review.yml doesn't store time_sig directly; default to 4
    # (all current color songs are 4/4 or 7/8 — the White song proposal
    #  can supply this explicitly via the pipeline caller)
    return 4
