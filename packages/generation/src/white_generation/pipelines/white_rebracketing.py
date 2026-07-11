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
    build_bar_pool(sub_proposal_dirs, white_key, white_bpm, beats_per_bar) -> list[dict]
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import mido
import yaml

from white_core.music.core.enharmonic import flat_to_sharp

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
    beats_per_bar: float,
) -> list[bytes]:
    """Slice a MIDI file into individual bar chunks.

    Each bar spans `ticks_per_beat * beats_per_bar` ticks. Notes are first paired
    into complete (start, end) spans across the *whole* file, then each span is
    assigned to the bar its start falls in — a note that starts within a bar but
    would extend past its end is truncated to the bar boundary.

    Pairing globally first (rather than filtering raw note_on/note_off events per
    bar) matters: a note held from the tail of one bar into the next has its real
    note_off at exactly the next bar's start tick, which is inside that *next*
    bar's tick range. Naively including that note_off in the next bar's own event
    stream would let it pair with a same-pitch note_on that starts the next bar's
    own chord — producing a spurious near-zero-length note there, while the first
    note's true ending becomes an unpaired orphan. Resolving spans globally avoids
    this: a note_off is only ever considered together with the note_on that
    genuinely started it, regardless of which bar the pairing happens to straddle.

    Tick offsets within each bar are re-zeroed so bar 0 starts at tick 0.

    Returns a list of MIDI byte strings, one per bar.
    """
    bar_ticks = int(ticks_per_beat * beats_per_bar)
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))

    # Pair note_on/note_off into complete spans using a per-(channel, pitch) queue,
    # so retriggered notes are matched in chronological order.
    pending: dict[tuple[int, int], list[tuple[int, int]]] = {}
    spans: list[tuple[int, int, int, int, int]] = (
        []
    )  # (start, end, channel, pitch, velocity)
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                pending.setdefault((msg.channel, msg.note), []).append(
                    (abs_tick, msg.velocity)
                )
            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                queue = pending.get((msg.channel, msg.note))
                if queue:
                    start, velocity = queue.pop(0)
                    spans.append((start, abs_tick, msg.channel, msg.note, velocity))

    if not spans:
        return []

    total_ticks = max(end for _, end, *_ in spans)
    n_bars = max(1, (total_ticks + bar_ticks - 1) // bar_ticks)

    bars: list[bytes] = []
    for bar_idx in range(n_bars):
        bar_start = bar_idx * bar_ticks
        bar_end = bar_start + bar_ticks

        bar_events: list[tuple[int, mido.Message]] = []
        for start, end, channel, pitch, velocity in spans:
            if start < bar_start or start >= bar_end:
                continue
            clipped_end = min(end, bar_end)
            bar_events.append(
                (
                    start - bar_start,
                    mido.Message(
                        "note_on",
                        channel=channel,
                        note=pitch,
                        velocity=velocity,
                        time=0,
                    ),
                )
            )
            bar_events.append(
                (
                    clipped_end - bar_start,
                    mido.Message(
                        "note_off", channel=channel, note=pitch, velocity=0, time=0
                    ),
                )
            )

        bar_events.sort(key=lambda e: (e[0], 0 if e[1].type == "note_off" else 1))

        bar_track = mido.MidiTrack()
        prev_rel = 0
        for rel_tick, msg in bar_events:
            bar_track.append(msg.copy(time=rel_tick - prev_rel))
            prev_rel = rel_tick

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
    beats_per_bar: float = 4.0,
) -> bytes:
    """Join a list of bar MIDI byte strings into a single MIDI file.

    The output file has a single track with a tempo message followed by all
    bar events in order.  Tick offsets are adjusted so bars flow continuously.

    beats_per_bar controls exactly how far the tick cursor advances per bar —
    use this to match the target time signature rather than inferring from event
    content (which breaks for sparse or empty bars).
    """
    if not bars:
        raise ValueError("concatenate_bars: bars list is empty")

    tempo_us = round(60_000_000 / bpm)
    merged_track = mido.MidiTrack()
    merged_track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))

    bar_step = int(beats_per_bar * ticks_per_beat)
    tick_cursor = 0
    prev_abs_global = 0  # persists across bars to encode gaps correctly
    for bar_bytes in bars:
        bar_mid = mido.MidiFile(file=io.BytesIO(bar_bytes))
        abs_events: list[tuple[int, mido.Message]] = []
        for track in bar_mid.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                if not msg.is_meta:
                    abs_events.append((abs_tick, msg))

        abs_events.sort(key=lambda x: x[0])

        for abs_tick, msg in abs_events:
            global_tick = tick_cursor + abs_tick
            delta = global_tick - prev_abs_global
            merged_track.append(msg.copy(time=delta))
            prev_abs_global = global_tick

        tick_cursor += bar_step

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
    beats_per_bar: float = 4.0,
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

            bars = extract_bars(rescaled, tpb, beats_per_bar)
            for bar_idx, bar_bytes in enumerate(bars):
                # Skip bars with no audible notes
                bar_mid = mido.MidiFile(file=io.BytesIO(bar_bytes))
                has_notes = any(
                    msg.type == "note_on" and msg.velocity > 0
                    for track in bar_mid.tracks
                    for msg in track
                )
                if not has_notes:
                    continue
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
