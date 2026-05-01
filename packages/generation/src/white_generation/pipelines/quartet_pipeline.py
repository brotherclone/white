#!/usr/bin/env python3
"""
Four-part (SATB/quartet) generation pipeline for the Music Production Pipeline.

Reads the approved melody MIDI for each section, derives three lower voices
(alto, tenor, bass-voice) using interval-offset templates, enforces basic
counterpoint constraints (no parallel 5ths/octaves, voice crossing correction),
and writes a multi-channel MIDI file per section alongside a review.yml.

Pipeline position: chords → drums → bass → melody → QUARTET

Usage:
    python -m app.generators.midi.pipelines.quartet_pipeline \\
        --production-dir shrink_wrapped/.../production/black__sequential_dissolution_v2 \\
        --singer gabriel --top-k 3

    # Promote an approved quartet candidate
    python -m app.generators.midi.production.promote_part \\
        --review <production-dir>/quartet/review.yml
"""

import argparse
import io
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mido
import numpy as np
import yaml

from white_composition.init_production import load_song_context
from white_generation.patterns.bass_patterns import (
    ALL_TEMPLATES as ALL_BASS_TEMPLATES,
)
from white_generation.patterns.bass_patterns import (
    BassPattern,
)
from white_generation.patterns.bass_patterns import (
    select_templates as select_bass_templates,
)
from white_generation.patterns.melody_patterns import (
    ALL_TEMPLATES as ALL_MELODY_TEMPLATES,
)
from white_generation.patterns.melody_patterns import (
    MelodyPattern,
    SingerRange,
)
from white_generation.patterns.quartet_patterns import (
    INSTRUMENT_RANGES,
    QUARTET_CHANNELS,
    VELOCITY,
    VoicePattern,
    check_parallels,
    clamp_to_voice_range,
    counterpoint_score,
    fix_voice_crossing,
    get_patterns_for_voice,
)
from white_generation.pipelines.bass_pipeline import (
    bass_pattern_to_midi_bytes,
    extract_section_chord_data,
)
from white_generation.pipelines.chord_pipeline import (
    compute_chromatic_match,
    get_chromatic_target,
)
from white_generation.pipelines.melody_pipeline import (
    generate_melody_for_section,
)

QUARTET_DIR_NAME = "quartet"
REVIEW_FILENAME = "review.yml"
MAX_OFFSET_CHANGE = 4  # semitones — cap per-beat leap in a generated voice


def _to_python(obj):
    """Recursively convert numpy types to native Python for YAML serialization."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Soprano note extraction
# ---------------------------------------------------------------------------


def extract_soprano_notes(midi_bytes: bytes) -> list[int]:
    """Extract ordered note-on pitches from a mono melody MIDI (channel 0).

    Returns absolute MIDI pitches in the order they appear across all tracks.
    """
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    notes: list[tuple[int, int]] = []  # (abs_tick, pitch)
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                notes.append((abs_tick, msg.note))
    notes.sort(key=lambda x: x[0])
    return [n for _, n in notes]


def extract_note_events(
    midi_bytes: bytes,
) -> list[tuple[int, int, int]]:
    """Extract (abs_tick, pitch, duration_ticks) triples from melody MIDI.

    Pairs each note-on with the next note-off for the same pitch using a
    per-pitch queue, so repeated pitches each get their correct duration.
    Returns events sorted by abs_tick.
    """
    from collections import defaultdict, deque

    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    tpb = mid.ticks_per_beat or 480

    # First pass: collect all on/off events in order
    raw: list[tuple[int, str, int]] = []  # (abs_tick, type, pitch)
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                raw.append((abs_tick, "on", msg.note))
            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                raw.append((abs_tick, "off", msg.note))
    raw.sort(key=lambda x: x[0])

    # Second pass: pair each note-on with the next note-off for the same pitch
    pending: dict[int, deque] = defaultdict(deque)  # pitch → queue of on-ticks
    on_events: list[tuple[int, int]] = []  # (abs_tick, pitch) in order
    off_by_on: dict[tuple[int, int], int] = {}  # (on_tick, pitch) → off_tick

    for abs_tick, kind, pitch in raw:
        if kind == "on":
            pending[pitch].append(abs_tick)
            on_events.append((abs_tick, pitch))
        elif kind == "off" and pending[pitch]:
            on_tick = pending[pitch].popleft()
            off_by_on[(on_tick, pitch)] = abs_tick

    on_events.sort(key=lambda x: x[0])

    result = []
    for i, (tick, pitch) in enumerate(on_events):
        off_tick = off_by_on.get((tick, pitch))
        if off_tick is not None:
            dur = max(off_tick - tick, tpb // 4)
        elif i + 1 < len(on_events):
            dur = max(on_events[i + 1][0] - tick, tpb // 4)
        else:
            dur = tpb  # quarter note fallback for final note
        result.append((tick, pitch, dur))

    return result


# ---------------------------------------------------------------------------
# Voice generation
# ---------------------------------------------------------------------------


def _apply_offset_with_leap_cap(
    base_note: int,
    offset: int,
    prev_voice_note: Optional[int],
) -> int:
    """Apply offset to base note, capping change from previous note at MAX_OFFSET_CHANGE."""
    candidate = base_note + offset
    if prev_voice_note is not None:
        delta = candidate - prev_voice_note
        if abs(delta) > MAX_OFFSET_CHANGE:
            sign = 1 if delta > 0 else -1
            candidate = prev_voice_note + sign * MAX_OFFSET_CHANGE
    return candidate


def generate_voice_notes(
    soprano_notes: list[int],
    pattern: VoicePattern,
) -> list[int]:
    """Generate absolute MIDI pitches for one lower voice from a template.

    Tiles the offset pattern to match the soprano note count, applies
    leap capping, then clamps to the voice range.
    """
    n = len(soprano_notes)
    offsets = pattern.interval_offsets
    tiled = [offsets[i % len(offsets)] for i in range(n)]

    result: list[int] = []
    prev: Optional[int] = None
    for soprano, offset in zip(soprano_notes, tiled):
        raw = _apply_offset_with_leap_cap(soprano, offset, prev)
        note = clamp_to_voice_range(raw, pattern.voice_type)
        result.append(note)
        prev = note
    return result


def resolve_parallel_violations(
    soprano_notes: list[int],
    voice_notes: list[int],
    voice_type: str,
    max_attempts: int = 3,
) -> list[int]:
    """Attempt to fix parallel 5ths/octaves by nudging the offending voice note ±1st.

    Tries shifting up then down. Accepts the first fix that eliminates the
    violation. Returns the (possibly corrected) voice note list.
    """
    notes = list(voice_notes)
    for _ in range(max_attempts):
        violations = check_parallels(soprano_notes, notes)
        if not violations:
            break
        # Parse the first violation beat index
        first = violations[0]
        try:
            beat = int(first.split("beat ")[1].split("→")[0])
        except (IndexError, ValueError):
            break
        # Try nudging the second note of the parallel pair
        target_beat = min(beat + 1, len(notes) - 1)
        original = notes[target_beat]
        fixed = False
        for shift in (1, -1, 2, -2):
            candidate = clamp_to_voice_range(original + shift, voice_type)
            notes[target_beat] = candidate
            if not check_parallels(soprano_notes, notes):
                fixed = True
                break
        if not fixed:
            notes[target_beat] = original
    return notes


# ---------------------------------------------------------------------------
# Multi-channel MIDI assembly
# ---------------------------------------------------------------------------


def build_quartet_midi(
    note_events: list[tuple[int, int, int]],
    alto_notes: list[int],
    tenor_notes: list[int],
    bass_voice_notes: list[int],
    bpm: int = 120,
    ticks_per_beat: int = 480,
) -> bytes:
    """Build a 4-channel MIDI file from soprano events + 3 generated voices.

    Each voice occupies its own MIDI channel (0=soprano, 1=alto,
    2=tenor, 3=bass-voice).  All voices share the same rhythmic grid
    as the soprano (same ticks, same duration).

    Args:
        note_events: [(abs_tick, pitch, duration_ticks), ...] from soprano.
        alto_notes: Absolute MIDI pitches per note event index.
        tenor_notes: Absolute MIDI pitches per note event index.
        bass_voice_notes: Absolute MIDI pitches per note event index.
        bpm: Tempo in beats per minute.
        ticks_per_beat: MIDI ticks per beat resolution.
    """
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    tempo = mido.bpm2tempo(bpm)

    # Build one merged track with all voices interleaved
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    # Collect all (abs_tick, msg) pairs across voices
    raw_events: list[tuple[int, mido.Message]] = []

    voice_data = [
        (QUARTET_CHANNELS["soprano"], [e[1] for e in note_events]),
        (QUARTET_CHANNELS["alto"], alto_notes),
        (QUARTET_CHANNELS["tenor"], tenor_notes),
        (QUARTET_CHANNELS["bass_voice"], bass_voice_notes),
    ]

    for ch, pitches in voice_data:
        vel_on = VELOCITY["normal"]
        vel_off = 0
        for i, (tick, _, dur) in enumerate(note_events):
            if i >= len(pitches):
                break
            pitch = pitches[i]
            raw_events.append(
                (
                    tick,
                    mido.Message(
                        "note_on", channel=ch, note=pitch, velocity=vel_on, time=0
                    ),
                )
            )
            raw_events.append(
                (
                    tick + dur,
                    mido.Message(
                        "note_off", channel=ch, note=pitch, velocity=vel_off, time=0
                    ),
                )
            )

    # Sort by tick, then note_off before note_on at same tick
    raw_events.sort(key=lambda x: (x[0], 0 if x[1].type == "note_off" else 1))

    # Convert absolute ticks to delta ticks
    prev_tick = 0
    for abs_tick, msg in raw_events:
        delta = abs_tick - prev_tick
        track.append(msg.copy(time=delta))
        prev_tick = abs_tick

    track.append(mido.MetaMessage("end_of_track", time=0))

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_quartet_candidate(
    soprano_notes: list[int],
    alto_notes: list[int],
    tenor_notes: list[int],
    bass_voice_notes: list[int],
    scorer_result: Optional[dict],
    color: str,
) -> dict:
    """Compute composite score for a quartet candidate.

    Counterpoint score: mean of per-voice counterpoint_score() results.
    Chromatic score: compute_chromatic_match(scorer_result, target) when a
        Refractor result is available; falls back to 0.5 otherwise.
    Composite: 30% counterpoint + 70% chromatic.

    Args:
        scorer_result: Pre-computed Refractor output dict for the MIDI, or None
            to skip chromatic scoring (counterpoint only; composite = counterpoint).
    """
    n = len(soprano_notes)
    alto_cp = counterpoint_score(soprano_notes[:n], alto_notes[:n])
    tenor_cp = counterpoint_score(soprano_notes[:n], tenor_notes[:n])
    bass_cp = counterpoint_score(soprano_notes[:n], bass_voice_notes[:n])
    mean_cp = round((alto_cp + tenor_cp + bass_cp) / 3, 3)

    if scorer_result is not None:
        target = get_chromatic_target(color)
        chromatic_val = compute_chromatic_match(scorer_result, target)
        confidence = float(scorer_result.get("confidence", 0.5))
        composite = round(0.30 * mean_cp + 0.70 * chromatic_val, 3)
    else:
        # No Refractor available — composite is counterpoint score only
        chromatic_val = None
        confidence = 0.0
        composite = mean_cp

    return {
        "counterpoint": mean_cp,
        "alto_score": alto_cp,
        "tenor_score": tenor_cp,
        "bass_voice_score": bass_cp,
        "chromatic": chromatic_val,
        "confidence": confidence,
        "composite": composite,
    }


# ---------------------------------------------------------------------------
# Simple-voicings helpers
# ---------------------------------------------------------------------------

_NOTE_NAME_TO_PC: dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}


def _note_name_to_pc(name: str) -> int:
    """Convert a note name string like 'D#2' or 'Gb3' to a pitch class 0–11."""
    letter = name[0].upper()
    pc = _NOTE_NAME_TO_PC.get(letter, 0)
    for ch in name[1:]:
        if ch == "#":
            pc += 1
        elif ch == "b":
            pc -= 1
        else:
            break  # hit the octave digit
    return pc % 12


def triad_pcs_from_chord_notes(notes: list[str]) -> set[int]:
    """Return the root/3rd/5th pitch classes from a chord's note-name list.

    Root is the lowest note's pitch class.  The 3rd and 5th are identified
    by their interval above the root (minor 3rd=3, major 3rd=4, perfect 5th=7).
    Any remaining notes (extensions, 7ths, 9ths, etc.) are excluded so that
    lower voices snap to consonant chord tones only.
    """
    if not notes:
        return set()
    pcs = [_note_name_to_pc(n) for n in notes]
    root = pcs[0]
    triad: set[int] = {root}
    for pc in pcs[1:]:
        interval = (pc - root) % 12
        if interval in (3, 4, 7):  # min3, maj3, p5
            triad.add(pc)
    # Fallback: if we only found the root, include the next two unique PCs
    if len(triad) < 2:
        for pc in pcs[1:]:
            triad.add(pc)
            if len(triad) >= 3:
                break
    return triad


def snap_note_to_triad(note: int, triad_pcs: set[int]) -> int:
    """Snap a MIDI note to the nearest pitch in triad_pcs (across all octaves)."""
    if not triad_pcs:
        return note
    best, best_dist = note, float("inf")
    for pc in triad_pcs:
        # Find the closest octave of this pitch class to the soprano note
        delta = (pc - note % 12) % 12
        if delta > 6:
            delta -= 12  # prefer downward snap when equidistant
        candidate = note + delta
        dist = abs(candidate - note)
        if dist < best_dist:
            best_dist = dist
            best = candidate
    return best


def load_section_triad_map(production_dir: Path, label_key: str) -> list[set[int]]:
    """Return a list of triad pitch-class sets, one per chord in the section.

    Reads chords/review.yml and finds the approved candidate whose label
    matches label_key.  Returns [] if the data is unavailable.
    """
    review_path = production_dir / "chords" / "review.yml"
    if not review_path.exists():
        return []
    with open(review_path) as f:
        data = yaml.safe_load(f) or {}
    for cand in data.get("candidates", []):
        cand_label = (
            str(cand.get("label", "") or "").lower().replace("-", "_").replace(" ", "_")
        )
        if cand_label == label_key and cand.get("status") in (
            "approved",
            "accepted",
        ):
            chords = cand.get("chords", [])
            return [triad_pcs_from_chord_notes(c.get("notes", [])) for c in chords]
    return []


def snap_soprano_to_triads(
    soprano_notes: list[int],
    triad_map: list[set[int]],
) -> list[int]:
    """Snap each soprano note to the nearest triad tone of its chord.

    Maps notes evenly across chords when note count and chord count differ.
    Returns the original list unchanged if triad_map is empty.
    """
    if not triad_map:
        return soprano_notes
    n_notes = len(soprano_notes)
    n_chords = len(triad_map)
    snapped = []
    for i, note in enumerate(soprano_notes):
        chord_idx = min(int(i * n_chords / n_notes), n_chords - 1)
        snapped.append(snap_note_to_triad(note, triad_map[chord_idx]))
    return snapped


# ---------------------------------------------------------------------------
# Multi-voice MIDI merger
# ---------------------------------------------------------------------------


def merge_voices_to_quartet_midi(
    voice_midis: list[tuple[bytes, int]],
    bpm: int,
    ticks_per_beat: int = 480,
) -> bytes:
    """Merge independently-generated voice MIDIs into a single multi-channel file.

    Args:
        voice_midis: List of (midi_bytes, target_channel) — one per voice.
            Each MIDI is remapped to its target channel. Channel 0 = violin I,
            1 = violin II, 2 = viola, 3 = cello.
        bpm: Tempo (written into track 0).
        ticks_per_beat: MIDI resolution of the output file.

    Returns:
        Multi-channel MIDI bytes.
    """
    tempo_us = mido.bpm2tempo(bpm)
    out = mido.MidiFile(ticks_per_beat=ticks_per_beat, type=0)
    track = mido.MidiTrack()
    out.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))

    raw: list[tuple[int, mido.Message]] = []

    for midi_bytes, target_ch in voice_midis:
        if not midi_bytes:
            continue
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
        # Rescale ticks if source resolution differs
        scale = ticks_per_beat / (mid.ticks_per_beat or ticks_per_beat)
        for src_track in mid.tracks:
            abs_tick = 0
            for msg in src_track:
                abs_tick += msg.time
                if msg.is_meta:
                    continue
                scaled = int(abs_tick * scale)
                remapped = msg.copy(channel=target_ch, time=0)
                raw.append((scaled, remapped))

    raw.sort(key=lambda x: (x[0], 0 if x[1].type == "note_off" else 1))

    prev = 0
    for abs_tick, msg in raw:
        track.append(msg.copy(time=abs_tick - prev))
        prev = abs_tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    buf = io.BytesIO()
    out.save(file=buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Per-instrument voice generators
# ---------------------------------------------------------------------------

# Instrument ranges as SingerRange-compatible objects
_VIOLIN_II_RANGE = SingerRange(
    "Violin II",
    INSTRUMENT_RANGES["violin_ii"][0],
    INSTRUMENT_RANGES["violin_ii"][1],
    "violin",
)
_VIOLA_RANGE = SingerRange(
    "Viola", INSTRUMENT_RANGES["viola"][0], INSTRUMENT_RANGES["viola"][1], "viola"
)


def _generate_string_voice(
    instrument: str,
    voicings: list[list[int]],
    singer_range: SingerRange,
    energy: str,
    time_sig: tuple[int, int],
    bpm: int,
    rng: random.Random,
) -> tuple[bytes, str]:
    """Generate an independent melodic voice (violin II or viola) using MelodyPattern.

    Prefers 'lead' use_case patterns; stepwise/arpeggiated contours for viola,
    conversational/stepwise for violin II.  Falls back to any available template.

    Returns (midi_bytes, pattern_name).
    """
    contour_pref = {
        "violin_ii": ["conversational", "stepwise", "arpeggiated", "haiku"],
        "viola": ["stepwise", "declarative", "arpeggiated", "conversational"],
    }.get(instrument, [])

    # Gather candidates: prefer lead use_case + preferred contours
    candidates = [
        p
        for p in ALL_MELODY_TEMPLATES
        if p.time_sig == time_sig and p.use_case == "lead" and p.contour in contour_pref
    ]
    if not candidates:
        candidates = [
            p
            for p in ALL_MELODY_TEMPLATES
            if p.time_sig == time_sig and p.use_case == "lead"
        ]
    if not candidates:
        candidates = [p for p in ALL_MELODY_TEMPLATES if p.time_sig == time_sig]
    if not candidates:
        candidates = list(ALL_MELODY_TEMPLATES)

    pattern: MelodyPattern = rng.choice(candidates)
    midi_bytes, _ = generate_melody_for_section(
        pattern, voicings, singer_range, bpm=bpm
    )
    return midi_bytes, pattern.name


def _generate_cello_voice(
    voicings: list[list[int]],
    energy: str,
    time_sig: tuple[int, int],
    bpm: int,
    rng: random.Random,
) -> tuple[bytes, str]:
    """Generate an independent cello line using BassPattern.

    Selects from bass templates appropriate for the energy level,
    clamps notes to the cello register (36–57).

    Returns (midi_bytes, pattern_name).
    """
    cello_register_max = INSTRUMENT_RANGES["cello"][1]  # 57
    cello_register_min = INSTRUMENT_RANGES["cello"][0]  # 36

    templates = select_bass_templates(ALL_BASS_TEMPLATES, time_sig, energy)
    if not templates:
        templates = [t for t in ALL_BASS_TEMPLATES if t.time_sig == time_sig] or list(
            ALL_BASS_TEMPLATES
        )

    pattern: BassPattern = rng.choice(templates)
    midi_bytes, _ = bass_pattern_to_midi_bytes(pattern, voicings, bpm=bpm)

    # Reclamp all notes from bass register (24-60) into cello register (36-57)
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    for track in mid.tracks:
        for msg in track:
            if msg.type in ("note_on", "note_off"):
                note = msg.note
                while note > cello_register_max:
                    note -= 12
                while note < cello_register_min:
                    note += 12
                msg.note = max(cello_register_min, min(cello_register_max, note))
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue(), pattern.name


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


def generate_quartet(
    production_dir: Path,
    section: str,
    singer: str = "gabriel",
    top_k: int = 3,
    seed: Optional[int] = None,
    scorer=None,
    simple_voicings: bool = False,
) -> list[dict]:
    """Generate string quartet candidates for one approved melody section.

    Violin I = approved melody (soprano).
    Violin II + Viola = independent melodic lines via MelodyPattern (chord-tone
        aware, instrument-register clamped, rhythmically independent).
    Cello = independent bass line via BassPattern (chord-tone based, cello register).

    Args:
        production_dir: Path to the song's production directory.
        section: Section label (e.g. 'chorus', 'verse').
        singer: Singer name — used only for display in review.yml.
        top_k: Number of candidates to generate.
        seed: Optional random seed.logicx for reproducibility.
        scorer: Optional pre-loaded Refractor instance.
        simple_voicings: If True, falls back to legacy interval-offset approach.

    Returns:
        List of candidate dicts, sorted by composite score descending.
    """
    rng = random.Random(seed)

    melody_approved = production_dir / "melody" / "approved"
    label_key = section.lower().replace("-", "_").replace(" ", "_")
    midi_path = melody_approved / f"{label_key}.mid"

    if not midi_path.exists():
        raise FileNotFoundError(
            f"No approved melody MIDI found for section '{section}' at {midi_path}"
        )

    # Song metadata
    ctx = load_song_context(production_dir)
    bpm = int(ctx.get("bpm", 120))
    color = str(ctx.get("color", "Red"))
    concept_text = ctx.get("concept", "") or f"{color} chromatic concept"
    time_sig_str = str(ctx.get("time_sig", "4/4"))
    ts_parts = time_sig_str.split("/")
    time_sig: tuple[int, int] = (int(ts_parts[0]), int(ts_parts[1]))

    soprano_midi_bytes = midi_path.read_bytes()

    # Load Refractor for chromatic scoring (optional)
    _scorer = scorer
    _concept_emb = None
    if _scorer is None:
        try:
            from white_analysis.refractor import Refractor

            _scorer = Refractor()
        except Exception:
            _scorer = None
    if _scorer is not None:
        try:
            _concept_emb = _scorer.prepare_concept(concept_text)
        except Exception:
            _scorer = None

    # Infer energy from section label
    energy = "high" if "chorus" in label_key or "hook" in label_key else "medium"
    if "intro" in label_key or "outro" in label_key or "bridge" in label_key:
        energy = "low"

    # --- Simple-voicings fallback (legacy interval-offset approach) ---
    if simple_voicings:
        soprano_notes = extract_soprano_notes(soprano_midi_bytes)
        note_events = extract_note_events(soprano_midi_bytes)
        triad_map = load_section_triad_map(production_dir, label_key)
        if triad_map:
            soprano_notes = snap_soprano_to_triads(soprano_notes, triad_map)
            print(
                f"  simple-voicings: snapped soprano to triads ({len(triad_map)} chords)"
            )

        alto_pats = get_patterns_for_voice("alto", energy)
        tenor_pats = get_patterns_for_voice("tenor", energy)
        bass_pats = get_patterns_for_voice("bass_voice", energy)
        rng.shuffle(alto_pats)
        rng.shuffle(tenor_pats)
        rng.shuffle(bass_pats)

        candidates = []
        for i in range(max(top_k * 2, len(alto_pats))):
            ap = alto_pats[i % len(alto_pats)]
            tp = tenor_pats[i % len(tenor_pats)]
            bp = bass_pats[i % len(bass_pats)]
            alto_raw = generate_voice_notes(soprano_notes, ap)
            tenor_raw = generate_voice_notes(soprano_notes, tp)
            bass_raw = generate_voice_notes(soprano_notes, bp)
            alto_fixed = resolve_parallel_violations(soprano_notes, alto_raw, "alto")
            tenor_fixed = resolve_parallel_violations(soprano_notes, tenor_raw, "tenor")
            bass_fixed = resolve_parallel_violations(
                soprano_notes, bass_raw, "bass_voice"
            )
            alto_final, tenor_final, bass_final = fix_voice_crossing(
                soprano_notes, alto_fixed, tenor_fixed, bass_fixed
            )
            midi_bytes = build_quartet_midi(
                note_events, alto_final, tenor_final, bass_final, bpm=bpm
            )
            scorer_result = None
            if _scorer is not None:
                try:
                    scorer_result = _scorer.score(midi_bytes, concept_emb=_concept_emb)
                except Exception:
                    pass
            scores = score_quartet_candidate(
                soprano_notes, alto_final, tenor_final, bass_final, scorer_result, color
            )
            candidates.append(
                {
                    "id": f"{label_key}_quartet_{i + 1:03d}",
                    "label": label_key,
                    "section": section,
                    "singer": singer,
                    "violin_ii_pattern": ap.name,
                    "viola_pattern": tp.name,
                    "cello_pattern": bp.name,
                    "approach": "simple_voicings",
                    "scores": scores,
                    "composite_score": scores["composite"],
                    "midi_bytes": midi_bytes,
                    "status": "pending",
                }
            )
        candidates.sort(key=lambda c: c["composite_score"], reverse=True)
        return candidates[:top_k]

    # --- Main path: melody patterns for violin II/viola, bass patterns for cello ---

    # Load chord voicings for this section (needed by violin II, viola, cello generators)
    try:
        chord_data, _ = extract_section_chord_data(production_dir)
        voicings = chord_data.get(label_key)
        if not voicings:
            # Try the first available section as a fallback
            voicings = next(iter(chord_data.values()), None)
    except Exception:
        voicings = None

    if not voicings:
        # Last resort: single C major triad
        voicings = [[48, 52, 55]]

    candidates = []
    for i in range(max(top_k * 2, 6)):
        # Violin II — melody pattern, violin II range
        vii_bytes, vii_pat = _generate_string_voice(
            "violin_ii", voicings, _VIOLIN_II_RANGE, energy, time_sig, bpm, rng
        )
        # Viola — melody pattern, viola range
        va_bytes, va_pat = _generate_string_voice(
            "viola", voicings, _VIOLA_RANGE, energy, time_sig, bpm, rng
        )
        # Cello — bass pattern, cello register
        vc_bytes, vc_pat = _generate_cello_voice(voicings, energy, time_sig, bpm, rng)

        midi_bytes = merge_voices_to_quartet_midi(
            [
                (soprano_midi_bytes, QUARTET_CHANNELS["violin_i"]),
                (vii_bytes, QUARTET_CHANNELS["violin_ii"]),
                (va_bytes, QUARTET_CHANNELS["viola"]),
                (vc_bytes, QUARTET_CHANNELS["cello"]),
            ],
            bpm=bpm,
        )

        scorer_result = None
        if _scorer is not None:
            try:
                scorer_result = _scorer.score(midi_bytes, concept_emb=_concept_emb)
            except Exception:
                pass

        target = get_chromatic_target(color)
        counterpoint_score = 0.8  # theory placeholder
        if scorer_result is not None:
            chromatic_val = compute_chromatic_match(scorer_result, target)
            confidence = float(scorer_result.get("confidence", 0.5))
            composite = round(0.30 * counterpoint_score + 0.70 * chromatic_val, 3)
        else:
            chromatic_val = None
            confidence = 0.0
            composite = round(counterpoint_score, 3)

        scores = {
            "counterpoint": counterpoint_score,
            "violin_ii_score": 0.8,
            "viola_score": 0.8,
            "cello_score": 0.8,
            "chromatic": chromatic_val,
            "confidence": confidence,
            "composite": composite,
        }

        candidates.append(
            {
                "id": f"{label_key}_quartet_{i + 1:03d}",
                "label": label_key,
                "section": section,
                "singer": singer,
                "violin_ii_pattern": vii_pat,
                "viola_pattern": va_pat,
                "cello_pattern": vc_pat,
                "approach": "string_quartet",
                "scores": scores,
                "composite_score": composite,
                "midi_bytes": midi_bytes,
                "status": "pending",
            }
        )

    candidates.sort(key=lambda c: c["composite_score"], reverse=True)
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------


def write_quartet_candidates(
    candidates: list[dict],
    production_dir: Path,
    section: str,
) -> tuple[Path, Path]:
    """Write MIDI files and review.yml for the given candidates.

    Returns (candidates_dir, review_path).
    """
    quartet_dir = production_dir / QUARTET_DIR_NAME
    candidates_dir = quartet_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    ctx = load_song_context(production_dir)
    color = str(ctx.get("color", ""))

    # Remove stale MIDI files for this section before writing new ones
    label_key = section.lower().replace("-", "_").replace(" ", "_")
    for stale in candidates_dir.glob(f"{label_key}_quartet_*.mid"):
        stale.unlink()

    review_entries = []
    for cand in candidates:
        filename = f"{cand['id']}.mid"
        (candidates_dir / filename).write_bytes(cand["midi_bytes"])

        review_entries.append(
            {
                "id": cand["id"],
                "label": cand["label"],
                "section": cand["section"],
                "singer": cand["singer"],
                "midi_file": f"candidates/{filename}",
                "violin_ii_pattern": cand.get("violin_ii_pattern", ""),
                "viola_pattern": cand.get("viola_pattern", ""),
                "cello_pattern": cand.get("cello_pattern", ""),
                "approach": cand.get("approach", "string_quartet"),
                "scores": _to_python(cand["scores"]),
                "composite_score": _to_python(cand["composite_score"]),
                "status": "pending",
            }
        )

    review_path = quartet_dir / REVIEW_FILENAME
    # Append to existing review.yml if present
    existing: dict = {}
    if review_path.exists():
        with open(review_path) as f:
            existing = yaml.safe_load(f) or {}

    existing_candidates = existing.get("candidates", [])
    # Replace entries for the same section label
    label_key = section.lower().replace("-", "_").replace(" ", "_")
    kept = [c for c in existing_candidates if c.get("label") != label_key]
    merged = kept + review_entries

    review_data = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "color": color,
        "phase": "quartet",
        "candidates": merged,
    }
    with open(review_path, "w") as f:
        yaml.dump(
            review_data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    return candidates_dir, review_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate four-part quartet voices from approved melody MIDI."
    )
    p.add_argument(
        "--production-dir", required=True, help="Path to production directory"
    )
    p.add_argument("--section", help="Section label to process (default: all approved)")
    p.add_argument("--singer", default="gabriel", help="Singer name (metadata only)")
    p.add_argument("--top-k", type=int, default=3, help="Candidates per section")
    p.add_argument("--seed.logicx", type=int, default=None, help="Random seed.logicx")
    p.add_argument(
        "--simple-voicings",
        action="store_true",
        help="Snap soprano notes to triad tones (root/3rd/5th) before harmonising, "
        "reducing dissonance from extended chord voicings.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    production_dir = Path(args.production_dir)
    if not production_dir.exists():
        print(
            f"ERROR: production directory not found: {production_dir}", file=sys.stderr
        )
        sys.exit(1)

    melody_approved = production_dir / "melody" / "approved"
    if not melody_approved.exists():
        print(
            f"ERROR: no melody/approved directory found in {production_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine sections to process
    if args.section:
        sections = [args.section]
    else:
        sections = [p.stem for p in sorted(melody_approved.glob("*.mid"))]
        if not sections:
            print("ERROR: no approved melody MIDI files found", file=sys.stderr)
            sys.exit(1)

    for section in sections:
        print(f"\nGenerating quartet for section: {section}")
        try:
            candidates = generate_quartet(
                production_dir,
                section=section,
                singer=args.singer,
                top_k=args.top_k,
                seed=args.seed,
                simple_voicings=args.simple_voicings,
            )
        except FileNotFoundError as e:
            print(f"  Skipped: {e}")
            continue

        candidates_dir, review_path = write_quartet_candidates(
            candidates, production_dir, section
        )
        print(f"  Wrote {len(candidates)} candidates to {candidates_dir}/")
        print(f"  Review: {review_path}")

    print(
        f"\nNext: Edit {production_dir / QUARTET_DIR_NAME / REVIEW_FILENAME} "
        "to approve candidates, then run promote_part."
    )


if __name__ == "__main__":
    main()
