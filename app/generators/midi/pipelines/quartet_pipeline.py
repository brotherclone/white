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
import mido
import yaml

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.generators.midi.patterns.quartet_patterns import (
    QUARTET_CHANNELS,
    VELOCITY,
    VoicePattern,
    check_parallels,
    clamp_to_voice_range,
    counterpoint_score,
    fix_voice_crossing,
    get_patterns_for_voice,
)
from app.generators.midi.pipelines.chord_pipeline import (
    compute_chromatic_match,
    get_chromatic_target,
)
from app.generators.midi.production.init_production import load_song_context


QUARTET_DIR_NAME = "quartet"
REVIEW_FILENAME = "review.yml"
MAX_OFFSET_CHANGE = 4  # semitones — cap per-beat leap in a generated voice


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

    Duration is estimated as the gap to the next note-on or note-off event.
    Returns events sorted by abs_tick.
    """
    mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
    tpb = mid.ticks_per_beat or 480

    on_events: list[tuple[int, int]] = []  # (abs_tick, pitch)
    off_events: dict[int, int] = {}  # pitch → first off abs_tick

    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                on_events.append((abs_tick, msg.note))
            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                if msg.note not in off_events:
                    off_events[msg.note] = abs_tick

    on_events.sort(key=lambda x: x[0])

    result = []
    for i, (tick, pitch) in enumerate(on_events):
        if i + 1 < len(on_events):
            default_dur = on_events[i + 1][0] - tick
        else:
            default_dur = tpb  # quarter note fallback
        dur = max(off_events.get(pitch, tick + default_dur) - tick, tpb // 4)
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
    else:
        chromatic_val = 0.5
        confidence = 0.0

    composite = round(0.30 * mean_cp + 0.70 * chromatic_val, 3)

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
# Main generation function
# ---------------------------------------------------------------------------


def generate_quartet(
    production_dir: Path,
    section: str,
    singer: str = "gabriel",
    top_k: int = 3,
    seed: Optional[int] = None,
    scorer=None,
) -> list[dict]:
    """Generate quartet candidates for one approved melody section.

    Reads `melody/approved/<section>.mid`, generates `top_k` candidates
    (each using a different combination of alto/tenor/bass-voice templates),
    and returns a list of candidate dicts ready for review.yml serialisation.

    Args:
        production_dir: Path to the song's production directory.
        section: Section label (e.g. 'chorus', 'verse').
        singer: Singer name — used only for display in review.yml.
        top_k: Number of candidates to generate.
        seed: Optional random seed for reproducibility.
        scorer: Optional pre-loaded Refractor instance. If None, one is loaded
            automatically; if Refractor is unavailable, chromatic scoring is skipped.

    Returns:
        List of candidate dicts, sorted by composite score descending.
    """
    if seed is not None:
        random.seed(seed)

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

    soprano_midi_bytes = midi_path.read_bytes()
    soprano_notes = extract_soprano_notes(soprano_midi_bytes)
    note_events = extract_note_events(soprano_midi_bytes)

    # Load Refractor for chromatic scoring (optional — falls back gracefully)
    _scorer = scorer
    _concept_emb = None
    if _scorer is None:
        try:
            from training.refractor import Refractor

            _scorer = Refractor()
        except Exception:
            _scorer = None
    if _scorer is not None:
        try:
            _concept_emb = _scorer.prepare_concept(concept_text)
        except Exception:
            _scorer = None

    if not soprano_notes:
        raise ValueError(f"No notes found in melody MIDI for section '{section}'")

    # Infer energy from section label
    energy = "high" if "chorus" in label_key or "hook" in label_key else "medium"
    if "intro" in label_key or "outro" in label_key or "bridge" in label_key:
        energy = "low"

    alto_pats = get_patterns_for_voice("alto", energy)
    tenor_pats = get_patterns_for_voice("tenor", energy)
    bass_pats = get_patterns_for_voice("bass_voice", energy)

    # Shuffle patterns and build candidates from combinations
    random.shuffle(alto_pats)
    random.shuffle(tenor_pats)
    random.shuffle(bass_pats)

    candidates = []
    combo_count = max(top_k * 2, len(alto_pats))  # over-generate then prune

    for i in range(combo_count):
        ap = alto_pats[i % len(alto_pats)]
        tp = tenor_pats[i % len(tenor_pats)]
        bp = bass_pats[i % len(bass_pats)]

        alto_raw = generate_voice_notes(soprano_notes, ap)
        tenor_raw = generate_voice_notes(soprano_notes, tp)
        bass_raw = generate_voice_notes(soprano_notes, bp)

        # Resolve parallel violations
        alto_fixed = resolve_parallel_violations(soprano_notes, alto_raw, "alto")
        tenor_fixed = resolve_parallel_violations(soprano_notes, tenor_raw, "tenor")
        bass_fixed = resolve_parallel_violations(soprano_notes, bass_raw, "bass_voice")

        # Fix voice crossing
        alto_final, tenor_final, bass_final = fix_voice_crossing(
            soprano_notes, alto_fixed, tenor_fixed, bass_fixed
        )

        midi_bytes = build_quartet_midi(
            note_events,
            alto_final,
            tenor_final,
            bass_final,
            bpm=bpm,
        )

        # Chromatic scoring via Refractor
        scorer_result: Optional[dict] = None
        if _scorer is not None:
            try:
                scorer_result = _scorer.score(midi_bytes, concept_emb=_concept_emb)
            except Exception:
                scorer_result = None

        scores = score_quartet_candidate(
            soprano_notes,
            alto_final,
            tenor_final,
            bass_final,
            scorer_result,
            color,
        )

        cand_id = f"{label_key}_quartet_{i + 1:03d}"
        candidates.append(
            {
                "id": cand_id,
                "label": label_key,
                "section": section,
                "singer": singer,
                "alto_pattern": ap.name,
                "tenor_pattern": tp.name,
                "bass_voice_pattern": bp.name,
                "scores": scores,
                "composite_score": scores["composite"],
                "midi_bytes": midi_bytes,
                "status": "pending",
            }
        )

    # Sort by composite, keep top_k
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
                "file": filename,
                "alto_pattern": cand["alto_pattern"],
                "tenor_pattern": cand["tenor_pattern"],
                "bass_voice_pattern": cand["bass_voice_pattern"],
                "scores": cand["scores"],
                "composite_score": cand["composite_score"],
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
    p.add_argument("--seed", type=int, default=None, help="Random seed")
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
