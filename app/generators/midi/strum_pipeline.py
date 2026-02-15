#!/usr/bin/env python3
"""
Strum/rhythm generation pipeline for the Music Production Pipeline.

Takes approved chord MIDI files and applies rhythmic patterns to produce
variations — same harmony, different time feel. No ChromaticScorer needed
since the harmony is unchanged.

Usage:
    python -m app.generators.midi.strum_pipeline \
        --production-dir shrinkwrapped/.../production/black__sequential_dissolution_v2 \
        --mode per-chord
"""

import argparse
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

import mido
import yaml

from app.generators.midi.strum_patterns import StrumPattern, get_patterns_for_time_sig


# ---------------------------------------------------------------------------
# Harmonic rhythm reader — variable chord durations from upstream phase
# ---------------------------------------------------------------------------


def read_approved_harmonic_rhythm(production_dir: Path) -> dict[str, list[float]]:
    """Read approved harmonic rhythm durations from the harmonic_rhythm phase.

    Returns dict mapping section label → list of bar durations per chord.
    If no harmonic rhythm phase exists, returns empty dict (backward compatible).
    """
    hr_dir = production_dir / "harmonic_rhythm"
    review_path = hr_dir / "review.yml"

    if not review_path.exists():
        return {}

    with open(review_path) as f:
        review = yaml.safe_load(f)

    durations_by_section: dict[str, list[float]] = {}
    for candidate in review.get("candidates", []):
        status = str(candidate.get("status", "")).lower()
        if status not in ("approved", "accepted"):
            continue
        section = candidate.get("section", "")
        if not section:
            continue
        section_key = section.lower().replace("-", "_").replace(" ", "_")

        # Use first approved per section
        if section_key not in durations_by_section:
            dist = candidate.get("distribution")
            if dist and isinstance(dist, list):
                durations_by_section[section_key] = [float(d) for d in dist]

    return durations_by_section


# ---------------------------------------------------------------------------
# MIDI parsing — extract chord voicings from approved MIDI files
# ---------------------------------------------------------------------------


def parse_chord_voicings(midi_path: Path) -> list[dict]:
    """Parse an approved chord MIDI file and extract voicings per bar.

    Returns list of dicts: [{notes: [int, ...], velocity: int, bar_ticks: int}, ...]
    """
    mid = mido.MidiFile(str(midi_path))
    tpb = mid.ticks_per_beat

    # Collect note-on events with absolute ticks
    events = []
    abs_tick = 0
    for msg in mid.tracks[0]:
        abs_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            events.append(
                {"note": msg.note, "velocity": msg.velocity, "tick": abs_tick}
            )

    if not events:
        return []

    # Group notes by their onset tick (simultaneous notes = one chord)
    chords = []
    current_tick = events[0]["tick"]
    current_notes = []
    current_vel = events[0]["velocity"]

    for ev in events:
        if ev["tick"] != current_tick:
            chords.append(
                {
                    "notes": sorted(current_notes),
                    "velocity": current_vel,
                    "tick": current_tick,
                }
            )
            current_tick = ev["tick"]
            current_notes = []
            current_vel = ev["velocity"]
        current_notes.append(ev["note"])

    if current_notes:
        chords.append(
            {
                "notes": sorted(current_notes),
                "velocity": current_vel,
                "tick": current_tick,
            }
        )

    # Determine bar length from spacing between chords
    if len(chords) >= 2:
        bar_ticks = chords[1]["tick"] - chords[0]["tick"]
    else:
        # Single chord — assume 4 beats
        bar_ticks = tpb * 4

    for chord in chords:
        chord["bar_ticks"] = bar_ticks

    return chords


# ---------------------------------------------------------------------------
# Pattern application
# ---------------------------------------------------------------------------


def apply_strum_pattern(
    voicing: list[int],
    pattern: StrumPattern,
    velocity: int = 80,
    ticks_per_beat: int = 480,
) -> list[dict]:
    """Apply a strum pattern to a chord voicing, producing MIDI events for one bar.

    Returns list of {notes: [int], onset_tick: int, duration_ticks: int, velocity: int}.
    For arpeggios, each onset has a single note from the chord.
    For block patterns, each onset has all notes from the chord.
    """
    events = []

    if pattern.is_arpeggio:
        if not voicing:
            return events
        # Sort notes for direction
        sorted_notes = sorted(voicing)
        if pattern.arp_direction == "down":
            sorted_notes = list(reversed(sorted_notes))

        for i, (onset, duration) in enumerate(zip(pattern.onsets, pattern.durations)):
            # Cycle through chord tones
            note = sorted_notes[i % len(sorted_notes)]
            events.append(
                {
                    "notes": [note],
                    "onset_tick": int(onset * ticks_per_beat),
                    "duration_ticks": int(duration * ticks_per_beat),
                    "velocity": velocity,
                }
            )
    else:
        for onset, duration in zip(pattern.onsets, pattern.durations):
            events.append(
                {
                    "notes": list(voicing),
                    "onset_tick": int(onset * ticks_per_beat),
                    "duration_ticks": int(duration * ticks_per_beat),
                    "velocity": velocity,
                }
            )

    return events


def strum_to_midi_bytes(
    chords: list[list[int]],
    pattern: StrumPattern,
    bpm: int = 120,
    velocity: int = 80,
    ticks_per_beat: int = 480,
    durations: list[float] | None = None,
) -> bytes:
    """Apply a strum pattern to a sequence of chord voicings and produce MIDI bytes.

    Each chord gets one bar with the pattern applied, unless durations is provided.
    When durations is given, each chord gets its assigned duration in bars — the
    pattern repeats for longer chords and truncates for shorter ones.

    Args:
        durations: Optional list of bars per chord (from harmonic rhythm phase).
                   If None, each chord gets exactly 1.0 bar.
    """
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    bar_ticks = int(pattern.bar_length_beats() * ticks_per_beat)

    # Collect all events with absolute ticks
    all_events = []  # (abs_tick, note, velocity, is_on)

    current_offset = 0
    for chord_idx, voicing in enumerate(chords):
        if durations is not None:
            chord_dur_ticks = int(
                durations[chord_idx] * pattern.bar_length_beats() * ticks_per_beat
            )
        else:
            chord_dur_ticks = bar_ticks

        # Apply pattern repeatedly to fill the chord's duration
        pattern_offset = 0
        while pattern_offset < chord_dur_ticks:
            bar_events = apply_strum_pattern(voicing, pattern, velocity, ticks_per_beat)
            for ev in bar_events:
                abs_on = current_offset + pattern_offset + ev["onset_tick"]
                # Don't let events exceed this chord's boundary
                if abs_on >= current_offset + chord_dur_ticks:
                    break
                abs_off = min(
                    abs_on + ev["duration_ticks"],
                    current_offset + chord_dur_ticks,
                )
                for note in ev["notes"]:
                    all_events.append((abs_on, note, ev["velocity"], True))
                    all_events.append((abs_off, note, 0, False))
            pattern_offset += bar_ticks

        current_offset += chord_dur_ticks

    # Sort: by tick, then note-offs before note-ons at same tick
    all_events.sort(key=lambda e: (e[0], not e[3], e[1]))

    # Convert to delta times
    prev_tick = 0
    for abs_tick, note, vel, is_on in all_events:
        delta = abs_tick - prev_tick
        msg_type = "note_on" if is_on else "note_off"
        track.append(mido.Message(msg_type, note=note, velocity=vel, time=delta))
        prev_tick = abs_tick

    track.append(mido.MetaMessage("end_of_track", time=0))

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Review YAML generation
# ---------------------------------------------------------------------------


def generate_strum_review_yaml(
    production_dir: str,
    candidates: list[dict],
    song_info: dict,
) -> dict:
    review_candidates = []
    for i, cand in enumerate(candidates):
        review_candidates.append(
            {
                "id": cand["id"],
                "midi_file": f"candidates/{cand['id']}.mid",
                "rank": i + 1,
                "source_chord": cand["source_chord"],
                "pattern": cand["pattern_name"],
                "pattern_description": cand["pattern_description"],
                "is_arpeggio": cand["is_arpeggio"],
                "mode": cand["mode"],
                # Human annotation fields
                "label": None,
                "status": "pending",
                "notes": "",
            }
        )

    return {
        "production_dir": str(production_dir),
        "pipeline": "strum-generation",
        "bpm": song_info.get("bpm", 120),
        "time_sig": f"{song_info['time_sig'][0]}/{song_info['time_sig'][1]}",
        "generated": datetime.now(timezone.utc).isoformat(),
        "candidates": review_candidates,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_strum_pipeline(
    production_dir: str,
    mode: str = "per-chord",
    filter_patterns: list[str] | None = None,
):
    """Run the strum generation pipeline end-to-end.

    1. Read approved chords and song metadata from chord review.yml
    2. Parse chord voicings from MIDI files
    3. Apply each rhythm pattern to each chord (per-chord) or full progression (progression)
    4. Write candidates + review.yml
    """
    prod_path = Path(production_dir)
    if not prod_path.exists():
        print(f"ERROR: Production directory not found: {prod_path}")
        sys.exit(1)

    # --- 1. Read chord review metadata ---
    print("=" * 60)
    print("STRUM/RHYTHM GENERATION PIPELINE")
    print("=" * 60)

    chord_review_path = prod_path / "chords" / "review.yml"
    if not chord_review_path.exists():
        print(f"ERROR: Chord review not found: {chord_review_path}")
        sys.exit(1)

    with open(chord_review_path) as f:
        chord_review = yaml.safe_load(f)

    bpm = chord_review.get("bpm", 120)

    # Parse time sig — try chord review first, then fall back to song proposal
    time_sig = (4, 4)
    time_sig_str = chord_review.get("time_sig")
    if time_sig_str and "/" in str(time_sig_str):
        parts = str(time_sig_str).split("/")
        time_sig = (int(parts[0]), int(parts[1]))
    else:
        # Try loading from song proposal for accurate time sig
        thread_dir = chord_review.get("thread", "")
        song_filename = chord_review.get("song_proposal", "")
        if thread_dir and song_filename:
            try:
                from app.generators.midi.chord_pipeline import load_song_proposal

                proposal = load_song_proposal(Path(thread_dir), song_filename)
                time_sig = tuple(proposal["time_sig"])
                bpm = proposal.get("bpm", bpm)
            except Exception:
                pass

    print(f"BPM:  {bpm}")
    print(f"Time: {time_sig[0]}/{time_sig[1]}")
    print(f"Mode: {mode}")

    song_info = {"bpm": bpm, "time_sig": time_sig}

    # --- 2. Find approved chords ---
    approved_dir = prod_path / "chords" / "approved"
    if not approved_dir.exists():
        print(f"ERROR: No approved chords directory: {approved_dir}")
        sys.exit(1)

    midi_files = sorted(approved_dir.glob("*.mid"))
    if not midi_files:
        print("ERROR: No approved chord MIDI files found")
        sys.exit(1)

    print(f"Approved chords: {len(midi_files)}")
    for f in midi_files:
        print(f"  {f.name}")

    # Parse voicings from each file
    chord_data = {}  # label → [voicing_notes, ...]
    for midi_file in midi_files:
        label = midi_file.stem
        voicings = parse_chord_voicings(midi_file)
        chord_data[label] = [v["notes"] for v in voicings]
        print(
            f"  {label}: {len(voicings)} chords, {[len(v['notes']) for v in voicings]} notes each"
        )

    # --- 3. Read approved harmonic rhythm (optional) ---
    hr_durations = read_approved_harmonic_rhythm(prod_path)
    if hr_durations:
        print(f"\nHarmonic rhythm loaded for: {', '.join(hr_durations.keys())}")
        for sec, durs in hr_durations.items():
            print(f"  {sec}: {' | '.join(f'{d:.1f}' for d in durs)} bars")
    else:
        print("\nNo approved harmonic rhythm — using uniform 1-bar-per-chord")

    # --- 4. Get patterns for time signature ---
    patterns = get_patterns_for_time_sig(time_sig, filter_patterns)
    print(f"\nPatterns: {len(patterns)}")
    for p in patterns:
        arp_tag = " [arp]" if p.is_arpeggio else ""
        print(f"  {p.name}{arp_tag}: {p.description}")

    # --- 5. Generate candidates ---
    candidates = []

    if mode == "per-chord":
        for label, voicings in chord_data.items():
            # Look up harmonic rhythm durations for this section
            section_durations = hr_durations.get(label, None)
            if section_durations and len(section_durations) != len(voicings):
                print(
                    f"  Warning: HR durations ({len(section_durations)}) != voicings ({len(voicings)}) for {label}, ignoring HR"
                )
                section_durations = None

            for pattern in patterns:
                candidate_id = f"{label}_{pattern.name}"
                midi_bytes = strum_to_midi_bytes(
                    voicings, pattern, bpm=bpm, durations=section_durations
                )
                candidates.append(
                    {
                        "id": candidate_id,
                        "midi_bytes": midi_bytes,
                        "source_chord": label,
                        "pattern_name": pattern.name,
                        "pattern_description": pattern.description,
                        "is_arpeggio": pattern.is_arpeggio,
                        "mode": "per-chord",
                    }
                )
    elif mode == "progression":
        # Concatenate all approved chords in file order
        all_voicings = []
        all_labels = []
        all_durations = []
        has_hr = bool(hr_durations)
        for label, voicings in chord_data.items():
            all_voicings.extend(voicings)
            all_labels.append(label)
            sec_durs = hr_durations.get(label)
            if sec_durs and len(sec_durs) == len(voicings):
                all_durations.extend(sec_durs)
            else:
                all_durations.extend([1.0] * len(voicings))
                has_hr = False  # mixed — can't guarantee consistency
        progression_label = "+".join(all_labels)
        prog_durations = all_durations if has_hr else None

        for pattern in patterns:
            candidate_id = f"progression_{pattern.name}"
            midi_bytes = strum_to_midi_bytes(
                all_voicings, pattern, bpm=bpm, durations=prog_durations
            )
            candidates.append(
                {
                    "id": candidate_id,
                    "midi_bytes": midi_bytes,
                    "source_chord": progression_label,
                    "pattern_name": pattern.name,
                    "pattern_description": pattern.description,
                    "is_arpeggio": pattern.is_arpeggio,
                    "mode": "progression",
                }
            )
    elif mode == "both":
        # Per-chord candidates
        for label, voicings in chord_data.items():
            section_durations = hr_durations.get(label, None)
            if section_durations and len(section_durations) != len(voicings):
                section_durations = None

            for pattern in patterns:
                candidate_id = f"{label}_{pattern.name}"
                midi_bytes = strum_to_midi_bytes(
                    voicings, pattern, bpm=bpm, durations=section_durations
                )
                candidates.append(
                    {
                        "id": candidate_id,
                        "midi_bytes": midi_bytes,
                        "source_chord": label,
                        "pattern_name": pattern.name,
                        "pattern_description": pattern.description,
                        "is_arpeggio": pattern.is_arpeggio,
                        "mode": "per-chord",
                    }
                )

        # Progression candidates
        all_voicings = []
        all_labels = []
        all_durations = []
        has_hr = bool(hr_durations)
        for label, voicings in chord_data.items():
            all_voicings.extend(voicings)
            all_labels.append(label)
            sec_durs = hr_durations.get(label)
            if sec_durs and len(sec_durs) == len(voicings):
                all_durations.extend(sec_durs)
            else:
                all_durations.extend([1.0] * len(voicings))
                has_hr = False
        progression_label = "+".join(all_labels)
        prog_durations = all_durations if has_hr else None

        for pattern in patterns:
            candidate_id = f"progression_{pattern.name}"
            midi_bytes = strum_to_midi_bytes(
                all_voicings, pattern, bpm=bpm, durations=prog_durations
            )
            candidates.append(
                {
                    "id": candidate_id,
                    "midi_bytes": midi_bytes,
                    "source_chord": progression_label,
                    "pattern_name": pattern.name,
                    "pattern_description": pattern.description,
                    "is_arpeggio": pattern.is_arpeggio,
                    "mode": "progression",
                }
            )

    print(f"\nGenerated {len(candidates)} candidates")

    # --- 5. Write MIDI files ---
    strums_dir = prod_path / "strums"
    candidates_dir = strums_dir / "candidates"
    approved_dir = strums_dir / "approved"
    # Clean old candidates to avoid stale files from previous runs
    if candidates_dir.exists():
        for old_file in candidates_dir.glob("*.mid"):
            old_file.unlink()
    candidates_dir.mkdir(parents=True, exist_ok=True)
    approved_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing {len(candidates)} MIDI files to {candidates_dir}/")
    for cand in candidates:
        path = candidates_dir / f"{cand['id']}.mid"
        path.write_bytes(cand["midi_bytes"])

    # --- 6. Write review YAML ---
    review = generate_strum_review_yaml(production_dir, candidates, song_info)
    review_path = strums_dir / "review.yml"
    with open(review_path, "w") as f:
        yaml.dump(
            review, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    # --- 7. Summary ---
    print(f"\n{'=' * 60}")
    print("STRUM GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Chords:     {len(chord_data)}")
    print(f"Patterns:   {len(patterns)}")
    print(f"Candidates: {len(candidates)}")
    print(f"Review:     {review_path}")
    print(f"\nNext: Edit {review_path} to label and approve candidates")
    print(f"Then: python -m app.generators.midi.promote_chords --review {review_path}")

    return candidates


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Strum/rhythm generation pipeline — apply rhythm patterns to approved chords"
    )
    parser.add_argument(
        "--production-dir",
        required=True,
        help="Song production directory (must contain chords/approved/)",
    )
    parser.add_argument(
        "--mode",
        choices=["per-chord", "progression", "both"],
        default="per-chord",
        help="Generation mode (default: per-chord)",
    )
    parser.add_argument(
        "--patterns",
        default=None,
        help="Comma-separated pattern names to include (default: all)",
    )

    args = parser.parse_args()

    filter_patterns = None
    if args.patterns:
        filter_patterns = [p.strip() for p in args.patterns.split(",")]

    run_strum_pipeline(
        production_dir=args.production_dir,
        mode=args.mode,
        filter_patterns=filter_patterns,
    )


if __name__ == "__main__":
    main()
