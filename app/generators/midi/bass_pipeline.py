#!/usr/bin/env python3
"""
Bass line generation pipeline for the Music Production Pipeline.

Reads approved chords, harmonic rhythm, and drum patterns. Generates bass line
candidates from templates, scores with theory + ChromaticScorer composite,
writes top candidates as MIDI files with a review YAML.

Pipeline position: chords → drums → harmonic rhythm → strums → BASS → melody

Usage:
    python -m app.generators.midi.bass_pipeline \
        --production-dir shrinkwrapped/.../production/black__sequential_dissolution_v2 \
        --seed 42 --top-k 5
"""

import argparse
import io
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mido
import numpy as np
import yaml

from app.generators.midi.bass_patterns import (
    ALL_TEMPLATES,
    BASS_CHANNEL,
    VELOCITY,
    BassPattern,
    bass_theory_score,
    extract_root,
    kick_alignment,
    make_fallback_pattern,
    resolve_tone,
    root_adherence,
    select_templates,
    voice_leading_score,
)
from app.generators.midi.chord_pipeline import (
    _to_python,
    compute_chromatic_match,
    get_chromatic_target,
    load_song_proposal,
)
from app.generators.midi.strum_pipeline import (
    parse_chord_voicings,
    read_approved_harmonic_rhythm,
)


# ---------------------------------------------------------------------------
# Kick onset extraction from drum MIDI
# ---------------------------------------------------------------------------

KICK_NOTE = 36  # GM percussion kick drum


def extract_kick_onsets(
    drum_midi_path: str,
    ticks_per_beat: int = 480,
) -> list[float]:
    """Extract kick drum onset positions (in beats) from an approved drum MIDI.

    Returns list of beat positions where kick hits occur within one bar.
    """
    mid = mido.MidiFile(str(drum_midi_path))
    tpb = mid.ticks_per_beat or ticks_per_beat

    onsets = []
    abs_tick = 0
    for msg in mid.tracks[0]:
        abs_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0 and msg.note == KICK_NOTE:
            beat_pos = abs_tick / tpb
            onsets.append(beat_pos)

    return onsets


def read_approved_kick_onsets(
    production_dir: Path,
) -> dict[str, list[float]]:
    """Read approved drum MIDI files and extract kick onsets per section.

    Returns dict mapping section label → list of kick beat positions.
    """
    drums_dir = production_dir / "drums"
    approved_dir = drums_dir / "approved"
    review_path = drums_dir / "review.yml"

    if not approved_dir.exists() or not review_path.exists():
        return {}

    with open(review_path) as f:
        drum_review = yaml.safe_load(f)

    section_kicks: dict[str, list[float]] = {}

    for candidate in drum_review.get("candidates", []):
        status = str(candidate.get("status", "")).lower()
        if status not in ("approved", "accepted"):
            continue
        section = candidate.get("section", "")
        label = candidate.get("label", "")
        if not section and not label:
            continue

        section_key = (section or label).lower().replace("-", "_").replace(" ", "_")

        # Only use first approved drum per section
        if section_key in section_kicks:
            continue

        # Find the approved MIDI file
        midi_file = None
        if label:
            candidate_path = approved_dir / f"{label}.mid"
            if candidate_path.exists():
                midi_file = str(candidate_path)
        if midi_file is None:
            for f in sorted(approved_dir.glob("*.mid")):
                if f.stem.lower().startswith(section_key):
                    midi_file = str(f)
                    break

        if midi_file is not None:
            section_kicks[section_key] = extract_kick_onsets(midi_file)

    return section_kicks


# ---------------------------------------------------------------------------
# Chord root extraction
# ---------------------------------------------------------------------------


def extract_section_chord_data(
    production_dir: Path,
) -> tuple[dict[str, list[list[int]]], dict]:
    """Read approved chord voicings and review metadata.

    Returns:
        chord_data: dict mapping section label → list of voicings (note lists)
        chord_review: the parsed chord review.yml dict
    """
    chord_review_path = production_dir / "chords" / "review.yml"
    if not chord_review_path.exists():
        raise FileNotFoundError(f"Chord review not found: {chord_review_path}")

    with open(chord_review_path) as f:
        chord_review = yaml.safe_load(f)

    approved_dir = production_dir / "chords" / "approved"
    if not approved_dir.exists():
        raise FileNotFoundError(f"No approved chords: {approved_dir}")

    midi_files = sorted(approved_dir.glob("*.mid"))
    if not midi_files:
        raise FileNotFoundError("No approved chord MIDI files found")

    chord_data: dict[str, list[list[int]]] = {}
    for midi_file in midi_files:
        label = midi_file.stem
        voicings = parse_chord_voicings(midi_file)
        chord_data[label] = [v["notes"] for v in voicings]

    return chord_data, chord_review


# ---------------------------------------------------------------------------
# Bass MIDI generation
# ---------------------------------------------------------------------------


def bass_pattern_to_midi_bytes(
    pattern: BassPattern,
    voicings: list[list[int]],
    bpm: int = 120,
    ticks_per_beat: int = 480,
    durations: list[float] | None = None,
) -> tuple[bytes, list[tuple[float, int]]]:
    """Generate bass MIDI bytes from a pattern and chord voicings.

    Args:
        pattern: The bass pattern template.
        voicings: List of chord voicings (one per chord in the section).
        bpm: Beats per minute.
        ticks_per_beat: MIDI resolution.
        durations: Optional per-chord durations in bars (from harmonic rhythm).

    Returns:
        Tuple of (midi_bytes, resolved_notes) where resolved_notes is
        [(beat_position_absolute, midi_note), ...] for scoring.
    """
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    bar_beats = pattern.bar_length_beats()
    bar_ticks = int(bar_beats * ticks_per_beat)

    all_events = []  # (abs_tick, note, velocity, is_on)
    resolved_notes = []  # (abs_beat_position, midi_note) for scoring

    current_offset_ticks = 0
    for chord_idx, voicing in enumerate(voicings):
        if durations is not None:
            chord_dur_beats = durations[chord_idx] * bar_beats
        else:
            chord_dur_beats = bar_beats

        chord_dur_ticks = int(chord_dur_beats * ticks_per_beat)

        # Determine next voicing for approach/passing tones
        next_voicing = (
            voicings[chord_idx + 1] if chord_idx + 1 < len(voicings) else None
        )

        # Apply pattern, repeating if chord duration > 1 bar
        pattern_offset_ticks = 0
        while pattern_offset_ticks < chord_dur_ticks:
            for i, (beat_pos, tone_sel, vel_level) in enumerate(pattern.notes):
                note = resolve_tone(tone_sel, voicing, next_voicing)
                velocity = VELOCITY.get(vel_level, 80)

                onset_ticks = pattern_offset_ticks + int(beat_pos * ticks_per_beat)
                abs_ticks = current_offset_ticks + onset_ticks

                # Don't exceed chord boundary
                if onset_ticks >= chord_dur_ticks:
                    break

                # Determine note duration
                if pattern.note_durations and i < len(pattern.note_durations):
                    dur_ticks = int(pattern.note_durations[i] * ticks_per_beat)
                elif i + 1 < len(pattern.notes):
                    next_onset = int(pattern.notes[i + 1][0] * ticks_per_beat)
                    dur_ticks = pattern_offset_ticks + next_onset - onset_ticks
                else:
                    dur_ticks = chord_dur_ticks - onset_ticks

                # Clamp duration to not exceed chord boundary
                max_dur = current_offset_ticks + chord_dur_ticks - abs_ticks
                dur_ticks = min(dur_ticks, max_dur)

                if dur_ticks > 0:
                    all_events.append((abs_ticks, note, velocity, True))
                    all_events.append((abs_ticks + dur_ticks, note, 0, False))

                    abs_beat = (current_offset_ticks + onset_ticks) / ticks_per_beat
                    resolved_notes.append((abs_beat, note))

            pattern_offset_ticks += bar_ticks

        current_offset_ticks += chord_dur_ticks

    # Sort: by tick, note-offs before note-ons at same tick
    all_events.sort(key=lambda e: (e[0], not e[3], e[1]))

    # Convert to delta times
    prev_tick = 0
    for abs_tick, note, velocity, is_on in all_events:
        delta = abs_tick - prev_tick
        msg_type = "note_on" if is_on else "note_off"
        track.append(
            mido.Message(
                msg_type, note=note, velocity=velocity, time=delta, channel=BASS_CHANNEL
            )
        )
        prev_tick = abs_tick

    track.append(mido.MetaMessage("end_of_track", time=0))

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue(), resolved_notes


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def bass_composite_score(
    theory: float,
    chromatic_match: float,
    scorer_result: dict,
    theory_breakdown: dict,
    theory_weight: float = 0.3,
    chromatic_weight: float = 0.7,
) -> tuple[float, dict]:
    """Compute weighted composite score for a bass candidate."""
    composite = theory_weight * theory + chromatic_weight * chromatic_match
    breakdown = {
        "composite": round(composite, 4),
        "theory": {k: round(v, 4) for k, v in theory_breakdown.items()},
        "chromatic": {
            "temporal": {m: round(v, 4) for m, v in scorer_result["temporal"].items()},
            "spatial": {m: round(v, 4) for m, v in scorer_result["spatial"].items()},
            "ontological": {
                m: round(v, 4) for m, v in scorer_result["ontological"].items()
            },
            "confidence": round(scorer_result["confidence"], 4),
            "match": round(chromatic_match, 4),
        },
    }
    return composite, breakdown


# ---------------------------------------------------------------------------
# Review YAML generation
# ---------------------------------------------------------------------------


def generate_bass_review_yaml(
    production_dir: str,
    sections: list[dict],
    ranked_by_section: dict[str, list[dict]],
    seed: int,
    scoring_weights: dict,
    song_info: dict,
) -> dict:
    """Generate the review YAML structure for bass candidates."""
    all_candidates = []
    global_rank = 0
    for section in sections:
        section_key = section["_section_key"]
        for item in ranked_by_section.get(section_key, []):
            global_rank += 1
            all_candidates.append(
                {
                    "id": item["id"],
                    "midi_file": f"candidates/{item['id']}.mid",
                    "rank": global_rank,
                    "section": section["label_display"],
                    "chord_source": section.get("chord_id", ""),
                    "style": item["style"],
                    "pattern_name": item["pattern_name"],
                    "energy": item["energy"],
                    "scores": _to_python(item["breakdown"]),
                    # Human annotation fields
                    "label": None,
                    "status": "pending",
                    "notes": "",
                }
            )

    return {
        "production_dir": str(production_dir),
        "pipeline": "bass-generation",
        "bpm": song_info.get("bpm", 120),
        "time_sig": f"{song_info['time_sig'][0]}/{song_info['time_sig'][1]}",
        "color": song_info.get("color_name", ""),
        "generated": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "scoring_weights": scoring_weights,
        "sections_found": [s["label_display"] for s in sections],
        "candidates": all_candidates,
    }


# ---------------------------------------------------------------------------
# Read approved sections from chord review
# ---------------------------------------------------------------------------


def read_approved_sections(chord_review: dict) -> list[dict]:
    """Extract approved sections from chord review YAML.

    Returns list of dicts: [{label, label_display, chord_id}, ...]
    """
    sections = []
    for candidate in chord_review.get("candidates", []):
        status = str(candidate.get("status", "")).lower()
        if status not in ("approved", "accepted"):
            continue
        label = candidate.get("label")
        if not label:
            continue
        sections.append(
            {
                "label": label.lower().replace("-", "_").replace(" ", "_"),
                "label_display": label,
                "chord_id": candidate.get("id", "unknown"),
            }
        )
    return sections


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_bass_pipeline(
    production_dir: str,
    thread_dir: Optional[str] = None,
    song_filename: Optional[str] = None,
    seed: int = 42,
    top_k: int = 5,
    theory_weight: float = 0.3,
    chromatic_weight: float = 0.7,
    onnx_path: Optional[str] = None,
):
    """Run the bass generation pipeline end-to-end.

    1. Read approved chords, harmonic rhythm, drums
    2. Generate bass candidates from templates
    3. Score with theory + ChromaticScorer
    4. Write top-k per section as MIDI files
    5. Write review.yml
    """
    np.random.seed(seed)

    prod_path = Path(production_dir)
    if not prod_path.exists():
        print(f"ERROR: Production directory not found: {prod_path}")
        sys.exit(1)

    # --- 1. Read approved chords ---
    print("=" * 60)
    print("BASS LINE GENERATION PIPELINE")
    print("=" * 60)

    chord_data, chord_review = extract_section_chord_data(prod_path)

    sections = read_approved_sections(chord_review)
    if not sections:
        print("ERROR: No approved chord sections found in chords/review.yml")
        sys.exit(1)

    print(f"Sections: {', '.join(s['label_display'] for s in sections)}")

    # --- 2. Load song info ---
    song_info = {
        "bpm": chord_review.get("bpm", 120),
        "color_name": chord_review.get("color", "White"),
        "concept": "",
        "time_sig": (4, 4),
    }

    thread_from_review = chord_review.get("thread", "")
    song_from_review = chord_review.get("song_proposal", "")
    t_dir = thread_dir or thread_from_review
    s_file = song_filename or song_from_review

    if t_dir and s_file:
        try:
            song_info = load_song_proposal(Path(t_dir), s_file)
        except Exception as e:
            print(f"  Warning: Could not load song proposal: {e}")

    time_sig_str = chord_review.get("time_sig")
    if time_sig_str and "/" in str(time_sig_str):
        parts = str(time_sig_str).split("/")
        song_info["time_sig"] = (int(parts[0]), int(parts[1]))

    time_sig = tuple(song_info["time_sig"])
    bpm = song_info["bpm"]

    print(f"BPM:   {bpm}")
    print(f"Time:  {time_sig[0]}/{time_sig[1]}")
    print(f"Color: {song_info['color_name']}")

    # --- 3. Read harmonic rhythm ---
    hr_durations = read_approved_harmonic_rhythm(prod_path)
    if hr_durations:
        print(f"Harmonic rhythm loaded for: {', '.join(hr_durations.keys())}")
    else:
        print("  No approved harmonic rhythm — using 1 bar per chord")

    # --- 4. Read approved drum kick onsets ---
    kick_onsets = read_approved_kick_onsets(prod_path)
    if kick_onsets:
        print(f"Kick onsets loaded for: {', '.join(kick_onsets.keys())}")
    else:
        print("  No approved drums — kick alignment scoring disabled")

    # --- 5. Load ChromaticScorer ---
    print("\nLoading ChromaticScorer...")
    from training.chromatic_scorer import ChromaticScorer

    scorer = ChromaticScorer(onnx_path=onnx_path) if onnx_path else ChromaticScorer()

    concept_text = song_info.get("concept", "")
    if not concept_text:
        concept_text = f"{song_info['color_name']} chromatic concept"
        print(f"  Warning: No concept text, using fallback: '{concept_text}'")
    concept_emb = scorer.prepare_concept(concept_text)
    print(f"  Concept encoded ({concept_emb.shape[0]}-dim)")

    target = get_chromatic_target(song_info["color_name"])

    # --- 6. Disambiguate section labels ---
    label_occurrence: dict[str, int] = {}
    for section in sections:
        label = section["label"]
        label_occurrence[label] = label_occurrence.get(label, 0) + 1
        section["_occurrence"] = label_occurrence[label]

    duplicate_labels = {k for k, v in label_occurrence.items() if v > 1}
    for section in sections:
        label = section["label"]
        if label in duplicate_labels:
            section["_section_key"] = f"{label}_{section['_occurrence']}"
        else:
            section["_section_key"] = label

    # --- 7. Generate and score per section ---
    ranked_by_section: dict[str, list[dict]] = {}
    all_midi_outputs: list[tuple[str, bytes]] = []

    for section in sections:
        section_key = section["_section_key"]
        label = section["label"]
        label_display = section["label_display"]

        voicings = chord_data.get(label, [])
        if not voicings:
            # Try section_key
            voicings = chord_data.get(section_key, [])
        if not voicings:
            # Try any key starting with label
            for k, v in chord_data.items():
                if k.lower().startswith(label):
                    voicings = v
                    break

        if not voicings:
            print(f"\n--- Section: {label_display} — SKIPPED (no chord voicings) ---")
            continue

        # Section durations from harmonic rhythm
        section_durations = hr_durations.get(label, None)
        if section_durations and len(section_durations) != len(voicings):
            print(
                f"  Warning: HR durations ({len(section_durations)}) != voicings ({len(voicings)}) for {label}, ignoring HR"
            )
            section_durations = None

        # Section kick onsets
        section_kicks = kick_onsets.get(label, None)
        if section_kicks is None:
            section_kicks = kick_onsets.get(section_key, None)
        if section_kicks is None:
            for dk, dv in kick_onsets.items():
                if dk.startswith(label):
                    section_kicks = dv
                    break

        print(
            f"\n--- Section: {label_display} [{section_key}] ({len(voicings)} chords) ---"
        )

        # Select templates
        # Use section energy mapping similar to drums
        DEFAULT_ENERGY = {
            "intro": "low",
            "verse": "medium",
            "chorus": "high",
            "bridge": "low",
            "outro": "medium",
        }
        target_energy = DEFAULT_ENERGY.get(label, "medium")

        templates = select_templates(ALL_TEMPLATES, time_sig, target_energy)
        if not templates:
            print(f"  No templates for {time_sig} — using fallback")
            templates = [make_fallback_pattern(time_sig)]

        print(
            f"  Templates: {len(templates)} candidates (energy target: {target_energy})"
        )

        # Generate MIDI and compute theory scores for each template
        candidates = []
        for tmpl in templates:
            midi_bytes, resolved_notes = bass_pattern_to_midi_bytes(
                tmpl, voicings, bpm=bpm, durations=section_durations
            )

            # Theory scoring
            chord_root = extract_root(voicings[0]) if voicings else 36
            ra = root_adherence(resolved_notes, chord_root, time_sig)
            ka = (
                kick_alignment(
                    [pos for pos, _ in resolved_notes],
                    section_kicks,
                )
                if section_kicks
                else None
            )

            # Voice leading: intervals between last note of each chord and first of next
            vl_intervals = []
            if len(voicings) > 1:
                for ci in range(len(voicings) - 1):
                    r1 = extract_root(voicings[ci])
                    r2 = extract_root(voicings[ci + 1])
                    vl_intervals.append(abs(r2 - r1))
            vl = voice_leading_score(vl_intervals)

            theory = bass_theory_score(ra, ka, vl)
            theory_breakdown = {
                "root_adherence": ra,
                "voice_leading": vl,
            }
            if ka is not None:
                theory_breakdown["kick_alignment"] = ka

            candidates.append(
                {
                    "template": tmpl,
                    "midi_bytes": midi_bytes,
                    "resolved_notes": resolved_notes,
                    "theory": theory,
                    "theory_breakdown": theory_breakdown,
                    "pattern_name": tmpl.name,
                    "style": tmpl.style,
                    "energy": tmpl.energy,
                }
            )

        # Score with ChromaticScorer
        print(f"  Scoring {len(candidates)} candidates...")
        scorer_candidates = [{"midi_bytes": c["midi_bytes"]} for c in candidates]
        scorer_results = scorer.score_batch(scorer_candidates, concept_emb=concept_emb)

        scorer_by_midi = {}
        for result in scorer_results:
            midi_key = id(result["candidate"]["midi_bytes"])
            scorer_by_midi[midi_key] = result

        # Composite scoring
        scored = []
        for i, cand in enumerate(candidates):
            midi_key = id(scorer_candidates[i]["midi_bytes"])
            scorer_result = scorer_by_midi.get(midi_key)
            if scorer_result is None:
                continue

            chromatic_match = compute_chromatic_match(scorer_result, target)
            comp, breakdown = bass_composite_score(
                cand["theory"],
                chromatic_match,
                scorer_result,
                cand["theory_breakdown"],
                theory_weight,
                chromatic_weight,
            )
            scored.append(
                {
                    "composite": comp,
                    "breakdown": breakdown,
                    "midi_bytes": cand["midi_bytes"],
                    "pattern_name": cand["pattern_name"],
                    "style": cand["style"],
                    "energy": cand["energy"],
                    "description": cand["template"].description,
                }
            )

        # Rank and take top-k
        scored.sort(key=lambda x: x["composite"], reverse=True)
        top = scored[:top_k]

        for rank, item in enumerate(top):
            item["rank"] = rank + 1
            item["id"] = f"bass_{section_key}_{rank + 1:02d}"
            all_midi_outputs.append((f"{item['id']}.mid", item["midi_bytes"]))

        ranked_by_section[section_key] = top

        # Print summary
        for item in top:
            theory_str = " ".join(
                f"{k}={v:.2f}" for k, v in item["breakdown"]["theory"].items()
            )
            print(
                f"  #{item['rank']} [{item['id']}] {item['pattern_name']:25s} "
                f"composite={item['breakdown']['composite']:.3f} "
                f"theory=[{theory_str}] "
                f"chromatic={item['breakdown']['chromatic']['match']:.3f}"
            )

    # --- 8. Write MIDI files ---
    bass_dir = prod_path / "bass"
    candidates_dir = bass_dir / "candidates"
    approved_dir = bass_dir / "approved"
    if candidates_dir.exists():
        for old_file in candidates_dir.glob("*.mid"):
            old_file.unlink()
    candidates_dir.mkdir(parents=True, exist_ok=True)
    approved_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting {len(all_midi_outputs)} MIDI files to {candidates_dir}/")
    for filename, midi_bytes in all_midi_outputs:
        path = candidates_dir / filename
        path.write_bytes(midi_bytes)

    # --- 9. Write review YAML ---
    review = generate_bass_review_yaml(
        production_dir,
        sections,
        ranked_by_section,
        seed,
        {"theory": theory_weight, "chromatic": chromatic_weight},
        song_info,
    )
    review_path = bass_dir / "review.yml"
    with open(review_path, "w") as f:
        yaml.dump(
            review, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    # --- 10. Summary ---
    total = sum(len(v) for v in ranked_by_section.values())
    print(f"\n{'=' * 60}")
    print("BASS GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Sections:   {len(sections)}")
    print(f"Candidates: {total}")
    print(f"Review:     {review_path}")
    print(f"\nNext: Edit {review_path} to label and approve candidates")
    print(f"Then: python -m app.generators.midi.promote_chords --review {review_path}")

    return ranked_by_section


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Bass line generation pipeline — generate, score, and review bass lines"
    )
    parser.add_argument(
        "--production-dir",
        required=True,
        help="Song production directory (must contain chords/approved/)",
    )
    parser.add_argument(
        "--thread",
        default=None,
        help="Shrinkwrapped thread directory (optional, auto-detected from chord review)",
    )
    parser.add_argument(
        "--song",
        default=None,
        help="Song proposal YAML filename (optional, auto-detected from chord review)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top candidates per section (default: 5)",
    )
    parser.add_argument(
        "--theory-weight",
        type=float,
        default=0.3,
        help="Theory score weight (default: 0.3)",
    )
    parser.add_argument(
        "--chromatic-weight",
        type=float,
        default=0.7,
        help="Chromatic score weight (default: 0.7)",
    )
    parser.add_argument(
        "--onnx-path",
        default=None,
        help="Path to fusion_model.onnx (default: training/data/fusion_model.onnx)",
    )

    args = parser.parse_args()

    run_bass_pipeline(
        production_dir=args.production_dir,
        thread_dir=args.thread,
        song_filename=args.song,
        seed=args.seed,
        top_k=args.top_k,
        theory_weight=args.theory_weight,
        chromatic_weight=args.chromatic_weight,
        onnx_path=args.onnx_path,
    )


if __name__ == "__main__":
    main()
