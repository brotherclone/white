#!/usr/bin/env python3
"""
Melody generation pipeline for the Music Production Pipeline.

Reads approved chords, harmonic rhythm, and song proposal metadata. Generates
melody candidates from contour templates within singer vocal range constraints,
scores with theory + ChromaticScorer composite, writes top candidates as MIDI
files with a review YAML.

Pipeline position: chords → drums → harmonic rhythm → strums → bass → MELODY

Usage:
    python -m app.generators.midi.melody_pipeline \
        --production-dir shrinkwrapped/.../production/black__sequential_dissolution_v2 \
        --singer gabriel --seed 42 --top-k 5
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

from app.generators.midi.melody_patterns import (
    ALL_TEMPLATES,
    MELODY_CHANNEL,
    VELOCITY,
    MelodyPattern,
    SingerRange,
    SINGERS,
    chord_tone_alignment,
    contour_quality,
    infer_singer,
    make_fallback_pattern,
    melody_theory_score,
    resolve_melody_notes,
    select_templates,
    singability_score,
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
# Read approved sections and chord data (reuse from bass pipeline pattern)
# ---------------------------------------------------------------------------


def read_approved_sections(chord_review: dict) -> list[dict]:
    """Extract approved sections from chord review YAML."""
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


def extract_section_chord_data(
    production_dir: Path,
) -> tuple[dict[str, list[list[int]]], dict]:
    """Read approved chord voicings and review metadata."""
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
# Melody MIDI generation
# ---------------------------------------------------------------------------


def melody_notes_to_midi_bytes(
    resolved_notes: list[tuple[float, int, float]],
    bpm: int = 120,
    ticks_per_beat: int = 480,
) -> bytes:
    """Convert resolved melody notes to MIDI bytes.

    Args:
        resolved_notes: List of (onset_beat, midi_note, duration_beats).
        bpm: Beats per minute.
        ticks_per_beat: MIDI resolution.

    Returns:
        MIDI file as bytes.
    """
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    # Build events
    events = []
    for onset, note, dur in resolved_notes:
        on_tick = int(onset * ticks_per_beat)
        off_tick = int((onset + dur) * ticks_per_beat)
        velocity = VELOCITY["normal"]
        events.append((on_tick, note, velocity, True))
        events.append((off_tick, note, 0, False))

    events.sort(key=lambda e: (e[0], not e[3], e[1]))

    prev_tick = 0
    for abs_tick, note, velocity, is_on in events:
        delta = abs_tick - prev_tick
        msg_type = "note_on" if is_on else "note_off"
        track.append(
            mido.Message(
                msg_type,
                note=note,
                velocity=velocity,
                time=delta,
                channel=MELODY_CHANNEL,
            )
        )
        prev_tick = abs_tick

    track.append(mido.MetaMessage("end_of_track", time=0))

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def generate_melody_for_section(
    pattern: MelodyPattern,
    voicings: list[list[int]],
    singer: SingerRange,
    bpm: int = 120,
    durations: list[float] | None = None,
) -> tuple[bytes, list[tuple[float, int, float]]]:
    """Generate melody MIDI for an entire section from a pattern and chord voicings.

    Applies the pattern to each chord in the section, accumulating the melody
    over the full duration.

    Returns:
        Tuple of (midi_bytes, all_resolved_notes).
    """
    bar_beats = pattern.bar_length_beats()
    all_notes: list[tuple[float, int, float]] = []
    offset_beats = 0.0

    for chord_idx, voicing in enumerate(voicings):
        if durations is not None and chord_idx < len(durations):
            chord_dur_beats = durations[chord_idx] * bar_beats
        else:
            chord_dur_beats = bar_beats

        next_voicing = (
            voicings[chord_idx + 1] if chord_idx + 1 < len(voicings) else None
        )

        # Resolve the pattern for this chord
        notes = resolve_melody_notes(pattern, voicing, singer, next_voicing)

        # Repeat pattern if chord duration > 1 bar
        pattern_dur = bar_beats
        repeat_offset = 0.0
        while repeat_offset < chord_dur_beats:
            for onset, note, dur in notes:
                abs_onset = offset_beats + repeat_offset + onset
                # Don't exceed chord boundary
                if repeat_offset + onset >= chord_dur_beats:
                    break
                # Clamp duration to chord boundary
                max_dur = (offset_beats + chord_dur_beats) - abs_onset
                clamped_dur = min(dur, max_dur)
                if clamped_dur > 0:
                    all_notes.append((abs_onset, note, clamped_dur))
            repeat_offset += pattern_dur

        offset_beats += chord_dur_beats

    midi_bytes = melody_notes_to_midi_bytes(all_notes, bpm=bpm)
    return midi_bytes, all_notes


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def melody_composite_score(
    theory: float,
    chromatic_match: float,
    scorer_result: dict,
    theory_breakdown: dict,
    theory_weight: float = 0.3,
    chromatic_weight: float = 0.7,
) -> tuple[float, dict]:
    """Compute weighted composite score for a melody candidate."""
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


def generate_melody_review_yaml(
    production_dir: str,
    sections: list[dict],
    ranked_by_section: dict[str, list[dict]],
    seed: int,
    scoring_weights: dict,
    song_info: dict,
    singer_name: str,
) -> dict:
    """Generate the review YAML structure for melody candidates."""
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
                    "contour": item["contour"],
                    "pattern_name": item["pattern_name"],
                    "energy": item["energy"],
                    "singer": singer_name,
                    "scores": _to_python(item["breakdown"]),
                    "label": None,
                    "status": "pending",
                    "notes": "",
                }
            )

    return {
        "production_dir": str(production_dir),
        "pipeline": "melody-generation",
        "bpm": song_info.get("bpm", 120),
        "time_sig": f"{song_info['time_sig'][0]}/{song_info['time_sig'][1]}",
        "color": song_info.get("color_name", ""),
        "singer": singer_name,
        "generated": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "scoring_weights": scoring_weights,
        "sections_found": [s["label_display"] for s in sections],
        "candidates": all_candidates,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

DEFAULT_ENERGY = {
    "intro": "low",
    "verse": "medium",
    "chorus": "high",
    "bridge": "low",
    "outro": "medium",
}


def run_melody_pipeline(
    production_dir: str,
    thread_dir: Optional[str] = None,
    song_filename: Optional[str] = None,
    singer_name: Optional[str] = None,
    seed: int = 42,
    top_k: int = 5,
    theory_weight: float = 0.3,
    chromatic_weight: float = 0.7,
    onnx_path: Optional[str] = None,
):
    """Run the melody generation pipeline end-to-end."""
    np.random.seed(seed)

    prod_path = Path(production_dir)
    if not prod_path.exists():
        print(f"ERROR: Production directory not found: {prod_path}")
        sys.exit(1)

    # --- 1. Read approved chords ---
    print("=" * 60)
    print("MELODY GENERATION PIPELINE")
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

    # --- 3. Determine singer ---
    if singer_name:
        singer_key = singer_name.lower().strip()
        if singer_key not in SINGERS:
            print(
                f"ERROR: Unknown singer '{singer_name}'. Available: {', '.join(SINGERS.keys())}"
            )
            sys.exit(1)
        singer = SINGERS[singer_key]
    else:
        # Try to get singer from song proposal
        proposal_singer = song_info.get("singer", "")
        if proposal_singer and proposal_singer.lower() in SINGERS:
            singer = SINGERS[proposal_singer.lower()]
        else:
            # Infer from key
            key_str = song_info.get("key", "")
            if key_str:
                from app.generators.midi.chord_pipeline import parse_key_string

                key_info = parse_key_string(key_str)
                tonic_midi = key_info.get("tonic_midi", 60)
            else:
                tonic_midi = 60
            singer = infer_singer(tonic_midi)

    print(
        f"Singer: {singer.name} ({singer.voice_type}, MIDI {singer.low}-{singer.high})"
    )

    # --- 4. Read harmonic rhythm ---
    hr_durations = read_approved_harmonic_rhythm(prod_path)
    if hr_durations:
        print(f"Harmonic rhythm loaded for: {', '.join(hr_durations.keys())}")
    else:
        print("  No approved harmonic rhythm — using 1 bar per chord")

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
            voicings = chord_data.get(section_key, [])
        if not voicings:
            for k, v in chord_data.items():
                if k.lower().startswith(label):
                    voicings = v
                    break

        if not voicings:
            print(f"\n--- Section: {label_display} — SKIPPED (no chord voicings) ---")
            continue

        section_durations = hr_durations.get(label, None)
        if section_durations and len(section_durations) != len(voicings):
            print(
                f"  Warning: HR durations ({len(section_durations)}) != voicings ({len(voicings)}) for {label}, ignoring HR"
            )
            section_durations = None

        print(
            f"\n--- Section: {label_display} [{section_key}] ({len(voicings)} chords) ---"
        )

        target_energy = DEFAULT_ENERGY.get(label, "medium")

        templates = select_templates(ALL_TEMPLATES, time_sig, target_energy)
        if not templates:
            print(f"  No templates for {time_sig} — using fallback")
            templates = [make_fallback_pattern(time_sig)]

        print(
            f"  Templates: {len(templates)} candidates (energy target: {target_energy})"
        )

        # Extract chord tones for scoring
        chord_tones_pc = set()
        for voicing in voicings:
            for n in voicing:
                chord_tones_pc.add(n % 12)

        candidates = []
        for tmpl in templates:
            midi_bytes, resolved_notes = generate_melody_for_section(
                tmpl,
                voicings,
                singer,
                bpm=bpm,
                durations=section_durations,
            )

            # Theory scoring
            sing = singability_score(resolved_notes, singer)
            ct = chord_tone_alignment(resolved_notes, chord_tones_pc, time_sig)
            cq = contour_quality(resolved_notes)
            theory = melody_theory_score(sing, ct, cq)

            theory_breakdown = {
                "singability": sing,
                "chord_tone_alignment": ct,
                "contour_quality": cq,
            }

            candidates.append(
                {
                    "template": tmpl,
                    "midi_bytes": midi_bytes,
                    "resolved_notes": resolved_notes,
                    "theory": theory,
                    "theory_breakdown": theory_breakdown,
                    "pattern_name": tmpl.name,
                    "contour": tmpl.contour,
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
            comp, breakdown = melody_composite_score(
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
                    "contour": cand["contour"],
                    "energy": cand["energy"],
                    "description": cand["template"].description,
                }
            )

        scored.sort(key=lambda x: x["composite"], reverse=True)
        top = scored[:top_k]

        for rank, item in enumerate(top):
            item["rank"] = rank + 1
            item["id"] = f"melody_{section_key}_{rank + 1:02d}"
            all_midi_outputs.append((f"{item['id']}.mid", item["midi_bytes"]))

        ranked_by_section[section_key] = top

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
    melody_dir = prod_path / "melody"
    candidates_dir = melody_dir / "candidates"
    approved_dir = melody_dir / "approved"
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
    review = generate_melody_review_yaml(
        production_dir,
        sections,
        ranked_by_section,
        seed,
        {"theory": theory_weight, "chromatic": chromatic_weight},
        song_info,
        singer.name,
    )
    review_path = melody_dir / "review.yml"
    with open(review_path, "w") as f:
        yaml.dump(
            review,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    # --- 10. Summary ---
    total = sum(len(v) for v in ranked_by_section.values())
    print(f"\n{'=' * 60}")
    print("MELODY GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Singer:     {singer.name} ({singer.voice_type})")
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
        description="Melody generation pipeline — generate, score, and review melodies",
    )
    parser.add_argument(
        "--production-dir",
        required=True,
        help="Song production directory (must contain chords/approved/)",
    )
    parser.add_argument(
        "--thread",
        default=None,
        help="Shrinkwrapped thread directory (optional)",
    )
    parser.add_argument(
        "--song",
        default=None,
        help="Song proposal YAML filename (optional)",
    )
    parser.add_argument(
        "--singer",
        default=None,
        help=f"Singer name: {', '.join(SINGERS.keys())} (optional, inferred from key)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
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

    run_melody_pipeline(
        production_dir=args.production_dir,
        thread_dir=args.thread,
        song_filename=args.song,
        singer_name=args.singer,
        seed=args.seed,
        top_k=args.top_k,
        theory_weight=args.theory_weight,
        chromatic_weight=args.chromatic_weight,
        onnx_path=args.onnx_path,
    )


if __name__ == "__main__":
    main()
