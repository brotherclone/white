#!/usr/bin/env python3
"""
Drum pattern generation pipeline for the Music Production Pipeline.

Reads approved chords from a song's production directory, maps song sections
to genre-appropriate drum templates, scores candidates with ChromaticScorer,
and writes top candidates as MIDI files with a review YAML.

Usage:
    python -m app.generators.midi.drum_pipeline \
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

from app.generators.midi.chord_pipeline import (
    _to_python,
    compute_chromatic_match,
    get_chromatic_target,
    load_song_proposal,
)
from app.generators.midi.drum_patterns import (
    ALL_TEMPLATES,
    DEFAULT_GENRE_FAMILY,
    DEFAULT_SECTION_ENERGY,
    GM_PERCUSSION,
    VELOCITY,
    DrumPattern,
    energy_appropriateness,
    make_fallback_pattern,
    map_genres_to_families,
    select_templates,
)

# ---------------------------------------------------------------------------
# Section reader — parse chord review.yml
# ---------------------------------------------------------------------------


def read_approved_sections(production_dir: Path) -> list[dict]:
    """Read chord review.yml and extract approved sections with bar counts.

    Returns list of dicts: [{label, bar_count, chord_id}, ...]
    """
    review_path = production_dir / "chords" / "review.yml"
    if not review_path.exists():
        raise FileNotFoundError(f"Chord review not found: {review_path}")

    with open(review_path) as f:
        review = yaml.safe_load(f)

    sections = []
    for candidate in review.get("candidates", []):
        status = str(candidate.get("status", "")).lower()
        if status not in ("approved", "accepted"):
            continue
        label = candidate.get("label")
        if not label:
            continue
        # Bar count = number of chords in the progression
        chords = candidate.get("chords", [])
        bar_count = len(chords) if chords else 4
        sections.append(
            {
                "label": label.lower().replace("-", "_").replace(" ", "_"),
                "label_display": label,
                "bar_count": bar_count,
                "chord_id": candidate.get("id", "unknown"),
            }
        )

    return sections


# ---------------------------------------------------------------------------
# Drum MIDI generation
# ---------------------------------------------------------------------------

DRUM_CHANNEL = 9  # MIDI channel 10 (0-indexed = 9)


def drum_pattern_to_midi_bytes(
    pattern: DrumPattern,
    bpm: int = 120,
    bar_count: int = 4,
    ticks_per_beat: int = 480,
) -> bytes:
    """Convert a drum pattern to MIDI bytes, repeating for bar_count bars.

    All events on MIDI channel 10 (index 9) using GM percussion mapping.
    """
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set tempo
    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    bar_length_beats = pattern.bar_length_beats()
    bar_ticks = int(bar_length_beats * ticks_per_beat)

    # Collect all events across all bars, then sort by absolute tick
    events = []  # (abs_tick, note, velocity, is_on)

    for bar_idx in range(bar_count):
        bar_offset = bar_idx * bar_ticks
        for voice_name, hits in pattern.voices.items():
            note = GM_PERCUSSION.get(voice_name)
            if note is None:
                continue
            for beat_pos, vel_level in hits:
                velocity = VELOCITY.get(vel_level, 90)
                abs_tick = bar_offset + int(beat_pos * ticks_per_beat)
                # Short note duration for percussion (1/16 note)
                note_dur = ticks_per_beat // 4
                events.append((abs_tick, note, velocity, True))
                events.append((abs_tick + note_dur, note, 0, False))

    # Sort by absolute tick, note-offs before note-ons at same tick
    events.sort(key=lambda e: (e[0], not e[3], e[1]))

    # Convert to delta times
    prev_tick = 0
    for abs_tick, note, velocity, is_on in events:
        delta = abs_tick - prev_tick
        msg_type = "note_on" if is_on else "note_off"
        track.append(
            mido.Message(
                msg_type, note=note, velocity=velocity, time=delta, channel=DRUM_CHANNEL
            )
        )
        prev_tick = abs_tick

    track.append(mido.MetaMessage("end_of_track", time=0))

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def drum_composite_score(
    energy_score: float,
    chromatic_match: float,
    scorer_result: dict,
    energy_weight: float = 0.3,
    chromatic_weight: float = 0.7,
) -> tuple[float, dict]:
    """Compute weighted composite score for a drum candidate.

    Returns (composite_score, full_breakdown).
    """
    composite = energy_weight * energy_score + chromatic_weight * chromatic_match
    breakdown = {
        "composite": round(composite, 4),
        "energy_appropriateness": round(energy_score, 4),
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


def generate_drum_review_yaml(
    production_dir: str,
    sections: list[dict],
    ranked_by_section: dict[str, list[dict]],
    seed: int,
    scoring_weights: dict,
    song_info: dict,
) -> dict:
    """Generate the review YAML structure for drum candidates."""
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
                    "genre_family": item["genre_family"],
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
        "pipeline": "drum-generation",
        "bpm": song_info.get("bpm", 120),
        "time_sig": f"{song_info['time_sig'][0]}/{song_info['time_sig'][1]}",
        "color": song_info.get("color_name", ""),
        "genre_families": song_info.get("_genre_families", []),
        "generated": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "scoring_weights": scoring_weights,
        "sections_found": [s["label_display"] for s in sections],
        "candidates": all_candidates,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_drum_pipeline(
    production_dir: str,
    thread_dir: Optional[str] = None,
    song_filename: Optional[str] = None,
    seed: int = 42,
    top_k: int = 5,
    energy_weight: float = 0.3,
    chromatic_weight: float = 0.7,
    energy_overrides: Optional[dict[str, str]] = None,
    genre_overrides: Optional[list[str]] = None,
    onnx_path: Optional[str] = None,
):
    """Run the drum generation pipeline end-to-end.

    1. Read approved chord sections
    2. Determine genre families and energy levels
    3. Select and generate drum pattern candidates
    4. Score with energy appropriateness + ChromaticScorer
    5. Write top-k per section as MIDI files
    6. Write review.yml
    """
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)

    prod_path = Path(production_dir)
    if not prod_path.exists():
        print(f"ERROR: Production directory not found: {prod_path}")
        sys.exit(1)

    # --- 1. Read approved chord sections ---
    print("=" * 60)
    print("DRUM PATTERN GENERATION PIPELINE")
    print("=" * 60)

    sections = read_approved_sections(prod_path)
    if not sections:
        print("ERROR: No approved chord sections found in chords/review.yml")
        print("Run the chord pipeline and approve candidates first.")
        sys.exit(1)

    print(f"Sections: {', '.join(s['label_display'] for s in sections)}")

    # --- 2. Load song info for genre/concept ---
    # Try to get song info from chord review.yml metadata
    chord_review_path = prod_path / "chords" / "review.yml"
    with open(chord_review_path) as f:
        chord_review = yaml.safe_load(f)

    # If thread_dir and song_filename provided, load full proposal
    # Otherwise reconstruct from chord review metadata
    song_info = {}
    if thread_dir and song_filename:
        song_info = load_song_proposal(Path(thread_dir), song_filename)
    else:
        # Reconstruct minimal info from chord review
        song_info = {
            "bpm": chord_review.get("bpm", 120),
            "color_name": chord_review.get("color", "White"),
            "concept": "",
            "key_root": "C",
            "mode": "Minor",
            "time_sig": (4, 4),
            "song_filename": chord_review.get("song_proposal", ""),
            "thread_dir": chord_review.get("thread", ""),
            "raw_proposal": {},
        }
        # Try to load the actual proposal for genres/moods
        thread_from_review = chord_review.get("thread", "")
        song_from_review = chord_review.get("song_proposal", "")
        if thread_from_review and song_from_review:
            try:
                song_info = load_song_proposal(
                    Path(thread_from_review), song_from_review
                )
            except (FileNotFoundError, Exception) as e:
                print(f"  Warning: Could not load song proposal: {e}")

        # Parse time sig from chord review if present
        time_sig_str = chord_review.get("time_sig")
        if time_sig_str and "/" in str(time_sig_str):
            parts = str(time_sig_str).split("/")
            song_info["time_sig"] = (int(parts[0]), int(parts[1]))

    # Get genre tags from the raw proposal
    raw_proposal = song_info.get("raw_proposal", {})
    genre_tags = raw_proposal.get("genres", [])
    if not genre_tags:
        genre_tags = []

    print(f"BPM:     {song_info['bpm']}")
    print(f"Time:    {song_info['time_sig'][0]}/{song_info['time_sig'][1]}")
    print(f"Color:   {song_info['color_name']}")
    print(f"Genres:  {genre_tags or '(none)'}")

    # --- 3. Determine genre families ---
    if genre_overrides:
        genre_families = genre_overrides
    else:
        genre_families = map_genres_to_families(genre_tags)
    if not genre_families:
        genre_families = [DEFAULT_GENRE_FAMILY]
        print(f"  No genre match — falling back to: {DEFAULT_GENRE_FAMILY}")
    print(f"Families: {', '.join(genre_families)}")

    song_info["_genre_families"] = genre_families

    # --- 4. Apply energy overrides ---
    section_energy = dict(DEFAULT_SECTION_ENERGY)
    if energy_overrides:
        section_energy.update(energy_overrides)

    # --- 5. Generate and score per section ---
    target = get_chromatic_target(song_info["color_name"])
    time_sig = tuple(song_info["time_sig"])

    # Load ChromaticScorer
    print("\nLoading ChromaticScorer...")
    from training.chromatic_scorer import ChromaticScorer

    scorer = ChromaticScorer(onnx_path=onnx_path) if onnx_path else ChromaticScorer()

    concept_text = song_info.get("concept", "")
    if not concept_text:
        concept_text = f"{song_info['color_name']} chromatic concept"
        print(f"  Warning: No concept text, using fallback: '{concept_text}'")
    concept_emb = scorer.prepare_concept(concept_text)
    print(f"  Concept encoded ({concept_emb.shape[0]}-dim)")

    ranked_by_section: dict[str, list[dict]] = {}
    all_midi_outputs: list[tuple[str, bytes]] = []  # (filename, midi_bytes)

    # Disambiguate duplicate section labels by appending occurrence count
    label_occurrence: dict[str, int] = {}
    for section in sections:
        label = section["label"]
        label_occurrence[label] = label_occurrence.get(label, 0) + 1
        count = label_occurrence[label]
        # Only suffix if there are duplicates (we need a second pass to know)
        section["_occurrence"] = count

    # Mark which labels have duplicates
    duplicate_labels = {k for k, v in label_occurrence.items() if v > 1}
    for section in sections:
        label = section["label"]
        if label in duplicate_labels:
            section["_section_key"] = f"{label}_{section['_occurrence']}"
        else:
            section["_section_key"] = label

    for section in sections:
        section_key = section["_section_key"]
        label = section["label"]
        label_display = section["label_display"]
        bar_count = section["bar_count"]
        target_energy = section_energy.get(label, "medium")

        print(
            f"\n--- Section: {label_display} [{section_key}] ({bar_count} bars, energy={target_energy}) ---"
        )

        # Select templates
        templates = select_templates(
            ALL_TEMPLATES, time_sig, genre_families, target_energy
        )

        # Add fallback if no templates found
        if not templates:
            print(f"  No templates for {time_sig} + {genre_families} — using fallback")
            templates = [make_fallback_pattern(time_sig)]

        print(f"  Templates: {len(templates)} candidates")

        # Generate MIDI for each template
        candidates = []
        for tmpl in templates:
            midi_bytes = drum_pattern_to_midi_bytes(
                tmpl, bpm=song_info["bpm"], bar_count=bar_count
            )
            e_score = energy_appropriateness(tmpl.energy, target_energy)
            candidates.append(
                {
                    "template": tmpl,
                    "midi_bytes": midi_bytes,
                    "energy_score": e_score,
                    "pattern_name": tmpl.name,
                    "genre_family": tmpl.genre_family,
                    "energy": tmpl.energy,
                }
            )

        # Score with ChromaticScorer
        print(f"  Scoring {len(candidates)} candidates...")
        scorer_candidates = [{"midi_bytes": c["midi_bytes"]} for c in candidates]
        scorer_results = scorer.score_batch(scorer_candidates, concept_emb=concept_emb)

        # Match back scorer results
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
            comp, breakdown = drum_composite_score(
                cand["energy_score"],
                chromatic_match,
                scorer_result,
                energy_weight,
                chromatic_weight,
            )
            scored.append(
                {
                    "composite": comp,
                    "breakdown": breakdown,
                    "midi_bytes": cand["midi_bytes"],
                    "pattern_name": cand["pattern_name"],
                    "genre_family": cand["genre_family"],
                    "energy": cand["energy"],
                    "description": cand["template"].description,
                }
            )

        # Rank and take top-k
        scored.sort(key=lambda x: x["composite"], reverse=True)
        top = scored[:top_k]

        for rank, item in enumerate(top):
            item["rank"] = rank + 1
            item["id"] = f"drum_{section_key}_{rank + 1:02d}"
            all_midi_outputs.append((f"{item['id']}.mid", item["midi_bytes"]))

        ranked_by_section[section_key] = top

        # Print summary
        for item in top:
            print(
                f"  #{item['rank']} [{item['id']}] {item['pattern_name']:25s} "
                f"composite={item['breakdown']['composite']:.3f} "
                f"energy={item['breakdown']['energy_appropriateness']:.1f} "
                f"chromatic={item['breakdown']['chromatic']['match']:.3f}"
            )

    # --- 6. Write MIDI files ---
    drums_dir = prod_path / "drums"
    candidates_dir = drums_dir / "candidates"
    approved_dir = drums_dir / "approved"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    approved_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting {len(all_midi_outputs)} MIDI files to {candidates_dir}/")
    for filename, midi_bytes in all_midi_outputs:
        path = candidates_dir / filename
        path.write_bytes(midi_bytes)

    # --- 7. Write review YAML ---
    review = generate_drum_review_yaml(
        production_dir,
        sections,
        ranked_by_section,
        seed,
        {"energy": energy_weight, "chromatic": chromatic_weight},
        song_info,
    )
    review_path = drums_dir / "review.yml"
    with open(review_path, "w") as f:
        yaml.dump(
            review, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    # --- 8. Summary ---
    total = sum(len(v) for v in ranked_by_section.values())
    print(f"\n{'=' * 60}")
    print("DRUM GENERATION COMPLETE")
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


def parse_energy_overrides(value: str) -> dict[str, str]:
    """Parse energy override string like 'verse=high,chorus=medium'."""
    overrides = {}
    for pair in value.split(","):
        pair = pair.strip()
        if "=" in pair:
            section, energy = pair.split("=", 1)
            overrides[section.strip().lower()] = energy.strip().lower()
    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="Drum pattern generation pipeline — generate, score, and review drum patterns"
    )
    parser.add_argument(
        "--production-dir",
        required=True,
        help="Song production directory (e.g., .../production/black__sequential_dissolution_v2)",
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
        "--energy-weight",
        type=float,
        default=0.3,
        help="Energy appropriateness weight (default: 0.3)",
    )
    parser.add_argument(
        "--chromatic-weight",
        type=float,
        default=0.7,
        help="Chromatic score weight (default: 0.7)",
    )
    parser.add_argument(
        "--energy-override",
        default=None,
        help="Override section energy: 'verse=high,chorus=medium'",
    )
    parser.add_argument(
        "--genre-override",
        default=None,
        help="Force genre families: 'krautrock,ambient'",
    )
    parser.add_argument(
        "--onnx-path",
        default=None,
        help="Path to fusion_model.onnx (default: training/data/fusion_model.onnx)",
    )

    args = parser.parse_args()

    energy_overrides = None
    if args.energy_override:
        energy_overrides = parse_energy_overrides(args.energy_override)

    genre_overrides = None
    if args.genre_override:
        genre_overrides = [g.strip() for g in args.genre_override.split(",")]

    run_drum_pipeline(
        production_dir=args.production_dir,
        thread_dir=args.thread,
        song_filename=args.song,
        seed=args.seed,
        top_k=args.top_k,
        energy_weight=args.energy_weight,
        chromatic_weight=args.chromatic_weight,
        energy_overrides=energy_overrides,
        genre_overrides=genre_overrides,
        onnx_path=args.onnx_path,
    )


if __name__ == "__main__":
    main()
