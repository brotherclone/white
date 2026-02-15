#!/usr/bin/env python3
"""
Harmonic rhythm generation pipeline for the Music Production Pipeline.

Reads approved chords and approved drum patterns, generates variable chord
duration distributions on a half-bar grid, scores with drum accent alignment
and ChromaticScorer temporal mode, writes candidates + review YAML.

Pipeline position: chords → drums → HARMONIC RHYTHM → strums → bass → melody

Usage:
    python -m app.generators.midi.harmonic_rhythm_pipeline \
        --production-dir shrinkwrapped/.../production/black__sequential_dissolution_v2 \
        --seed 42 --top-k 20
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from app.generators.midi.chord_pipeline import (
    _to_python,
    compute_chromatic_match,
    get_chromatic_target,
    load_song_proposal,
)
from app.generators.midi.harmonic_rhythm import (
    accents_to_halfbar_mask,
    default_accent_mask,
    distribution_to_midi_bytes,
    drum_alignment_score,
    enumerate_distributions,
    extract_drum_accents,
)
from app.generators.midi.strum_pipeline import parse_chord_voicings


# ---------------------------------------------------------------------------
# Read approved sections + chord voicings
# ---------------------------------------------------------------------------


def read_approved_chords(production_dir: Path) -> list[dict]:
    """Read chord review.yml and return approved chords with voicings.

    Returns list of dicts: [{label, chord_id, voicings: [[int,...], ...], ...}]
    """
    review_path = production_dir / "chords" / "review.yml"
    if not review_path.exists():
        raise FileNotFoundError(f"Chord review not found: {review_path}")

    with open(review_path) as f:
        review = yaml.safe_load(f)

    approved_dir = production_dir / "chords" / "approved"
    sections = []

    for candidate in review.get("candidates", []):
        status = str(candidate.get("status", "")).lower()
        if status not in ("approved", "accepted"):
            continue
        label = candidate.get("label")
        if not label:
            continue

        label_key = label.lower().replace("-", "_").replace(" ", "_")
        chord_id = candidate.get("id", "unknown")

        # Find approved MIDI file for this section
        # The promote tool names files by label, so look for label-based names
        midi_files = sorted(approved_dir.glob("*.mid"))
        voicings = []

        # Try exact label match first
        for midi_file in midi_files:
            if midi_file.stem.lower().startswith(label_key):
                parsed = parse_chord_voicings(midi_file)
                voicings = [v["notes"] for v in parsed]
                break

        # If no match by label, fall back to reading chords from review YAML
        if not voicings and "chords" in candidate:

            for chord_data in candidate["chords"]:
                notes = chord_data.get("notes", [])
                if notes:
                    # Convert note names to MIDI numbers
                    midi_notes = _note_names_to_midi(notes)
                    voicings.append(midi_notes)

        if not voicings:
            continue

        sections.append(
            {
                "label": label_key,
                "label_display": label,
                "chord_id": chord_id,
                "voicings": voicings,
                "n_chords": len(voicings),
            }
        )

    return sections


def _note_names_to_midi(note_names: list[str]) -> list[int]:
    """Convert note names like ['D#2', 'A#3'] to MIDI note numbers."""
    NOTE_MAP = {
        "C": 0,
        "D": 2,
        "E": 4,
        "F": 5,
        "G": 7,
        "A": 9,
        "B": 11,
    }

    midi_notes = []
    for name in note_names:
        name = name.strip()
        if not name:
            continue
        # Parse: letter + optional sharp/flat + octave
        idx = 0
        base = name[idx].upper()
        idx += 1
        accidental = 0
        while idx < len(name) and name[idx] in "#b":
            if name[idx] == "#":
                accidental += 1
            else:
                accidental -= 1
            idx += 1
        octave = int(name[idx:])
        midi_note = (octave + 1) * 12 + NOTE_MAP.get(base, 0) + accidental
        midi_notes.append(midi_note)

    return midi_notes


# ---------------------------------------------------------------------------
# Read approved drum patterns
# ---------------------------------------------------------------------------


def read_approved_drums(
    production_dir: Path,
    time_sig: tuple[int, int] = (4, 4),
) -> dict[str, list[float]]:
    """Read approved drum MIDI files and extract accent masks per section.

    Returns dict mapping section label → accent mask (list of strong beat positions).
    """
    drums_dir = production_dir / "drums"
    approved_dir = drums_dir / "approved"
    review_path = drums_dir / "review.yml"

    if not approved_dir.exists() or not review_path.exists():
        return {}

    # Read drum review to map files to sections
    with open(review_path) as f:
        drum_review = yaml.safe_load(f)

    section_accents: dict[str, list[float]] = {}

    for candidate in drum_review.get("candidates", []):
        status = str(candidate.get("status", "")).lower()
        if status not in ("approved", "accepted"):
            continue
        section = candidate.get("section", "")
        label = candidate.get("label", "")
        if not section and not label:
            continue

        section_key = (section or label).lower().replace("-", "_").replace(" ", "_")

        # Find the approved MIDI file
        midi_file = None
        if label:
            candidate_path = approved_dir / f"{label}.mid"
            if candidate_path.exists():
                midi_file = str(candidate_path)

        if midi_file is None:
            # Try to find any approved file for this section
            for f in sorted(approved_dir.glob("*.mid")):
                if f.stem.lower().startswith(section_key):
                    midi_file = str(f)
                    break

        if midi_file is None:
            continue

        # Only use first approved drum for each section (most accents overlap anyway)
        if section_key not in section_accents:
            accents = extract_drum_accents(midi_file, time_sig)
            section_accents[section_key] = accents_to_halfbar_mask(accents, time_sig)

    return section_accents


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def hr_composite_score(
    alignment: float,
    chromatic_match: float,
    scorer_result: dict,
    alignment_weight: float = 0.3,
    chromatic_weight: float = 0.7,
) -> tuple[float, dict]:
    """Compute weighted composite score for a harmonic rhythm candidate.

    Returns (composite_score, full_breakdown).
    """
    composite = alignment_weight * alignment + chromatic_weight * chromatic_match
    breakdown = {
        "composite": round(composite, 4),
        "drum_alignment": round(alignment, 4),
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


def generate_hr_review_yaml(
    production_dir: str,
    sections: list[dict],
    ranked_by_section: dict[str, list[dict]],
    seed: int,
    scoring_weights: dict,
    song_info: dict,
) -> dict:
    """Generate the review YAML structure for harmonic rhythm candidates."""
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
                    "distribution": item["distribution"],
                    "total_bars": round(sum(item["distribution"]), 1),
                    "scores": _to_python(item["breakdown"]),
                    # Human annotation fields
                    "label": None,
                    "status": "pending",
                    "notes": "",
                }
            )

    return {
        "production_dir": str(production_dir),
        "pipeline": "harmonic-rhythm",
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
# Main pipeline
# ---------------------------------------------------------------------------


def run_harmonic_rhythm_pipeline(
    production_dir: str,
    seed: int = 42,
    top_k: int = 20,
    alignment_weight: float = 0.3,
    chromatic_weight: float = 0.7,
    onnx_path: Optional[str] = None,
):
    """Run the harmonic rhythm generation pipeline end-to-end.

    1. Read approved chord sections + voicings
    2. Read approved drum patterns + extract accents
    3. Enumerate duration distributions per section
    4. Score with drum alignment + ChromaticScorer
    5. Write top-k per section as MIDI files
    6. Write review.yml
    """
    np.random.seed(seed)

    prod_path = Path(production_dir)
    if not prod_path.exists():
        print(f"ERROR: Production directory not found: {prod_path}")
        sys.exit(1)

    # --- 1. Read approved chords ---
    print("=" * 60)
    print("HARMONIC RHYTHM GENERATION PIPELINE")
    print("=" * 60)

    sections = read_approved_chords(prod_path)
    if not sections:
        print("ERROR: No approved chord sections found in chords/review.yml")
        print("Run the chord pipeline and approve candidates first.")
        sys.exit(1)

    print(f"Sections: {', '.join(s['label_display'] for s in sections)}")

    # --- 2. Load song info ---
    chord_review_path = prod_path / "chords" / "review.yml"
    with open(chord_review_path) as f:
        chord_review = yaml.safe_load(f)

    song_info = {
        "bpm": chord_review.get("bpm", 120),
        "color_name": chord_review.get("color", "White"),
        "concept": "",
        "time_sig": (4, 4),
    }

    # Try to load full song proposal
    thread_from_review = chord_review.get("thread", "")
    song_from_review = chord_review.get("song_proposal", "")
    if thread_from_review and song_from_review:
        try:
            song_info = load_song_proposal(Path(thread_from_review), song_from_review)
        except Exception as e:
            print(f"  Warning: Could not load song proposal: {e}")

    # Parse time sig from chord review if present
    time_sig_str = chord_review.get("time_sig")
    if time_sig_str and "/" in str(time_sig_str):
        parts = str(time_sig_str).split("/")
        song_info["time_sig"] = (int(parts[0]), int(parts[1]))

    time_sig = tuple(song_info["time_sig"])
    bpm = song_info["bpm"]

    print(f"BPM:   {bpm}")
    print(f"Time:  {time_sig[0]}/{time_sig[1]}")
    print(f"Color: {song_info['color_name']}")

    # --- 3. Read approved drums ---
    drum_accents = read_approved_drums(prod_path, time_sig)
    if drum_accents:
        print(f"Drum accents loaded for: {', '.join(drum_accents.keys())}")
    else:
        print("  Warning: No approved drum patterns found — using default accent mask")

    # --- 4. Load ChromaticScorer ---
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

    # --- 5. Disambiguate section labels ---
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

    # --- 6. Generate and score per section ---
    ranked_by_section: dict[str, list[dict]] = {}
    all_midi_outputs: list[tuple[str, bytes]] = []

    for section in sections:
        section_key = section["_section_key"]
        label = section["label"]
        label_display = section["label_display"]
        n_chords = section["n_chords"]
        voicings = section["voicings"]

        print(f"\n--- Section: {label_display} [{section_key}] ({n_chords} chords) ---")

        # Get accent mask for this section
        accent_mask = drum_accents.get(label, None)
        if accent_mask is None:
            # Try section_key (for disambiguated labels like bridge_1)
            accent_mask = drum_accents.get(section_key, None)
        if accent_mask is None:
            # Try matching any key that starts with the label
            for dk, dv in drum_accents.items():
                if dk.startswith(label):
                    accent_mask = dv
                    break
        if accent_mask is None:
            accent_mask = default_accent_mask(time_sig)
            print(f"  No drum accents for '{label}' — using default mask")
        else:
            print(f"  Drum accent mask: {accent_mask}")

        # Enumerate distributions
        distributions = enumerate_distributions(n_chords, seed=seed)
        print(f"  Distributions: {len(distributions)} candidates")

        # Generate MIDI and score each
        candidates = []
        for dist in distributions:
            midi_bytes = distribution_to_midi_bytes(
                voicings, dist, bpm=bpm, time_sig=time_sig
            )
            alignment = drum_alignment_score(dist, accent_mask, time_sig)
            candidates.append(
                {
                    "distribution": dist,
                    "midi_bytes": midi_bytes,
                    "alignment": alignment,
                }
            )

        # Score with ChromaticScorer
        print(f"  Scoring {len(candidates)} candidates with ChromaticScorer...")
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
            comp, breakdown = hr_composite_score(
                cand["alignment"],
                chromatic_match,
                scorer_result,
                alignment_weight,
                chromatic_weight,
            )
            scored.append(
                {
                    "composite": comp,
                    "breakdown": breakdown,
                    "distribution": cand["distribution"],
                    "midi_bytes": cand["midi_bytes"],
                }
            )

        # Rank and take top-k
        scored.sort(key=lambda x: x["composite"], reverse=True)
        top = scored[:top_k]

        for rank, item in enumerate(top):
            item["rank"] = rank + 1
            item["id"] = f"hr_{section_key}_{rank + 1:03d}"
            all_midi_outputs.append((f"{item['id']}.mid", item["midi_bytes"]))

        ranked_by_section[section_key] = top

        # Print summary
        for item in top[:5]:
            dist_str = " | ".join(f"{d:.1f}" for d in item["distribution"])
            total = sum(item["distribution"])
            print(
                f"  #{item['rank']} [{item['id']}] [{dist_str}] = {total:.1f} bars  "
                f"composite={item['breakdown']['composite']:.3f} "
                f"align={item['breakdown']['drum_alignment']:.2f} "
                f"chromatic={item['breakdown']['chromatic']['match']:.3f}"
            )
        if len(top) > 5:
            print(f"  ... and {len(top) - 5} more")

    # --- 7. Write MIDI files ---
    hr_dir = prod_path / "harmonic_rhythm"
    candidates_dir = hr_dir / "candidates"
    approved_dir = hr_dir / "approved"
    # Clean old candidates
    if candidates_dir.exists():
        for old_file in candidates_dir.glob("*.mid"):
            old_file.unlink()
    candidates_dir.mkdir(parents=True, exist_ok=True)
    approved_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting {len(all_midi_outputs)} MIDI files to {candidates_dir}/")
    for filename, midi_bytes in all_midi_outputs:
        path = candidates_dir / filename
        path.write_bytes(midi_bytes)

    # --- 8. Write review YAML ---
    review = generate_hr_review_yaml(
        production_dir,
        sections,
        ranked_by_section,
        seed,
        {"alignment": alignment_weight, "chromatic": chromatic_weight},
        song_info,
    )
    review_path = hr_dir / "review.yml"
    with open(review_path, "w") as f:
        yaml.dump(
            review, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    # --- 9. Summary ---
    total = sum(len(v) for v in ranked_by_section.values())
    print(f"\n{'=' * 60}")
    print("HARMONIC RHYTHM GENERATION COMPLETE")
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
        description="Harmonic rhythm generation — variable chord durations scored against drums + chromatic"
    )
    parser.add_argument(
        "--production-dir",
        required=True,
        help="Song production directory (must contain chords/approved/ and drums/approved/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top candidates per section (default: 20)",
    )
    parser.add_argument(
        "--alignment-weight",
        type=float,
        default=0.3,
        help="Drum alignment weight (default: 0.3)",
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

    run_harmonic_rhythm_pipeline(
        production_dir=args.production_dir,
        seed=args.seed,
        top_k=args.top_k,
        alignment_weight=args.alignment_weight,
        chromatic_weight=args.chromatic_weight,
        onnx_path=args.onnx_path,
    )


if __name__ == "__main__":
    main()
