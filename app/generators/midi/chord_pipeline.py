#!/usr/bin/env python3
"""
Chord generation pipeline for the Music Production Pipeline.

Reads a song proposal from a shrinkwrapped thread, generates chord progression
candidates via Markov chains, scores them with both music theory metrics and
ChromaticScorer, and writes the top candidates as MIDI files with a review YAML.

Usage:
    python -m app.generators.midi.chord_pipeline \
        --thread shrinkwrapped/white-the-breathing-machine-learns-to-sing \
        --song "song_proposal_Black (0x221f20)_sequential_dissolution_v2.yml" \
        --seed 42 --num-candidates 200 --top-k 10
"""

import argparse
import io
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mido
import numpy as np
import yaml

from app.generators.midi.prototype.generator import ChordProgressionGenerator
from app.generators.midi.harmonic_rhythm import enumerate_distributions
from app.generators.midi.strum_patterns import (
    StrumPattern,
    get_patterns_for_time_sig,
    strum_to_midi_bytes,
)


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
# Chromatic target mapping (color → mode distributions)
# ---------------------------------------------------------------------------

# Target distributions: [Past, Present, Future] / [Thing, Place, Person] / [Imagined, Forgotten, Known]
# Primary mode gets 0.8, others split 0.1 each.
CHROMATIC_TARGETS = {
    "Red": {
        "temporal": [0.8, 0.1, 0.1],
        "spatial": [0.8, 0.1, 0.1],
        "ontological": [0.1, 0.1, 0.8],
    },
    "Orange": {
        "temporal": [0.8, 0.1, 0.1],
        "spatial": [0.1, 0.8, 0.1],
        "ontological": [0.8, 0.1, 0.1],
    },
    "Yellow": {
        "temporal": [0.1, 0.8, 0.1],
        "spatial": [0.1, 0.8, 0.1],
        "ontological": [0.1, 0.1, 0.8],
    },
    "Green": {
        "temporal": [0.1, 0.8, 0.1],
        "spatial": [0.1, 0.8, 0.1],
        "ontological": [0.1, 0.8, 0.1],
    },
    "Blue": {
        "temporal": [0.8, 0.1, 0.1],
        "spatial": [0.1, 0.1, 0.8],
        "ontological": [0.1, 0.1, 0.8],
    },
    "Indigo": {
        "temporal": [0.1, 0.1, 0.8],
        "spatial": [0.8, 0.1, 0.1],
        "ontological": [0.1, 0.8, 0.1],
    },
    "Violet": {
        "temporal": [0.1, 0.8, 0.1],
        "spatial": [0.1, 0.1, 0.8],
        "ontological": [0.8, 0.1, 0.1],
    },
    # Black and White use uniform — scorer confidence still provides signal
    "Black": {
        "temporal": [1 / 3, 1 / 3, 1 / 3],
        "spatial": [1 / 3, 1 / 3, 1 / 3],
        "ontological": [1 / 3, 1 / 3, 1 / 3],
    },
    "White": {
        "temporal": [1 / 3, 1 / 3, 1 / 3],
        "spatial": [1 / 3, 1 / 3, 1 / 3],
        "ontological": [1 / 3, 1 / 3, 1 / 3],
    },
}

TEMPORAL_MODES = ["past", "present", "future"]
SPATIAL_MODES = ["thing", "place", "person"]
ONTOLOGICAL_MODES = ["imagined", "forgotten", "known"]


def get_chromatic_target(color_name: str) -> dict:
    """Map rainbow color name to target mode distributions.

    Returns dict with temporal, spatial, ontological arrays (each length 3, sums to 1).
    """
    normalized = color_name.strip().capitalize()
    if normalized not in CHROMATIC_TARGETS:
        print(f"  Warning: Unknown color '{color_name}', using uniform targets")
        return CHROMATIC_TARGETS["White"]
    return CHROMATIC_TARGETS[normalized]


# ---------------------------------------------------------------------------
# Song proposal parsing
# ---------------------------------------------------------------------------


def parse_key_string(key_str: str) -> tuple[str, str]:
    """Parse key string like 'F# minor' into (root, mode).

    Returns (key_root, mode) where mode is 'Major' or 'Minor' to match
    the chord prototype's convention.
    """
    key_str = key_str.strip()
    # Handle unicode symbols
    key_str = key_str.replace("♭", "b").replace("♯", "#")

    parts = key_str.split()
    if len(parts) < 2:
        return parts[0] if parts else "C", "Major"

    root = parts[0]
    mode_str = " ".join(parts[1:]).lower()

    if "minor" in mode_str or "min" in mode_str:
        mode = "Minor"
    else:
        mode = "Major"

    return root, mode


def load_song_proposal(thread_dir: Path, song_filename: str) -> dict:
    """Load and parse a song proposal from a shrinkwrapped thread.

    Returns dict with: key_root, mode, bpm, time_sig, concept, color_name, raw_proposal.
    """
    song_path = thread_dir / "yml" / song_filename
    if not song_path.exists():
        raise FileNotFoundError(f"Song proposal not found: {song_path}")

    with open(song_path) as f:
        proposal = yaml.safe_load(f)

    # Parse key
    key_str = proposal.get("key", "C major")
    key_root, mode = parse_key_string(key_str)

    # Parse BPM
    bpm = proposal.get("bpm", 120)

    # Parse time signature
    tempo = proposal.get("tempo", {})
    time_sig_num = tempo.get("numerator", 4) if isinstance(tempo, dict) else 4
    time_sig_den = tempo.get("denominator", 4) if isinstance(tempo, dict) else 4

    # Parse color
    rainbow_color = proposal.get("rainbow_color", {})
    if isinstance(rainbow_color, dict):
        color_name = rainbow_color.get("color_name", "White")
    else:
        color_name = str(rainbow_color)

    # Get concept from thread manifest
    manifest_path = thread_dir / "manifest.yml"
    concept = ""
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        concept = manifest.get("concept", "")

    # Also get the song-level concept if available
    song_concept = proposal.get("concept", "")

    return {
        "key_root": key_root,
        "mode": mode,
        "bpm": bpm,
        "time_sig": (time_sig_num, time_sig_den),
        "concept": song_concept or concept,
        "color_name": color_name,
        "song_filename": song_filename,
        "thread_dir": str(thread_dir),
        "raw_proposal": proposal,
    }


# ---------------------------------------------------------------------------
# MIDI export
# ---------------------------------------------------------------------------


def progression_to_midi_bytes(
    progression: list[dict], bpm: int = 120, ticks_per_beat: int = 480
) -> bytes:
    """Convert a chord progression (list of chord dicts) to MIDI file bytes.

    Each chord is held for one bar (in 4/4 — 4 beats).
    """
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set tempo
    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    # One bar per chord (4 beats)
    bar_ticks = ticks_per_beat * 4

    for chord in progression:
        midi_notes = chord.get("midi_notes", [])
        if not midi_notes:
            continue

        # Note on — all notes simultaneously
        for i, note in enumerate(midi_notes):
            track.append(mido.Message("note_on", note=note, velocity=80, time=0))

        # Note off after one bar
        for i, note in enumerate(midi_notes):
            time = bar_ticks if i == 0 else 0
            track.append(mido.Message("note_off", note=note, velocity=0, time=time))

    track.append(mido.MetaMessage("end_of_track", time=0))

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def write_midi_file(midi_bytes: bytes, path: Path):
    """Write MIDI bytes to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(midi_bytes)


# ---------------------------------------------------------------------------
# Chord primitive — HR + strum baking
# ---------------------------------------------------------------------------


def sample_hr_distribution(n_chords: int, rng) -> list[float]:
    """Pick a random HR duration distribution for n_chords."""
    distributions = enumerate_distributions(n_chords)
    return rng.choice(distributions)


def sample_strum_pattern(
    time_sig: tuple[int, int], rng, filter_names: list[str] | None = None
) -> StrumPattern:
    """Pick a random strum pattern for the given time signature."""
    patterns = get_patterns_for_time_sig(time_sig, filter_names)
    return rng.choice(patterns)


def progression_to_primitive_midi_bytes(
    progression: list[dict],
    bpm: int,
    time_sig: tuple[int, int],
    hr_dist: list[float],
    strum_pattern: StrumPattern,
) -> bytes:
    """Generate primitive MIDI with HR distribution and strum articulation baked in."""
    voicings = [chord.get("midi_notes", []) for chord in progression]
    return strum_to_midi_bytes(voicings, strum_pattern, bpm=bpm, durations=hr_dist)


def generate_scratch_beat(
    bpm: int,
    bar_count: int,
    time_sig: tuple[int, int],
    genre_families: list[str] | None = None,
) -> bytes:
    """Generate a minimal scratch drum MIDI for auditioning a chord primitive.

    Uses the lowest-energy template from the inferred genre family.
    """
    from app.generators.midi.drum_patterns import (
        ALL_TEMPLATES,
        DEFAULT_GENRE_FAMILY,
        select_templates,
    )
    from app.generators.midi.drum_pipeline import drum_pattern_to_midi_bytes

    families = genre_families or [DEFAULT_GENRE_FAMILY]
    templates = select_templates(ALL_TEMPLATES, time_sig, families, "low")
    if not templates:
        templates = select_templates(
            ALL_TEMPLATES, time_sig, [DEFAULT_GENRE_FAMILY], "low"
        )
    if not templates:
        templates = [
            t for t in ALL_TEMPLATES if t.time_sig == time_sig
        ] or ALL_TEMPLATES[:1]

    return drum_pattern_to_midi_bytes(templates[0], bpm=bpm, bar_count=bar_count)


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def compute_chromatic_match(scorer_result: dict, target: dict) -> float:
    """Compute how well ChromaticScorer output matches the target distribution.

    Uses dot product between predicted and target distributions (range 0-1).
    Higher = better match.
    """
    match = 0.0
    for dim, modes in [
        ("temporal", TEMPORAL_MODES),
        ("spatial", SPATIAL_MODES),
        ("ontological", ONTOLOGICAL_MODES),
    ]:
        target_dist = np.array(target[dim])
        pred_dist = np.array([scorer_result[dim][m] for m in modes])
        # Dot product of normalized distributions — 1.0 when identical
        match += np.dot(pred_dist, target_dist)
    # Average across 3 dimensions, weighted by confidence
    confidence = scorer_result.get("confidence", 0.5)
    return (match / 3.0) * (0.5 + 0.5 * confidence)


def composite_score(
    theory_score: float,
    theory_breakdown: dict,
    chromatic_match: float,
    scorer_result: dict,
    theory_weight: float = 0.3,
    chromatic_weight: float = 0.7,
) -> tuple[float, dict]:
    """Compute weighted composite score.

    Returns (composite_score, full_breakdown).
    """
    composite = theory_weight * theory_score + chromatic_weight * chromatic_match
    breakdown = {
        "composite": round(composite, 4),
        "theory": {k: round(v, 4) for k, v in theory_breakdown.items()},
        "theory_total": round(theory_score, 4),
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
# Progression summary (human-readable)
# ---------------------------------------------------------------------------


def progression_summary(progression: list[dict]) -> str:
    """Create a human-readable summary of a chord progression."""
    parts = []
    for chord in progression:
        func = chord.get("function", "?")
        name = chord.get("chord_name", "?")
        parts.append(f"{func}({name})")
    return " - ".join(parts)


# ---------------------------------------------------------------------------
# Review YAML generation
# ---------------------------------------------------------------------------


def generate_review_yaml(
    song_info: dict, ranked_candidates: list[dict], seed: int, scoring_weights: dict
) -> dict:
    """Generate the review YAML structure."""
    candidates = []
    for item in ranked_candidates:
        candidates.append(
            {
                "id": item["id"],
                "midi_file": f"candidates/{item['id']}.mid",
                "scratch_midi": f"candidates/{item['id']}_scratch.mid",
                "rank": item["rank"],
                "scores": _to_python(item["breakdown"]),
                "hr_distribution": _to_python(item.get("hr_distribution", [])),
                "strum_pattern": item.get("strum_pattern", "whole"),
                "progression": item["summary"],
                "chords": [
                    {
                        "function": c.get("function", "?"),
                        "name": c.get("chord_name", "?"),
                        "notes": c.get("note_names", []),
                    }
                    for c in item["progression"]
                ],
                # Human annotation fields
                "label": None,
                "status": "pending",
                "notes": "",
            }
        )

    return {
        "song_proposal": song_info["song_filename"],
        "thread": str(song_info["thread_dir"]),
        "key": f"{song_info['key_root']} {song_info['mode'].lower()}",
        "bpm": song_info["bpm"],
        "color": song_info["color_name"],
        "generated": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "scoring_weights": scoring_weights,
        "candidates": candidates,
    }


# ---------------------------------------------------------------------------
# Song slug for directory naming
# ---------------------------------------------------------------------------


def song_slug(song_filename: str) -> str:
    """Convert song proposal filename to a directory-safe slug."""
    # Remove file extension
    name = Path(song_filename).stem
    # Remove the song_proposal_ prefix
    name = re.sub(r"^song_proposal_", "", name)
    # Remove hex color codes
    name = re.sub(r"\(0x[0-9a-fA-F]+\)", "", name)
    # Clean up
    name = re.sub(r"[^\w]+", "_", name).strip("_").lower()
    return name


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_chord_pipeline(
    thread_dir: str,
    song_filename: str,
    seed: int = 42,
    num_candidates: int = 200,
    top_k: int = 10,
    progression_length: int = 4,
    theory_weight: float = 0.3,
    chromatic_weight: float = 0.7,
    onnx_path: Optional[str] = None,
    strum_patterns: Optional[list[str]] = None,
):
    """Run the chord generation pipeline end-to-end.

    1. Load song proposal
    2. Generate chord candidates via Markov chains
    3. Score with theory + ChromaticScorer
    4. Write top-k as MIDI files
    5. Write review.yml
    """
    import random as _random

    rng = _random.Random(seed)
    _random.seed(seed)
    np.random.seed(seed)

    thread_path = Path(thread_dir)
    if not thread_path.exists():
        print(f"ERROR: Thread directory not found: {thread_path}")
        sys.exit(1)

    # --- 1. Load song proposal ---
    print("=" * 60)
    print("CHORD GENERATION PIPELINE")
    print("=" * 60)

    song_info = load_song_proposal(thread_path, song_filename)
    print(f"Song:    {song_filename}")
    print(f"Thread:  {thread_path.name}")
    print(f"Key:     {song_info['key_root']} {song_info['mode']}")
    print(f"BPM:     {song_info['bpm']}")
    print(f"Color:   {song_info['color_name']}")
    print(f"Time:    {song_info['time_sig'][0]}/{song_info['time_sig'][1]}")

    target = get_chromatic_target(song_info["color_name"])
    print(f"Target:  temporal={target['temporal']}")
    print(f"         spatial={target['spatial']}")
    print(f"         ontological={target['ontological']}")

    # Infer genre families for scratch beat generation
    from app.generators.midi.drum_patterns import (
        DEFAULT_GENRE_FAMILY,
        map_genres_to_families,
    )

    raw_proposal = song_info.get("raw_proposal", {})
    genre_tags = raw_proposal.get("genres", []) or []
    genre_families = map_genres_to_families(genre_tags) or [DEFAULT_GENRE_FAMILY]

    # --- 2. Generate chord candidates ---
    print(f"\nGenerating {num_candidates} candidates (seed={seed})...")
    gen = ChordProgressionGenerator()

    raw_candidates = gen.generate_progression_brute_force(
        key_root=song_info["key_root"],
        mode=song_info["mode"],
        length=progression_length,
        num_candidates=num_candidates,
        top_k=num_candidates,  # Keep all for chromatic scoring
        use_graph=True,
    )

    print(f"  Generated {len(raw_candidates)} valid candidates")
    if len(raw_candidates) == 0:
        print(
            "ERROR: No valid candidates generated. Check key/mode availability in chord database."
        )
        sys.exit(1)

    # --- 3. Score with ChromaticScorer ---
    print("\nScoring with ChromaticScorer...")

    from training.chromatic_scorer import ChromaticScorer

    scorer = ChromaticScorer(onnx_path=onnx_path) if onnx_path else ChromaticScorer()

    # Prepare concept embedding once
    concept_text = song_info["concept"]
    if not concept_text:
        concept_text = f"{song_info['color_name']} chromatic concept"
        print(f"  Warning: No concept text found, using fallback: '{concept_text}'")

    concept_emb = scorer.prepare_concept(concept_text)
    print(f"  Concept encoded ({concept_emb.shape[0]}-dim)")

    # Convert all candidates to MIDI bytes for scoring
    midi_candidates = []
    for theory_score, progression, theory_breakdown in raw_candidates:
        midi_bytes = progression_to_midi_bytes(progression, bpm=song_info["bpm"])
        midi_candidates.append(
            {
                "midi_bytes": midi_bytes,
                "theory_score": theory_score,
                "theory_breakdown": theory_breakdown,
                "progression": progression,
            }
        )

    # Batch score with ChromaticScorer
    scorer_candidates = [{"midi_bytes": c["midi_bytes"]} for c in midi_candidates]
    scorer_results = scorer.score_batch(scorer_candidates, concept_emb=concept_emb)

    # Build scorer lookup by candidate reference (the candidate dict is preserved in results)
    # scorer_results are sorted by confidence, but we need to match back to our candidates
    # Use the midi_bytes reference to match
    scorer_by_midi = {}
    for result in scorer_results:
        midi_key = id(result["candidate"]["midi_bytes"])
        scorer_by_midi[midi_key] = result

    # --- 4. Composite scoring ---
    print(
        f"\nComputing composite scores (theory={theory_weight}, chromatic={chromatic_weight})..."
    )

    scored = []
    for i, cand in enumerate(midi_candidates):
        midi_key = id(scorer_candidates[i]["midi_bytes"])
        scorer_result = scorer_by_midi.get(midi_key)
        if scorer_result is None:
            continue

        chromatic_match = compute_chromatic_match(scorer_result, target)
        comp, breakdown = composite_score(
            cand["theory_score"],
            cand["theory_breakdown"],
            chromatic_match,
            scorer_result,
            theory_weight,
            chromatic_weight,
        )

        scored.append(
            {
                "composite": comp,
                "breakdown": breakdown,
                "progression": cand["progression"],
                "midi_bytes": cand["midi_bytes"],
                "summary": progression_summary(cand["progression"]),
            }
        )

    # Rank by composite score
    scored.sort(key=lambda x: x["composite"], reverse=True)
    top_candidates = scored[:top_k]

    # Assign IDs and ranks
    for rank, item in enumerate(top_candidates):
        item["rank"] = rank + 1
        item["id"] = f"chord_{rank + 1:03d}"

    # --- 5. Write MIDI files (with HR + strum baked in) + scratch beats ---
    slug = song_slug(song_filename)
    output_dir = thread_path / "production" / slug / "chords"
    candidates_dir = output_dir / "candidates"
    approved_dir = output_dir / "approved"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    approved_dir.mkdir(parents=True, exist_ok=True)

    time_sig = tuple(song_info["time_sig"])
    print(
        f"\nWriting {len(top_candidates)} chord primitives + scratch beats to {candidates_dir}/"
    )
    for item in top_candidates:
        n_chords = len(item["progression"])
        hr_dist = sample_hr_distribution(n_chords, rng)
        strum_pat = sample_strum_pattern(time_sig, rng, filter_names=strum_patterns)
        bar_count = int(sum(hr_dist))

        primitive_bytes = progression_to_primitive_midi_bytes(
            item["progression"], song_info["bpm"], time_sig, hr_dist, strum_pat
        )
        scratch_bytes = generate_scratch_beat(
            song_info["bpm"], bar_count, time_sig, genre_families
        )

        write_midi_file(primitive_bytes, candidates_dir / f"{item['id']}.mid")
        write_midi_file(scratch_bytes, candidates_dir / f"{item['id']}_scratch.mid")

        item["hr_distribution"] = hr_dist
        item["strum_pattern"] = strum_pat.name

    # --- 6. Write review YAML ---
    review = generate_review_yaml(
        song_info,
        top_candidates,
        seed,
        {"theory": theory_weight, "chromatic": chromatic_weight},
    )
    review_path = output_dir / "review.yml"
    with open(review_path, "w") as f:
        yaml.dump(
            review, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    print(f"Review file: {review_path}")

    # --- 7. Summary ---
    print(f"\n{'=' * 60}")
    print(f"TOP {len(top_candidates)} CANDIDATES")
    print(f"{'=' * 60}")
    for item in top_candidates:
        print(
            f"\n  #{item['rank']} [{item['id']}] composite={item['breakdown']['composite']:.3f}"
        )
        print(
            f"     theory={item['breakdown']['theory_total']:.3f}  chromatic={item['breakdown']['chromatic']['match']:.3f}  confidence={item['breakdown']['chromatic']['confidence']:.3f}"
        )
        print(f"     {item['summary']}")
        print(
            f"     HR: {item.get('hr_distribution', [])}  strum: {item.get('strum_pattern', '?')}"
        )

    print(f"\nOutput: {output_dir}")
    print(f"Next: Edit {review_path} to label and approve candidates")
    print(f"Then: python -m app.generators.midi.promote_part --review {review_path}")

    return top_candidates


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Chord generation pipeline — generate, score, and review chord progressions"
    )
    parser.add_argument(
        "--thread", required=True, help="Shrinkwrapped thread directory"
    )
    parser.add_argument("--song", required=True, help="Song proposal YAML filename")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=200,
        help="Number of candidates to generate (default: 200)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top candidates to keep (default: 10)",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=4,
        help="Chord progression length in bars (default: 4)",
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
    parser.add_argument(
        "--strum-patterns",
        default=None,
        help="Comma-separated strum pattern names to use (e.g. whole,half,quarter). Default: all patterns for the time signature.",
    )

    args = parser.parse_args()

    strum_filter = (
        [p.strip() for p in args.strum_patterns.split(",") if p.strip()]
        if args.strum_patterns
        else None
    )

    run_chord_pipeline(
        thread_dir=args.thread,
        song_filename=args.song,
        seed=args.seed,
        num_candidates=args.num_candidates,
        top_k=args.top_k,
        progression_length=args.length,
        theory_weight=args.theory_weight,
        chromatic_weight=args.chromatic_weight,
        onnx_path=args.onnx_path,
        strum_patterns=strum_filter,
    )


if __name__ == "__main__":
    main()
