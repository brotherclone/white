#!/usr/bin/env python3
"""
Chord generation pipeline for the Music Production Pipeline.

Reads a song proposal from a shrink_wrapped thread, generates chord progression
candidates via Markov chains, scores them with both music theory metrics and
Refractor, and writes the top candidates as MIDI files with a review YAML.

Usage:
    python -m app.generators.midi.pipelines.chord_pipeline \
        --thread shrink_wrapped/white-the-breathing-machine-learns-to-sing \
        --song "song_proposal_Black (0x221f20)_sequential_dissolution_v2.yml" \
        --seed.logicx 42 --num-candidates 200 --top-k 10
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

from white_composition.init_production import load_initial_proposal
from white_core.concepts.chromatic_targets import (
    CHROMATIC_TARGETS,
    ONTOLOGICAL_MODES,
    SPATIAL_MODES,
    TEMPORAL_MODES,
)
from white_core.music.core.enharmonic import normalize_to_flat
from white_generation.artist_catalog import load_artist_context
from white_generation.chord_generator.generator import ChordProgressionGenerator
from white_generation.patterns.harmonic_rhythm import enumerate_distributions
from white_generation.patterns.strum_patterns import (
    StrumPattern,
    get_patterns_for_time_sig,
    strum_to_midi_bytes,
)
from white_generation.util.midi_cleanup import trim_midi_tempo_track as _trim_midi


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
    the chord prototype's convention.  Enharmonic roots are normalised to
    the spelling used in the chord database (e.g. A# → Bb, D# → Eb).
    """
    key_str = key_str.strip()
    # Handle unicode symbols
    key_str = key_str.replace("♭", "b").replace("♯", "#")

    parts = key_str.split()
    if len(parts) < 2:
        return parts[0] if parts else "C", "Major"

    root = parts[0]
    mode_str = " ".join(parts[1:]).lower()

    # Normalise enharmonic root to flat spelling used by the chord database
    root = normalize_to_flat(root)

    _minor_modes = {
        "minor",
        "min",
        "aeolian",
        "dorian",
        "phrygian",
        "locrian",
        "harmonic_minor",
        "melodic_minor",
    }
    if mode_str in _minor_modes or "minor" in mode_str:
        mode = "Minor"
    else:
        mode = "Major"

    return root, mode


def load_song_proposal(thread_dir: Path, song_filename: str) -> dict:
    """Load and parse a song proposal from a shrink_wrapped thread.

    Returns dict with: key_root, mode, bpm, time_sig (tuple), concept,
    color_name, singer, song_filename, thread_dir, raw_proposal.

    Delegates to load_song_proposal_unified for all parsing; remaps fields
    to the chord_pipeline convention (color_name, time_sig as tuple).
    """
    from white_composition.production_plan import (
        load_song_proposal_unified,
    )

    song_path = Path(thread_dir) / "yml" / song_filename
    if not song_path.exists():
        raise FileNotFoundError(f"Song proposal not found: {song_path}")

    unified = load_song_proposal_unified(song_path, thread_dir=Path(thread_dir))

    # chord_pipeline uses time_sig as a tuple and color_name (not color)
    ts_parts = unified["time_sig"].split("/")
    time_sig_tuple = (int(ts_parts[0]), int(ts_parts[1]))

    return {
        "key_root": normalize_to_flat(unified["key_root"]),
        "mode": unified["mode"],
        "bpm": unified["bpm"],
        "time_sig": time_sig_tuple,
        "concept": unified["concept"],
        "color_name": unified["color"],
        "singer": unified["singer"],
        "sub_proposals": unified.get("sub_proposals") or [],
        "song_filename": song_filename,
        "thread_dir": str(thread_dir),
        "raw_proposal": unified["raw_proposal"],
        "musical_constraints": unified.get("musical_constraints"),
    }


# ---------------------------------------------------------------------------
# MIDI export
# ---------------------------------------------------------------------------


def progression_to_midi_bytes(
    progression: list[dict],
    bpm: int = 120,
    ticks_per_beat: int = 480,
    time_sig: tuple[int, int] = (4, 4),
) -> bytes:
    """Convert a chord progression (list of chord dicts) to MIDI file bytes.

    Each chord is held for one bar. Bar length is derived from time_sig so
    3/4 songs get 3-beat bars rather than the old hardcoded 4-beat default.
    """
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set tempo
    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    # One bar per chord — beats per bar derived from time signature
    beats_per_bar = time_sig[0] * (4.0 / time_sig[1])
    bar_ticks = int(ticks_per_beat * beats_per_bar)

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
    """Pick a random strum pattern for the given time signature.

    Arp patterns are excluded by default. Pass filter_names containing an arp
    pattern name (e.g. ["arp_up"]) to opt in explicitly.
    """
    patterns = get_patterns_for_time_sig(time_sig, filter_names)
    if filter_names is None:
        patterns = [p for p in patterns if not p.is_arpeggio]
    if not patterns:
        patterns = get_patterns_for_time_sig(time_sig)
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
    from white_generation.patterns.drum_patterns import (
        ALL_TEMPLATES,
        DEFAULT_GENRE_FAMILY,
        select_templates,
    )
    from white_generation.pipelines.drum_pipeline import drum_pattern_to_midi_bytes

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
    """Compute how well Refractor output matches the target distribution.

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
        source = item.get("source", "markov")
        is_diatonic = source == "diatonic"
        is_constrained = source == "constrained"
        if is_diatonic:
            notes = "Diatonic workhorse — clean triads, assign to verse/chorus"
        elif is_constrained:
            notes = f"Constrained — sequence from proposal: {item.get('harmonic_sequence', '')}"
        else:
            notes = ""
        entry = {
            "id": item["id"],
            "midi_file": f"candidates/{item['id']}.mid",
            "scratch_midi": f"candidates/{item['id']}_scratch.mid",
            "rank": item.get("rank"),
            "source": source,
            "scores": (
                _to_python(item.get("breakdown")) if item.get("breakdown") else None
            ),
            "hr_distribution": _to_python(item.get("hr_distribution", [])),
            "strum_pattern": item.get("strum_pattern", "whole"),
            "progression": item["summary"],
            "chords": [
                {
                    "function": c.get("function", "?"),
                    "name": c.get("chord_name", "?"),
                    "notes": c.get("note_names", []),
                }
                for c in item.get("progression") or []
            ],
            # Human annotation fields
            "label": None,
            "status": "pending",
            "notes": notes,
        }
        if is_constrained and item.get("harmonic_sequence"):
            entry["harmonic_sequence"] = item["harmonic_sequence"]
        if item.get("bar_sources"):
            entry["bar_sources"] = item["bar_sources"]
        candidates.append(entry)

    time_sig = song_info.get("time_sig", (4, 4))
    review = {
        "key": f"{song_info['key_root']} {song_info['mode'].lower()}",
        "bpm": song_info["bpm"],
        "time_sig": f"{time_sig[0]}/{time_sig[1]}",
        "color": song_info["color_name"],
        "thread": song_info.get("thread_dir", ""),
        "song_proposal": song_info.get("song_filename", ""),
        "generated": datetime.now(timezone.utc).isoformat(),
        "seed.logicx": seed,
        "scoring_weights": scoring_weights,
        "candidates": candidates,
    }
    if song_info.get("sub_proposals"):
        review["sub_proposals"] = song_info["sub_proposals"]
    return review


# ---------------------------------------------------------------------------
# Diatonic workhorse candidates
# ---------------------------------------------------------------------------

DIATONIC_PATTERNS: dict[str, dict[str, list[str]]] = {
    "Major": {
        "I_V_vi_IV": ["I", "V", "vi", "IV"],
        "I_IV_V": ["I", "IV", "V"],
        "I_vi_IV_V": ["I", "vi", "IV", "V"],
        "ii_V_I": ["II", "V", "I"],
    },
    "Minor": {
        "i_VII_VI_VII": ["i", "VII", "VI", "VII"],
        "i_VI_III_VII": ["i", "VI", "III", "VII"],
        "i_iv_v": ["i", "iv", "v"],
        "i_VI_VII_i": ["i", "VI", "VII", "i"],
    },
}


def build_diatonic_candidates(
    key_root: str,
    mode: str,
    bpm: int,
    time_sig: tuple[int, int],
    gen: "ChordProgressionGenerator",
    rng,
    genre_families: list[str],
) -> list[dict]:
    """Build diatonic workhorse candidates from the chord bank.

    One candidate per pattern. Patterns where any degree is missing from the
    bank are skipped silently. Returns candidate dicts ready for MIDI export
    and review.yml serialisation.
    """
    pattern_set = DIATONIC_PATTERNS.get(mode, {})
    candidates = []

    for pattern_name, degrees in pattern_set.items():
        progression = []
        for degree in degrees:
            rows = gen.get_chord_by_function(key_root, mode, degree, category="triad")
            if rows.is_empty():
                # fall back to any category if no triad exists for this degree
                rows = gen.get_chord_by_function(key_root, mode, degree)
            if rows.is_empty():
                progression = []
                break
            row = rows.row(0, named=True)
            progression.append(
                {
                    "chord_id": row["chord_id"],
                    "chord_name": row["chord_name"],
                    "function": row["function"],
                    "note_names": row["note_names"],
                    "midi_notes": list(row["midi_notes"]),
                    "quality": row["quality"],
                }
            )
        if not progression:
            continue

        n = len(progression)
        hr_dist = sample_hr_distribution(n, rng)
        strum_pat = sample_strum_pattern(time_sig, rng)
        bar_count = int(sum(hr_dist))

        midi_bytes = progression_to_primitive_midi_bytes(
            progression, bpm, time_sig, hr_dist, strum_pat
        )
        scratch_bytes = generate_scratch_beat(bpm, bar_count, time_sig, genre_families)

        candidates.append(
            {
                "id": f"diatonic_{pattern_name}",
                "midi_bytes": midi_bytes,
                "scratch_bytes": scratch_bytes,
                "progression": progression,
                "summary": " – ".join(
                    f"{c['function']}({c['chord_name']})" for c in progression
                ),
                "hr_distribution": hr_dist,
                "strum_pattern": strum_pat.name,
                "source": "diatonic",
            }
        )

    return candidates


# ---------------------------------------------------------------------------
# Constrained candidates — harmonic_sequence from proposal
# ---------------------------------------------------------------------------


def build_constrained_candidates(
    harmonic_sequence: str,
    key_root: str,
    mode: str,
    bpm: int,
    time_sig: tuple[int, int],
    gen: "ChordProgressionGenerator",
    rng,
    genre_families: list[str],
    num_candidates: int,
    scorer,
    concept_emb,
    target: dict,
    theory_weight: float,
    chromatic_weight: float,
) -> list[dict]:
    """Build and score candidates from a fixed harmonic sequence.

    Parses ``harmonic_sequence`` (e.g. ``"i iv i"``) into Roman numeral tokens,
    resolves each to its chord pool, and generates ``num_candidates`` voicing
    combinations by sampling one chord per position per candidate.  Each
    combination is scored through the same theory+Refractor composite pipeline
    used for Markov candidates.

    Returns scored candidates sorted by composite score with
    ``source: "constrained"`` and ``harmonic_sequence`` echoed on each entry.
    Tokens that resolve to no chords are skipped with a warning; returns ``[]``
    only when no tokens resolve to usable chords.
    """
    tokens = harmonic_sequence.split()
    if not tokens:
        return []

    # Resolve each token to its chord pool (prefer triads; fall back to any)
    chord_pools: list[list[dict]] = []
    for token in tokens:
        rows = gen.get_chord_by_function(key_root, mode, token, category="triad")
        if rows.is_empty():
            rows = gen.get_chord_by_function(key_root, mode, token)
        if rows.is_empty():
            print(
                f"  Warning: constrained token '{token}' has no chords in "
                f"{key_root} {mode} — skipping token"
            )
            continue
        chord_pools.append(rows.to_dicts())

    if not chord_pools:
        print("  Warning: constrained sequence resolved to no usable chords — skipping")
        return []

    # Generate voicing combinations
    raw_candidates = []
    for _ in range(num_candidates):
        progression = []
        for pool in chord_pools:
            chord = rng.choice(pool)
            progression.append(
                {
                    "chord_id": chord["chord_id"],
                    "chord_name": chord["chord_name"],
                    "function": chord["function"],
                    "note_names": chord["note_names"],
                    "midi_notes": list(chord["midi_notes"]),
                    "quality": chord["quality"],
                }
            )
        theory_score, theory_breakdown = gen.score_progression(
            progression, key_root, mode
        )
        midi_bytes = progression_to_midi_bytes(progression, bpm=bpm)
        raw_candidates.append(
            {
                "midi_bytes": midi_bytes,
                "theory_score": theory_score,
                "theory_breakdown": theory_breakdown,
                "progression": progression,
            }
        )

    # Batch score with Refractor
    scorer_inputs = [{"midi_bytes": c["midi_bytes"]} for c in raw_candidates]
    scorer_results = scorer.score_batch(scorer_inputs, concept_emb=concept_emb)
    scorer_by_midi = {id(r["candidate"]["midi_bytes"]): r for r in scorer_results}

    scored = []
    for i, cand in enumerate(raw_candidates):
        midi_key = id(scorer_inputs[i]["midi_bytes"])
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
                "source": "constrained",
                "harmonic_sequence": harmonic_sequence,
            }
        )

    scored.sort(key=lambda x: x["composite"], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Song slug for directory naming
# ---------------------------------------------------------------------------


def song_slug(song_filename: str) -> str:
    """Convert song proposal filename to a directory-safe slug."""
    # Remove file extension
    name = Path(song_filename).stem
    # Remove the song_proposal_ prefix
    name = re.sub(r"^song_proposal_", "", name)
    # Remove hex color codes (with surrounding whitespace)
    name = re.sub(r"\s*\(0x[0-9a-fA-F]+\)\s*", "", name)
    # Clean up — collapse any run of non-word chars to a single underscore
    name = re.sub(r"[^\w]+", "_", name).strip("_").lower()
    return name


# ---------------------------------------------------------------------------
# White synthesis — donor-mode candidate generation
# ---------------------------------------------------------------------------


def is_white_mode(song_info: dict) -> bool:
    """Return True when the song is White (donor + cut-up mode)."""
    return str(song_info.get("color_name", "")).strip().capitalize() == "White"


def generate_white_candidates(
    song_info: dict,
    bar_pool: list[dict],
    num_candidates: int,
    top_k: int,
    seed: int,
    progression_length: int,
    theory_weight: float,
    chromatic_weight: float,
    onnx_path: Optional[str],
) -> list[dict]:
    """Generate White chord candidates by random cut-up of the bar pool.

    Each candidate draws `progression_length` bars from the pool with
    replacement, shuffles them, concatenates into a MIDI, and scores with
    Refractor + theory metrics.  Returns a ranked list of top_k dicts
    matching the format expected by generate_review_yaml().
    """
    import random as _random

    from white_analysis.refractor import Refractor
    from white_generation.pipelines.white_rebracketing import concatenate_bars

    rng = _random.Random(seed)
    tpb = 480

    if not bar_pool:
        raise ValueError("generate_white_candidates: bar_pool is empty")

    ts = song_info.get("time_sig", (4, 4))
    beats_per_bar = ts[0] * (4.0 / ts[1])

    scorer = Refractor(onnx_path=onnx_path) if onnx_path else Refractor()
    target = get_chromatic_target(song_info["color_name"])

    concept_text = (
        song_info.get("concept") or f"{song_info['color_name']} chromatic concept"
    )
    concept_emb = scorer.prepare_concept(concept_text)

    bpm = song_info["bpm"]
    raw_candidates = []

    for _ in range(num_candidates):
        drawn = rng.choices(bar_pool, k=progression_length)
        rng.shuffle(drawn)
        midi_bytes = concatenate_bars(
            [b["midi_bytes"] for b in drawn], tpb, bpm, beats_per_bar=beats_per_bar
        )
        raw_candidates.append(
            {
                "midi_bytes": midi_bytes,
                "bar_sources": [
                    {
                        "source_dir": b["source_dir"],
                        "donor_color": b["donor_color"],
                        "source_file": b["source_file"],
                        "bar_index": b["bar_index"],
                    }
                    for b in drawn
                ],
            }
        )

    # Score with Refractor
    scorer_results = scorer.score_batch(
        [{"midi_bytes": c["midi_bytes"]} for c in raw_candidates],
        concept_emb=concept_emb,
    )
    scorer_by_idx = {id(r["candidate"]["midi_bytes"]): r for r in scorer_results}

    scored = []
    for cand in raw_candidates:
        sr = scorer_by_idx.get(id(cand["midi_bytes"]))
        if sr is None:
            continue
        chromatic_match = compute_chromatic_match(sr, target)
        # Theory score is 0 for cut-up candidates (no chord-function analysis available)
        theory_score = 0.0
        theory_breakdown = {
            "melody": 0.0,
            "voice_leading": 0.0,
            "variety": 0.0,
            "graph_probability": 0.0,
        }
        comp, breakdown = composite_score(
            theory_score,
            theory_breakdown,
            chromatic_match,
            sr,
            theory_weight,
            chromatic_weight,
        )
        scored.append(
            {
                "composite": comp,
                "breakdown": breakdown,
                "midi_bytes": cand["midi_bytes"],
                "bar_sources": cand["bar_sources"],
                # White candidates have no chord-function progression; use bar sources as summary
                "progression": [],
                "summary": f"cut-up ({len(cand['bar_sources'])} bars from "
                f"{len({b['donor_color'] for b in cand['bar_sources']})} colors)",
            }
        )

    scored.sort(key=lambda x: x["composite"], reverse=True)
    top = scored[:top_k]
    for rank, item in enumerate(top):
        item["rank"] = rank + 1
        item["id"] = f"chord_{rank + 1:03d}"

    return top


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
    sub_proposals: Optional[list[str]] = None,
):
    """Run the chord generation pipeline end-to-end.

    1. Load song proposal
    2. Generate chord candidates via Markov chains
    3. Score with theory + Refractor
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
    from white_generation.patterns.drum_patterns import (
        DEFAULT_GENRE_FAMILY,
        map_genres_to_families,
    )

    raw_proposal = song_info.get("raw_proposal", {})
    genre_tags = raw_proposal.get("genres", []) or []

    # Resolve production dir early so we can read initial_proposal.yml
    _prod_slug = song_slug(song_filename)
    _production_dir = thread_path / "production" / _prod_slug
    _initial = load_initial_proposal(_production_dir)

    # Print style context from artist catalog (informational for human reviewing candidates)
    # Prefer sounds_like from initial_proposal.yml (Claude-generated); fall back to proposal YAML
    sounds_like = _initial.get("sounds_like") or raw_proposal.get("sounds_like") or []
    if sounds_like:
        artist_context = load_artist_context(sounds_like)
        if artist_context:
            print(f"\n{artist_context}\n")
    genre_families = map_genres_to_families(genre_tags) or [DEFAULT_GENRE_FAMILY]

    # --- 2. Generate chord candidates ---
    # Merge CLI sub_proposals into song_info (CLI takes precedence over proposal YAML)
    if sub_proposals:
        song_info["sub_proposals"] = list(sub_proposals)

    if is_white_mode(song_info):
        # White donor + cut-up mode
        from white_generation.pipelines.white_rebracketing import build_bar_pool

        sub_dirs = [Path(p) for p in song_info.get("sub_proposals", [])]
        if not sub_dirs:
            print(
                "ERROR: White mode requires sub_proposals (via --sub-proposals or proposal YAML)"
            )
            sys.exit(1)

        white_key = f"{song_info['key_root']} {song_info['mode'].lower()}"
        print(
            f"\nWhite synthesis mode — building bar pool from {len(sub_dirs)} sub-proposal(s)..."
        )
        for p in sub_dirs:
            if not Path(p).exists() or not (Path(p) / "chords" / "review.yml").exists():
                print(
                    f"ERROR: sub-proposal path missing or has no chords/review.yml: {p}"
                )
                sys.exit(1)

        ts = tuple(song_info.get("time_sig", (4, 4)))
        target_beats_per_bar = ts[0] * (4.0 / ts[1])
        bar_pool = build_bar_pool(
            sub_dirs, white_key, song_info["bpm"], beats_per_bar=target_beats_per_bar
        )
        print(f"  Bar pool: {len(bar_pool)} bars from {len(sub_dirs)} sub-proposal(s)")

        print(
            f"\nGenerating {num_candidates} cut-up candidates (seed.logicx={seed})..."
        )
        top_candidates = generate_white_candidates(
            song_info=song_info,
            bar_pool=bar_pool,
            num_candidates=num_candidates,
            top_k=top_k,
            seed=seed,
            progression_length=progression_length,
            theory_weight=theory_weight,
            chromatic_weight=chromatic_weight,
            onnx_path=onnx_path,
        )

        # Write White candidate MIDIs directly (no HR/strum re-application needed —
        # bars already carry their original rhythm; just write + scratch beat)
        slug = song_slug(song_filename)
        output_dir = thread_path / "production" / slug / "chords"
        candidates_dir = output_dir / "candidates"
        approved_dir = output_dir / "approved"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        approved_dir.mkdir(parents=True, exist_ok=True)

        time_sig = tuple(song_info["time_sig"])
        print(
            f"\nWriting {len(top_candidates)} White cut-up candidates to {candidates_dir}/"
        )
        for item in top_candidates:
            write_midi_file(item["midi_bytes"], candidates_dir / f"{item['id']}.mid")
            _trim_midi(candidates_dir / f"{item['id']}.mid")
            bar_count = progression_length
            scratch_bytes = generate_scratch_beat(
                song_info["bpm"], bar_count, time_sig, genre_families
            )
            write_midi_file(scratch_bytes, candidates_dir / f"{item['id']}_scratch.mid")
            _trim_midi(candidates_dir / f"{item['id']}_scratch.mid")

        review = generate_review_yaml(
            song_info,
            top_candidates,
            seed,
            {"theory": theory_weight, "chromatic": chromatic_weight},
        )
        review_path = output_dir / "review.yml"
        with open(review_path, "w") as f:
            yaml.dump(
                review,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=float("inf"),
            )

        print(f"Review file: {review_path}")
        print(f"\nOutput: {output_dir}")
        print(f"Next: Edit {review_path} to label and approve candidates")
        print(
            f"Then: python -m app.generators.midi.production.promote_part --review {review_path}"
        )
        return top_candidates

    # --- Non-White: Markov generation ---
    print(f"\nGenerating {num_candidates} candidates (seed.logicx={seed})...")
    gen = ChordProgressionGenerator()

    resolved_key = gen.resolve_key(song_info["key_root"], song_info["mode"])
    if resolved_key != song_info["key_root"]:
        print(
            f"  Enharmonic: {song_info['key_root']} → {resolved_key} "
            f"(no {song_info['mode']} chords for {song_info['key_root']})"
        )
        song_info["key_root"] = resolved_key

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

    # --- 3. Score with Refractor ---
    print("\nScoring with Refractor...")

    from white_analysis.refractor import Refractor

    scorer = Refractor(onnx_path=onnx_path) if onnx_path else Refractor()

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

    # Batch score with Refractor
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

    # --- 5. Build and score diatonic candidates alongside Markov ---
    # Build diatonic BEFORE sorting so they can compete for rank positions.
    time_sig = tuple(song_info["time_sig"])
    diatonic = build_diatonic_candidates(
        song_info["key_root"],
        song_info["mode"],
        song_info["bpm"],
        time_sig,
        gen,
        rng,
        genre_families,
    )

    if diatonic:
        print(f"\nScoring {len(diatonic)} diatonic candidates with Refractor...")
        # Use simple MIDI (no HR/strum) for scoring consistency with Markov candidates
        diatonic_score_inputs = [
            {
                "midi_bytes": progression_to_midi_bytes(
                    d["progression"], bpm=song_info["bpm"]
                )
            }
            for d in diatonic
        ]
        diatonic_results = scorer.score_batch(
            diatonic_score_inputs, concept_emb=concept_emb
        )
        diatonic_by_midi = {
            id(r["candidate"]["midi_bytes"]): r for r in diatonic_results
        }

        for i, d in enumerate(diatonic):
            sr = diatonic_by_midi.get(id(diatonic_score_inputs[i]["midi_bytes"]))
            if sr is None:
                continue
            theory_score, theory_breakdown = gen.score_progression(
                d["progression"], song_info["key_root"], song_info["mode"]
            )
            chromatic_match = compute_chromatic_match(sr, target)
            comp, breakdown = composite_score(
                theory_score,
                theory_breakdown,
                chromatic_match,
                sr,
                theory_weight,
                chromatic_weight,
            )
            scored.append(
                {
                    "composite": comp,
                    "breakdown": breakdown,
                    "progression": d["progression"],
                    "midi_bytes": d["midi_bytes"],  # already HR+strum baked
                    "scratch_bytes": d["scratch_bytes"],
                    "summary": d["summary"],
                    "hr_distribution": d["hr_distribution"],
                    "strum_pattern": d["strum_pattern"],
                    "source": "diatonic",
                    "id": d["id"],
                }
            )

    # --- 5b. Build and score constrained candidates (harmonic_sequence from proposal) ---
    constraints = song_info.get("musical_constraints")
    _harmonic_sequence = (
        getattr(constraints, "harmonic_sequence", None) if constraints else None
    )
    if _harmonic_sequence:
        print(
            f"\nBuilding constrained candidates for sequence: '{_harmonic_sequence}'..."
        )
        constrained = build_constrained_candidates(
            harmonic_sequence=_harmonic_sequence,
            key_root=song_info["key_root"],
            mode=song_info["mode"],
            bpm=song_info["bpm"],
            time_sig=time_sig,
            gen=gen,
            rng=rng,
            genre_families=genre_families,
            num_candidates=num_candidates,
            scorer=scorer,
            concept_emb=concept_emb,
            target=target,
            theory_weight=theory_weight,
            chromatic_weight=chromatic_weight,
        )
        if constrained:
            print(f"  {len(constrained)} constrained candidates generated")
            scored.extend(constrained)

    # Merge, re-sort, and select final pool.
    # Markov candidates are capped at top_k; diatonic always included in full.
    # Constrained candidates are always included (like diatonic — human-directed).
    # Final list is sorted by composite so ranks truly reflect scores.
    scored.sort(key=lambda x: x["composite"], reverse=True)
    diatonic_ids = {d["id"] for d in diatonic}
    # Constrained and diatonic candidates are always included in full;
    # Markov candidates are capped at top_k.
    _always_include = {"diatonic", "constrained"}
    markov_in_scored = [
        c
        for c in scored
        if c.get("source") not in _always_include and c.get("id") not in diatonic_ids
    ][:top_k]
    special_candidates = [
        c
        for c in scored
        if c.get("source") in _always_include or c.get("id") in diatonic_ids
    ]
    top_candidates = sorted(
        markov_in_scored + special_candidates,
        key=lambda x: x["composite"],
        reverse=True,
    )

    # Assign global ranks in composite order; Markov/constrained IDs are sequential
    # among non-diatonic entries; diatonic entries keep their pattern-based IDs.
    markov_counter = 0
    for rank, item in enumerate(top_candidates, 1):
        item["rank"] = rank
        if item.get("source") != "diatonic" and not item.get("id"):
            markov_counter += 1
            item["id"] = f"chord_{markov_counter:03d}"
        elif item.get("source") not in _always_include:
            markov_counter += 1
            item["id"] = f"chord_{markov_counter:03d}"

    # --- 6. Write MIDI files for all candidates ---
    slug = song_slug(song_filename)
    output_dir = thread_path / "production" / slug / "chords"
    candidates_dir = output_dir / "candidates"
    approved_dir = output_dir / "approved"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    approved_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\nWriting {len(top_candidates)} chord primitives + scratch beats to {candidates_dir}/"
    )
    for item in top_candidates:
        is_diatonic = item.get("source") == "diatonic"
        if is_diatonic:
            # midi_bytes already has HR+strum baked from build_diatonic_candidates
            write_midi_file(item["midi_bytes"], candidates_dir / f"{item['id']}.mid")
            _trim_midi(candidates_dir / f"{item['id']}.mid")
            write_midi_file(
                item["scratch_bytes"], candidates_dir / f"{item['id']}_scratch.mid"
            )
            _trim_midi(candidates_dir / f"{item['id']}_scratch.mid")
        else:
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
            _trim_midi(candidates_dir / f"{item['id']}.mid")
            write_midi_file(scratch_bytes, candidates_dir / f"{item['id']}_scratch.mid")
            _trim_midi(candidates_dir / f"{item['id']}_scratch.mid")

            item["hr_distribution"] = hr_dist
            item["strum_pattern"] = strum_pat.name

    # --- 7. Write review YAML ---
    review = generate_review_yaml(
        song_info,
        top_candidates,
        seed,
        {"theory": theory_weight, "chromatic": chromatic_weight},
    )
    # Surface performance_notes for the human reviewer (no pipeline effect)
    _perf_notes = (
        getattr(constraints, "performance_notes", None) if constraints else None
    )
    if _perf_notes:
        review["performance_notes"] = _perf_notes
    review_path = output_dir / "review.yml"
    with open(review_path, "w") as f:
        yaml.dump(
            review,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=float("inf"),
        )

    print(f"Review file: {review_path}")

    # --- 8. Summary ---
    print(f"\n{'=' * 60}")
    print(f"TOP {len(top_candidates)} CANDIDATES")
    print(f"{'=' * 60}")
    for item in top_candidates:
        if item.get("source") == "diatonic":
            print(f"\n  [diatonic] [{item['id']}]")
        else:
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
    print(
        f"Then: python -m app.generators.midi.production.promote_part --review {review_path}"
    )

    return top_candidates


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Chord generation pipeline — generate, score, and review chord progressions. "
            "Run init_production.py first to generate initial_proposal.yml with artist context."
        )
    )
    parser.add_argument(
        "--production-dir",
        default=None,
        help="Production directory (e.g. shrink_wrapped/.../production/song_slug). "
        "Derives --thread and --song automatically; takes precedence over both.",
    )
    parser.add_argument(
        "--thread", default=None, help="shrink_wrapped thread directory"
    )
    parser.add_argument("--song", default=None, help="Song proposal YAML filename")
    parser.add_argument(
        "--seed.logicx",
        dest="seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
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
        help="Path to refractor.onnx (default: training/data/refractor.onnx)",
    )
    parser.add_argument(
        "--strum-patterns",
        default=None,
        help="Comma-separated strum pattern names to use (e.g. whole,half,quarter). Default: all patterns for the time signature.",
    )
    parser.add_argument(
        "--sub-proposals",
        nargs="+",
        default=None,
        metavar="DIR",
        help="(White mode) Production dirs to draw donor chord material from. "
        "Space-separated paths. Also readable from sub_proposals in the song proposal YAML.",
    )

    args = parser.parse_args()

    # Resolve --production-dir → --thread / --song
    thread_val = args.thread
    song_val = args.song
    if args.production_dir:
        prod_path = Path(args.production_dir)
        thread_val = str(prod_path.parent.parent)
        song_val = prod_path.name + ".yml"
    elif not thread_val or not song_val:
        parser.error(
            "Either --production-dir or both --thread and --song are required."
        )

    strum_filter = (
        [p.strip() for p in args.strum_patterns.split(",") if p.strip()]
        if args.strum_patterns
        else None
    )

    run_chord_pipeline(
        thread_dir=thread_val,
        song_filename=song_val,
        seed=args.seed,
        num_candidates=args.num_candidates,
        top_k=args.top_k,
        progression_length=args.length,
        theory_weight=args.theory_weight,
        chromatic_weight=args.chromatic_weight,
        onnx_path=args.onnx_path,
        strum_patterns=strum_filter,
        sub_proposals=args.sub_proposals,
    )


if __name__ == "__main__":
    main()
