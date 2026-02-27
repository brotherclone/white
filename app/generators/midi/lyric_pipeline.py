#!/usr/bin/env python3
"""
Lyric generation pipeline for the Music Production Pipeline.

Generates N complete lyric drafts (all vocal sections) via Claude API, scores
each with ChromaticScorer in text-only mode, computes a syllable fitting score
(syllables vs. melody notes) per section, and writes melody/lyrics_review.yml
(append-only). Integrates with promote_part.py to copy an approved .txt to
melody/lyrics.txt.

Pipeline position: chords → drums → bass → melody → LYRICS

Usage:
    python -m app.generators.midi.lyric_pipeline \\
        --production-dir shrinkwrapped/.../production/yellow__... \\
        --num-candidates 3

    # Register manually placed .txt files
    python -m app.generators.midi.lyric_pipeline \\
        --production-dir ... --sync-candidates
"""

import argparse
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mido
import yaml
from dotenv import load_dotenv

load_dotenv()

from app.generators.midi.chord_pipeline import (  # noqa: E402
    _to_python,
    compute_chromatic_match,
    get_chromatic_target,
)
from app.generators.midi.production_plan import load_plan  # noqa: E402
from app.generators.midi.song_evaluator import _count_syllables  # noqa: E402

LYRICS_REVIEW_FILENAME = "lyrics_review.yml"


# ---------------------------------------------------------------------------
# Note counting
# ---------------------------------------------------------------------------


def _count_notes(midi_path: Path) -> int:
    """Count note_on events with velocity > 0 across all tracks."""
    try:
        mid = mido.MidiFile(str(midi_path))
    except Exception:
        return 0
    count = 0
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                count += 1
    return count


# ---------------------------------------------------------------------------
# Vocal section reading
# ---------------------------------------------------------------------------


def _read_vocal_sections(plan, melody_dir: Path) -> list[dict]:
    """Extract vocal sections from the production plan with note counts and contours.

    Deduplicates by name (first occurrence wins).
    Looks up note counts from approved melody MIDI files.
    Reads contour from melody/review.yml for each approved label.
    """
    # Load melody review to get contour info
    melody_review_path = melody_dir / "review.yml"
    contour_by_label: dict[str, str] = {}
    if melody_review_path.exists():
        with open(melody_review_path) as f:
            melody_review = yaml.safe_load(f) or {}
        for cand in melody_review.get("candidates", []):
            label = cand.get("label")
            status = str(cand.get("status", "")).lower()
            if label and status in ("approved", "accepted"):
                label_clean = label.lower().replace("-", "_").replace(" ", "_")
                contour_by_label[label_clean] = cand.get("contour", "stepwise")

    # Collect all vocal occurrences grouped by name, preferring any that have a
    # melody loop set (later occurrences often have more loops filled in).
    from collections import defaultdict

    vocal_by_name: dict[str, list] = defaultdict(list)
    for section in plan.sections:
        if section.vocals:
            vocal_by_name[section.name].append(section)

    # Preserve order of first appearance, but pick the best occurrence per name
    seen_names: set[str] = set()
    sections = []
    for section in plan.sections:
        if not section.vocals:
            continue
        if section.name in seen_names:
            continue
        seen_names.add(section.name)
        # Prefer the occurrence with a melody loop; fall back to first
        best = next(
            (s for s in vocal_by_name[section.name] if s.loops.get("melody")),
            section,
        )

        # Find melody MIDI for note count (use best occurrence)
        midi_path: Optional[Path] = None
        melody_label = best.loops.get("melody")
        approved_dir = melody_dir / "approved"
        if melody_label:
            candidate = approved_dir / f"{melody_label}.mid"
            if candidate.exists():
                midi_path = candidate
        if midi_path is None:
            matches = sorted(approved_dir.glob(f"{best.name}*.mid"))
            if matches:
                midi_path = matches[0]

        if midi_path is not None:
            per_loop_notes = _count_notes(midi_path)
            total_notes = per_loop_notes * best.repeat
            approved_label = midi_path.stem
        else:
            total_notes = 0
            approved_label = None

        contour = contour_by_label.get(approved_label or best.name, "stepwise")

        sections.append(
            {
                "name": best.name,
                "bars": best.bars,
                "repeat": best.repeat,
                "total_notes": total_notes,
                "contour": contour,
                "approved_label": approved_label,
            }
        )

    return sections


# ---------------------------------------------------------------------------
# Syllable fitting
# ---------------------------------------------------------------------------

# Severity order for "worst wins" overall verdict
_VERDICT_ORDER = ["spacious", "paste-ready", "tight but workable", "splits needed"]


def _fitting_verdict(ratio: float) -> str:
    """Classify a syllables/notes ratio into a fitting verdict."""
    if ratio < 0.75:
        return "spacious"
    elif ratio <= 1.10:
        return "paste-ready"
    elif ratio <= 1.30:
        return "tight but workable"
    else:
        return "splits needed"


def _compute_fitting(candidate_text: str, vocal_sections: list[dict]) -> dict:
    """Compute syllable fitting for each vocal section.

    Parses [section_name] blocks from candidate_text, counts syllables per
    section (stripping # comment lines and [header] lines), and computes ratio
    against melody note counts.

    Returns:
        {
            section_name: {syllables, notes, ratio, verdict},
            ...,
            "overall": worst_verdict  # spacious treated same as paste-ready
        }
    """
    parsed = _parse_sections(candidate_text)
    result: dict = {}
    worst_idx = 0  # paste-ready is the baseline (index 1, but spacious maps there too)

    for sec in vocal_sections:
        name = sec["name"]
        notes = sec["total_notes"]
        lyric_text = parsed.get(name, "")
        syllable_count = sum(
            _count_syllables(line)
            for line in lyric_text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        if notes > 0:
            ratio = syllable_count / notes
        else:
            ratio = 1.0

        verdict = _fitting_verdict(ratio)
        # "spacious" treated same as "paste-ready" for overall ranking
        verdict_for_rank = verdict if verdict != "spacious" else "paste-ready"
        verdict_idx = _VERDICT_ORDER.index(verdict_for_rank)
        if verdict_idx > worst_idx:
            worst_idx = verdict_idx

        result[name] = {
            "syllables": syllable_count,
            "notes": notes,
            "ratio": round(ratio, 3),
            "verdict": verdict,
        }

    result["overall"] = _VERDICT_ORDER[worst_idx]
    return result


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _build_prompt(plan, vocal_sections: list[dict], syllable_targets: dict) -> str:
    """Build the Claude prompt for lyric generation.

    Includes song metadata, color target, and per-section syllable targets.
    """
    target = get_chromatic_target(plan.color)
    temporal_modes = ["past", "present", "future"]
    spatial_modes = ["thing", "place", "person"]
    ontological_modes = ["imagined", "forgotten", "known"]

    def dominant_mode(modes, dist):
        idx = max(range(len(dist)), key=lambda i: dist[i])
        return modes[idx]

    dominant_temporal = dominant_mode(temporal_modes, target["temporal"])
    dominant_spatial = dominant_mode(spatial_modes, target["spatial"])
    dominant_ontological = dominant_mode(ontological_modes, target["ontological"])

    lines = [
        f'You are writing lyrics for a song titled "{plan.title}".',
        "",
        "SONG METADATA:",
        f"  Color: {plan.color}",
        f"  BPM: {plan.bpm}",
        f"  Time signature: {plan.time_sig}",
        f"  Key: {plan.key}",
        f"  Concept: {plan.concept}",
        "",
        "CHROMATIC TARGET (the emotional/conceptual space to express):",
        f"  Temporal mode: {dominant_temporal}  "
        "(past=memory/history, present=immediacy, future=anticipation)",
        f"  Spatial mode: {dominant_spatial}  "
        "(thing=object/artifact, place=location/environment, person=human/being)",
        f"  Ontological mode: {dominant_ontological}  "
        "(imagined=fictional/possible, forgotten=lost/erased, known=certain/present)",
        "",
        f"Write lyrics that express the {plan.color} chromatic concept: "
        f"{dominant_temporal}, {dominant_spatial}, {dominant_ontological}.",
        "",
        "SECTIONS TO WRITE:",
    ]

    for sec in vocal_sections:
        name = sec["name"]
        lo, hi = syllable_targets.get(name, (0, 0))
        denom = max(sec["bars"] * sec["repeat"], 1)
        notes_per_bar = sec["total_notes"] / denom
        lines.extend(
            [
                "",
                f"  [{name}]",
                f"    Bars per occurrence: {sec['bars']}",
                f"    Repeats: {sec['repeat']}",
                f"    Melody contour: {sec['contour']}",
                f"    Target syllables: {lo}–{hi}  (≈{notes_per_bar:.1f} notes/bar)",
            ]
        )

    lines.extend(
        [
            "",
            "OUTPUT FORMAT:",
            "  Use [section_name] headers exactly as listed above.",
            "  Write one block per unique section name.",
            "  Output only the lyrics — no commentary, no explanations.",
            "  Lines starting with # are ignored (you may use them for stage directions).",
            "",
            "Example:",
            "  [verse]",
            "  First line of verse",
            "  Second line of verse",
            "",
            "  [chorus]",
            "  First line of chorus",
            "  Second line of chorus",
            "",
            "Now write the complete lyrics:",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


def _call_api(client, prompt: str, model: str) -> str:
    """Call the Anthropic API and return the generated text."""
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------


def _parse_sections(text: str) -> dict[str, str]:
    """Parse [section_name] headers from lyric text.

    Returns dict of section_name → lyric block (comment lines stripped).
    Section names are lowercased and spaces converted to underscores.
    """
    result: dict[str, str] = {}
    current_section: Optional[str] = None
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        header_match = re.match(r"^\[([^\]]+)\]\s*$", stripped)
        if header_match:
            if current_section is not None:
                result[current_section] = "\n".join(current_lines).strip()
            current_section = header_match.group(1).strip().lower().replace(" ", "_")
            current_lines = []
        elif current_section is not None:
            if not stripped.startswith("#"):
                current_lines.append(line)

    if current_section is not None:
        result[current_section] = "\n".join(current_lines).strip()

    return result


# ---------------------------------------------------------------------------
# Review YAML load / init
# ---------------------------------------------------------------------------


def _load_or_init_review(melody_dir: Path, plan, model: str, seed: int) -> dict:
    """Load existing lyrics_review.yml or create a fresh header dict."""
    review_path = melody_dir / LYRICS_REVIEW_FILENAME
    if review_path.exists():
        with open(review_path) as f:
            return yaml.safe_load(f) or {}

    # Deduplicate vocal section names preserving order
    seen: set[str] = set()
    unique_vocal: list[str] = []
    for s in plan.sections:
        if s.vocals and s.name not in seen:
            unique_vocal.append(s.name)
            seen.add(s.name)

    return {
        "production_dir": str(melody_dir.parent),
        "pipeline": "lyric-generation",
        "bpm": plan.bpm,
        "time_sig": plan.time_sig,
        "color": plan.color,
        "generated": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "model": model,
        "scoring_weights": {"chromatic": 1.0},
        "vocal_sections": unique_vocal,
        "candidates": [],
    }


# ---------------------------------------------------------------------------
# Candidate ID generation
# ---------------------------------------------------------------------------


def _next_candidate_id(review: dict) -> str:
    """Return the next candidate ID as 'lyrics_NN'.

    Inspects existing candidate ids, finds the max number, and increments.
    Returns 'lyrics_01' if no candidates exist.
    """
    max_n = 0
    for cand in review.get("candidates", []):
        cid = cand.get("id", "")
        m = re.match(r"lyrics_(\d+)$", cid)
        if m:
            n = int(m.group(1))
            if n > max_n:
                max_n = n
    return f"lyrics_{max_n + 1:02d}"


# ---------------------------------------------------------------------------
# Candidate sync
# ---------------------------------------------------------------------------


def sync_lyric_candidates(melody_dir: Path) -> int:
    """Scan melody/candidates/*.txt for untracked files and add stub entries.

    Safe to run after manually placing a .txt file. Does not wipe or
    regenerate anything.

    Returns:
        Number of new entries added.
    """
    review_path = melody_dir / LYRICS_REVIEW_FILENAME
    candidates_dir = melody_dir / "candidates"

    if not review_path.exists():
        print(f"ERROR: No lyrics_review.yml found at {review_path}")
        print("Run the lyric pipeline first to create a lyrics_review.yml base.")
        return 0

    with open(review_path) as f:
        review = yaml.safe_load(f) or {}

    existing_files = {
        Path(c["file"]).name for c in review.get("candidates", []) if c.get("file")
    }
    existing_ids = {c["id"] for c in review.get("candidates", []) if c.get("id")}

    if not candidates_dir.exists():
        print(f"No candidates/ directory at {candidates_dir}")
        return 0

    new_files = sorted(
        f for f in candidates_dir.glob("*.txt") if f.name not in existing_files
    )

    if not new_files:
        print("All candidate files are already tracked in lyrics_review.yml")
        return 0

    added = 0
    for txt_file in new_files:
        stub_id = txt_file.stem
        if stub_id in existing_ids:
            i = 2
            while f"{stub_id}_{i}" in existing_ids:
                i += 1
            stub_id = f"{stub_id}_{i}"

        stub = {
            "id": stub_id,
            "file": f"candidates/{txt_file.name}",
            "status": "pending",
            "notes": "",
        }
        review.setdefault("candidates", []).append(stub)
        existing_ids.add(stub_id)
        print(f"  + {txt_file.name}  →  id: {stub_id}")
        added += 1

    with open(review_path, "w") as f:
        yaml.dump(
            review, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    print(f"\nAdded {added} entries to lyrics_review.yml")
    print(f"Edit {review_path}")
    print("Set status: approved on the entry you want, then run promote_part")
    return added


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_lyric_pipeline(
    production_dir: str,
    num_candidates: int = 3,
    model: str = "claude-sonnet-4-6",
    seed: int = 42,
    onnx_path: Optional[str] = None,
    skip_scoring: bool = False,
) -> dict:
    """Run the lyric generation pipeline end-to-end.

    Returns:
        The lyrics_review.yml dict after writing.
    """
    prod_path = Path(production_dir)
    if not prod_path.exists():
        print(f"ERROR: Production directory not found: {prod_path}")
        sys.exit(1)

    melody_dir = prod_path / "melody"

    print("=" * 60)
    print("LYRIC GENERATION PIPELINE")
    print("=" * 60)

    # --- 1. Load production plan ---
    plan = load_plan(prod_path)
    if plan is None:
        print(f"ERROR: production_plan.yml not found in {prod_path}")
        sys.exit(1)

    print(f"Song:  {plan.title}")
    print(f"Color: {plan.color}")
    print(f"BPM:   {plan.bpm}  Time: {plan.time_sig}  Key: {plan.key}")

    # --- 2. Read vocal sections ---
    vocal_sections = _read_vocal_sections(plan, melody_dir)
    if not vocal_sections:
        print("ERROR: No vocal sections found in production_plan.yml")
        print("Set vocals: true on sections that need lyrics.")
        sys.exit(1)

    print(f"\nVocal sections ({len(vocal_sections)}):")
    for sec in vocal_sections:
        print(
            f"  {sec['name']}: {sec['bars']}b × {sec['repeat']}"
            f" = {sec['total_notes']} notes"
        )

    # --- 3. Syllable targets ---
    syllable_targets = {
        sec["name"]: (
            math.floor(sec["total_notes"] * 0.75),
            math.floor(sec["total_notes"] * 1.05),
        )
        for sec in vocal_sections
    }

    # --- 4. Build prompt ---
    prompt = _build_prompt(plan, vocal_sections, syllable_targets)

    # --- 5. Generate candidates ---
    from anthropic import Anthropic

    client = Anthropic()
    print(f"\nGenerating {num_candidates} lyric candidate(s) via {model}...")
    texts = []
    for i in range(num_candidates):
        print(f"  Candidate {i + 1}/{num_candidates}...")
        text = _call_api(client, prompt, model)
        texts.append(text)

    # --- 6. Score with ChromaticScorer (text-only) ---
    scorer_results_map: dict[int, Optional[dict]] = {}
    target = get_chromatic_target(plan.color)

    if not skip_scoring:
        print("\nLoading ChromaticScorer...")
        try:
            from training.chromatic_scorer import ChromaticScorer

            scorer = (
                ChromaticScorer(onnx_path=onnx_path) if onnx_path else ChromaticScorer()
            )
            concept_text = plan.concept or f"{plan.color} chromatic concept"
            concept_emb = scorer.prepare_concept(concept_text)
            print(f"  Concept encoded ({concept_emb.shape[0]}-dim)")

            scorer_candidates = [{"lyric_text": t} for t in texts]
            scorer_results = scorer.score_batch(
                scorer_candidates, concept_emb=concept_emb
            )
            for result in scorer_results:
                idx = scorer_candidates.index(result["candidate"])
                scorer_results_map[idx] = result
        except Exception as e:
            print(f"  Warning: ChromaticScorer unavailable ({e}), skipping scoring")
    else:
        print("\nSkipping ChromaticScorer (--skip-scoring)")

    # --- 7. Compute fitting + chromatic match ---
    scored_entries = []
    for idx, text in enumerate(texts):
        result = scorer_results_map.get(idx)
        chromatic_match = compute_chromatic_match(result, target) if result else 0.0
        fitting = _compute_fitting(text, vocal_sections)
        scored_entries.append(
            {
                "text": text,
                "original_idx": idx,
                "chromatic_result": result,
                "chromatic_match": chromatic_match,
                "fitting": fitting,
            }
        )

    # Sort by chromatic match descending (stable — preserves generation order on tie)
    scored_entries.sort(key=lambda e: e["chromatic_match"], reverse=True)

    # --- 8. Write candidate .txt files ---
    candidates_dir = melody_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    review = _load_or_init_review(melody_dir, plan, model, seed)

    new_entries = []
    for rank, entry in enumerate(scored_entries):
        cid = _next_candidate_id(review)
        txt_path = candidates_dir / f"{cid}.txt"
        txt_path.write_text(entry["text"], encoding="utf-8")

        result = entry["chromatic_result"]
        if result is not None:
            chromatic_block = _to_python(
                {
                    "temporal": result["temporal"],
                    "spatial": result["spatial"],
                    "ontological": result["ontological"],
                    "confidence": round(result["confidence"], 4),
                    "match": round(entry["chromatic_match"], 4),
                }
            )
        else:
            chromatic_block = None
        fitting_block = _to_python(entry["fitting"])

        candidate_entry = {
            "id": cid,
            "file": f"candidates/{cid}.txt",
            "rank": rank + 1,
            "chromatic": chromatic_block,
            "fitting": fitting_block,
            "status": "pending",
            "notes": "",
        }
        review.setdefault("candidates", []).append(candidate_entry)
        new_entries.append(candidate_entry)

    # --- 9. Save review YAML (append-only — existing entries preserved) ---
    review_path = melody_dir / LYRICS_REVIEW_FILENAME
    with open(review_path, "w") as f:
        yaml.dump(
            review, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    # --- 10. Summary ---
    print(f"\n{'=' * 60}")
    print("LYRIC GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Candidates: {len(new_entries)}")
    print(f"Review:     {review_path}")
    print()
    print(f"{'Rank':<5} {'ID':<12} {'Match':<8} {'Overall Fit'}")
    print("-" * 40)
    for entry in new_entries:
        match = entry["chromatic"]["match"]
        overall = entry["fitting"].get("overall", "?")
        print(f"  #{entry['rank']:<3} {entry['id']:<12} {match:.3f}    {overall}")

    print(f"\nNext: Edit {review_path} to approve a candidate")
    print(f"Then: python -m app.generators.midi.promote_part --review {review_path}")

    return review


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Lyric generation pipeline — generate, score, and review lyrics",
    )
    parser.add_argument(
        "--production-dir",
        required=True,
        help="Song production directory (must contain production_plan.yml)",
    )
    parser.add_argument(
        "--sync-candidates",
        action="store_true",
        help=(
            "Scan candidates/*.txt for files not in lyrics_review.yml and add stubs. "
            "Does not regenerate or wipe anything."
        ),
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=3,
        help="Number of lyric drafts to generate (default: 3)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Claude model to use (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for review header (default: 42)",
    )
    parser.add_argument(
        "--onnx-path",
        default=None,
        help="Path to fusion_model.onnx (default: training/data/fusion_model.onnx)",
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip ChromaticScorer (useful when torch/DeBERTa unavailable locally)",
    )

    args = parser.parse_args()

    if args.sync_candidates:
        melody_dir = Path(args.production_dir) / "melody"
        sync_lyric_candidates(melody_dir)
        return

    run_lyric_pipeline(
        production_dir=args.production_dir,
        num_candidates=args.num_candidates,
        model=args.model,
        seed=args.seed,
        onnx_path=args.onnx_path,
        skip_scoring=args.skip_scoring,
    )


if __name__ == "__main__":
    main()
