#!/usr/bin/env python3
"""
Lyric generation pipeline for the Music Production Pipeline.

Generates N complete lyric drafts (all vocal sections) via Claude API, scores
each with Refractor in text-only mode, computes a syllable fitting score
(syllables vs. melody notes) per section, and writes melody/lyrics_review.yml
(append-only). Integrates with promote_part.py to copy an approved .txt to
melody/lyrics.txt.

Vocal sections are derived from arrangement.txt (track 4 = melody = vocal).
Song metadata is read from the song proposal YAML via chords/review.yml.
No production_plan.yml is required.

Pipeline position: chords → drums → bass → melody → arrangement export → LYRICS

Usage:
    python -m app.generators.midi.pipelines.lyric_pipeline \\
        --production-dir shrink_wrapped/.../production/yellow__... \\
        --num-candidates 3

    # Register manually placed .txt files
    python -m app.generators.midi.pipelines.lyric_pipeline \\
        --production-dir ... --sync-candidates
"""

import argparse
import math
import re
import sys
import mido
import yaml

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from app.generators.artist_catalog import load_artist_context  # noqa: E402
from app.generators.midi.pipelines.chord_pipeline import (  # noqa: E402
    _to_python,
    compute_chromatic_match,
    get_chromatic_target,
)
from app.generators.midi.production.song_evaluator import _count_syllables  # noqa: E402

load_dotenv()

LYRICS_REVIEW_FILENAME = "lyrics_review.yml"

MELODY_CHANNEL = 4  # track 4 in arrangement.txt = melody = vocal


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
# Arrangement parser
# ---------------------------------------------------------------------------


def _parse_timecode_secs(tc: str) -> float:
    """Parse HH:MM:SS:FF.ff Logic timecode to seconds (30fps assumed)."""
    tc = tc.strip()
    parts = tc.split(":")
    if len(parts) != 4:
        return 0.0
    try:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        frames = float(parts[3])
        return h * 3600.0 + m * 60.0 + s + frames / 30.0
    except (ValueError, IndexError):
        return 0.0


def parse_arrangement(arrangement_path: Path) -> list[dict]:
    """Parse arrangement.txt into a list of clip dicts.

    Each dict: {timecode_secs, clip_name, channel, duration_secs}
    Lines are tab-separated: timecode  clip_name  channel  duration
    """
    clips = []
    with open(arrangement_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("\t") if p.strip()]
            if len(parts) < 4:
                continue
            try:
                timecode_secs = _parse_timecode_secs(parts[0])
                clip_name = parts[1]
                channel = int(parts[2])
                duration_secs = _parse_timecode_secs(parts[3])
                clips.append(
                    {
                        "timecode_secs": timecode_secs,
                        "clip_name": clip_name,
                        "channel": channel,
                        "duration_secs": duration_secs,
                    }
                )
            except (ValueError, IndexError):
                continue
    return clips


# ---------------------------------------------------------------------------
# Song proposal loader
# ---------------------------------------------------------------------------


def _find_and_load_proposal(production_dir: Path) -> dict:
    """Find and load the song proposal for a production directory.

    Reads thread + song_proposal from chords/review.yml to resolve the path.
    Returns a metadata dict with: title, bpm, time_sig, key, color, concept,
    genres, mood, singer, sounds_like (empty — not in proposal YAML).
    Returns {} if the proposal cannot be found.
    """
    chord_review_path = production_dir / "chords" / "review.yml"
    if not chord_review_path.exists():
        return {}
    with open(chord_review_path) as f:
        chord_review = yaml.safe_load(f) or {}

    thread = chord_review.get("thread", "")
    song_proposal_file = chord_review.get("song_proposal", "")
    if not thread or not song_proposal_file:
        return {}

    # Proposals live in <thread>/yml/<file>
    for candidate in [
        Path(thread) / "yml" / song_proposal_file,
        Path(thread) / song_proposal_file,
    ]:
        if candidate.exists():
            from app.generators.midi.production.production_plan import (
                load_song_proposal,
            )

            meta = load_song_proposal(candidate)
            meta["sounds_like"] = []  # not stored in proposal; defaults to empty
            meta["singer"] = str(chord_review.get("singer", ""))
            return meta

    return {}


# ---------------------------------------------------------------------------
# Vocal section reading from arrangement
# ---------------------------------------------------------------------------


def read_vocal_sections_from_arrangement(
    arrangement_path: Path,
    melody_dir: Path,
    bpm: int,
    time_sig_str: str,
) -> list[dict]:
    """Extract vocal sections from arrangement.txt.

    Track 4 (MELODY_CHANNEL) clips are vocal by definition — no vocals flag needed.
    Returns one entry per unique clip label, in first-seen order.

    Each entry: {approved_label, name, bars, repeat, total_notes, contour}
    """
    clips = parse_arrangement(arrangement_path)

    parts = str(time_sig_str).split("/")
    num, den = int(parts[0]), int(parts[1])
    beats_per_bar = num * (4.0 / den)
    secs_per_bar = beats_per_bar * (60.0 / bpm)

    # Load melody review for contour info
    melody_review_path = melody_dir / "review.yml"
    contour_by_label: dict[str, str] = {}
    if melody_review_path.exists():
        with open(melody_review_path) as f:
            melody_review = yaml.safe_load(f) or {}
        for cand in melody_review.get("candidates", []):
            label = cand.get("label")
            status = str(cand.get("status", "")).lower()
            if label and status in ("approved", "accepted"):
                contour_by_label[label] = cand.get("contour", "stepwise")

    # Collect track 4 clips, accumulate per unique label
    melody_clips = [c for c in clips if c["channel"] == MELODY_CHANNEL]
    seen: dict[str, dict] = {}
    order: list[str] = []

    for clip in melody_clips:
        label = clip["clip_name"]
        if label not in seen:
            seen[label] = {"count": 0, "total_duration": 0.0}
            order.append(label)
        seen[label]["count"] += 1
        seen[label]["total_duration"] += clip["duration_secs"]

    approved_dir = melody_dir / "approved"
    sections = []
    for label in order:
        data = seen[label]
        repeat = data["count"]
        loop_duration = data["total_duration"] / repeat
        bars = max(round(loop_duration / secs_per_bar), 1)

        midi_path = approved_dir / f"{label}.mid"
        per_loop_notes = _count_notes(midi_path) if midi_path.exists() else 0
        total_notes = per_loop_notes * repeat

        sections.append(
            {
                "approved_label": label,
                "name": label,
                "bars": bars,
                "repeat": repeat,
                "total_notes": total_notes,
                "contour": contour_by_label.get(label, "stepwise"),
            }
        )

    return sections


# ---------------------------------------------------------------------------
# Syllable fitting
# ---------------------------------------------------------------------------

_VERDICT_ORDER = ["spacious", "paste-ready", "tight but workable", "splits needed"]


def _fitting_verdict(ratio: float) -> str:
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
    """
    parsed = _parse_sections(candidate_text)
    result: dict = {}
    worst_idx = 0

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


def _build_prompt(
    meta: dict,
    vocal_sections: list[dict],
    syllable_targets: dict,
    artist_context: str = "",
) -> str:
    """Build the Claude prompt for lyric generation.

    meta keys used: title, color, bpm, time_sig, key, concept
    vocal_sections entries use 'name' as the [header] label (loop label).
    """
    color = meta.get("color", "")
    target = get_chromatic_target(color)
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
        f'You are writing lyrics for a song titled "{meta.get("title", "")}".',
        "",
        "SONG METADATA:",
        f"  Color: {color}",
        f"  BPM: {meta.get('bpm', '')}",
        f"  Time signature: {meta.get('time_sig', '')}",
        f"  Key: {meta.get('key', '')}",
        f"  Concept: {meta.get('concept', '')}",
        "",
        "CHROMATIC TARGET (the emotional/conceptual space to express):",
        f"  Temporal mode: {dominant_temporal}  "
        "(past=memory/history, present=immediacy, future=anticipation)",
        f"  Spatial mode: {dominant_spatial}  "
        "(thing=object/artifact, place=location/environment, person=human/being)",
        f"  Ontological mode: {dominant_ontological}  "
        "(imagined=fictional/possible, forgotten=lost/erased, known=certain/present)",
        "",
        f"Write lyrics that express the {color} chromatic concept: "
        f"{dominant_temporal}, {dominant_spatial}, {dominant_ontological}.",
        "",
        "SECTIONS TO WRITE:",
        "(Headers are melody loop labels — each maps to one MIDI clip.)",
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
                f"    Bars per loop: {sec['bars']}  ×  {sec['repeat']} occurrence(s)",
                f"    Melody contour: {sec['contour']}",
                f"    Target syllables: {lo}–{hi}  (≈{notes_per_bar:.1f} notes/bar)",
            ]
        )

    if artist_context:
        lines.extend(["", artist_context])

    lines.extend(
        [
            "",
            "OUTPUT FORMAT:",
            "  Use [loop_label] headers exactly as listed above.",
            "  Write one block per unique loop label.",
            "  Output only the lyrics — no commentary, no explanations.",
            "  Lines starting with # are ignored (you may use them for stage directions).",
            "",
            "Example:",
            "  [melody_verse_alternate]",
            "  First line of verse",
            "  Second line of verse",
            "",
            "  [melody_bridge]",
            "  First line of bridge",
            "  Second line of bridge",
            "",
            "Now write the complete lyrics:",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


def _call_api(client, prompt: str, model: str) -> str:
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


def _load_or_init_review(melody_dir: Path, meta: dict, model: str, seed: int) -> dict:
    """Load existing lyrics_review.yml or create a fresh header dict."""
    review_path = melody_dir / LYRICS_REVIEW_FILENAME
    if review_path.exists():
        with open(review_path) as f:
            return yaml.safe_load(f) or {}

    return {
        "production_dir": str(melody_dir.parent),
        "pipeline": "lyric-generation",
        "bpm": meta.get("bpm"),
        "time_sig": meta.get("time_sig"),
        "color": meta.get("color"),
        "generated": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "model": model,
        "scoring_weights": {"chromatic": 1.0},
        "candidates": [],
    }


# ---------------------------------------------------------------------------
# Candidate ID generation
# ---------------------------------------------------------------------------


def _next_candidate_id(review: dict) -> str:
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
    """Scan melody/candidates/*.txt for untracked files and add stub entries."""
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

    Reads vocal sections from arrangement.txt (track 4 = vocal).
    Reads song metadata from the song proposal YAML.
    No production_plan.yml required.

    Returns:
        The lyrics_review.yml dict after writing.
    """
    prod_path = Path(production_dir)
    if not prod_path.exists():
        print(f"ERROR: Production directory not found: {prod_path}")
        sys.exit(1)

    melody_dir = prod_path / "melody"
    arrangement_path = prod_path / "arrangement.txt"

    print("=" * 60)
    print("LYRIC GENERATION PIPELINE")
    print("=" * 60)

    # --- 1. Check arrangement exists ---
    if not arrangement_path.exists():
        print(
            "ERROR: arrangement.txt not found — export from Logic before generating lyrics"
        )
        sys.exit(1)

    # --- 2. Load song metadata from proposal ---
    meta = _find_and_load_proposal(prod_path)
    if not meta:
        # Fall back to chord review for minimal metadata
        chord_review_path = prod_path / "chords" / "review.yml"
        if chord_review_path.exists():
            with open(chord_review_path) as f:
                cr = yaml.safe_load(f) or {}
            meta = {
                "title": "",
                "bpm": int(cr.get("bpm", 120)),
                "time_sig": str(cr.get("time_sig", "4/4")),
                "key": str(cr.get("key", "")),
                "color": str(cr.get("color", "")),
                "concept": "",
                "sounds_like": [],
                "genres": [],
                "mood": [],
                "singer": str(cr.get("singer", "")),
            }
        else:
            print(
                "ERROR: Could not load song metadata (no proposal or chord review found)"
            )
            sys.exit(1)

    print(f"Song:  {meta.get('title', '(untitled)')}")
    print(f"Color: {meta.get('color', '')}")
    print(
        f"BPM:   {meta.get('bpm')}  Time: {meta.get('time_sig')}  Key: {meta.get('key', '')}"
    )

    # --- 3. Read vocal sections from arrangement ---
    vocal_sections = read_vocal_sections_from_arrangement(
        arrangement_path, melody_dir, meta["bpm"], meta["time_sig"]
    )
    if not vocal_sections:
        print("ERROR: No melody clips found on track 4 in arrangement.txt")
        print(
            "Export the arrangement from Logic after placing melody clips on track 4."
        )
        sys.exit(1)

    print(f"\nVocal sections ({len(vocal_sections)}) from arrangement:")
    for sec in vocal_sections:
        print(
            f"  {sec['name']}: {sec['bars']}b × {sec['repeat']}"
            f" = {sec['total_notes']} notes"
        )

    # --- 4. Syllable targets ---
    syllable_targets = {
        sec["name"]: (
            math.floor(sec["total_notes"] * 0.75),
            math.floor(sec["total_notes"] * 1.05),
        )
        for sec in vocal_sections
    }

    # --- 5. Build prompt ---
    artist_context = load_artist_context(meta.get("sounds_like") or [])
    prompt = _build_prompt(meta, vocal_sections, syllable_targets, artist_context)

    # --- 6. Generate candidates ---
    from anthropic import Anthropic

    client = Anthropic()
    print(f"\nGenerating {num_candidates} lyric candidate(s) via {model}...")
    texts = []
    for i in range(num_candidates):
        print(f"  Candidate {i + 1}/{num_candidates}...")
        text = _call_api(client, prompt, model)
        texts.append(text)

    # --- 7. Score with Refractor (text-only) ---
    scorer_results_map: dict[int, Optional[dict]] = {}
    target = get_chromatic_target(meta.get("color", ""))

    if not skip_scoring:
        print("\nLoading Refractor...")
        try:
            from training.refractor import Refractor

            scorer = Refractor(onnx_path=onnx_path) if onnx_path else Refractor()
            concept_text = (
                meta.get("concept") or f"{meta.get('color', '')} chromatic concept"
            )
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
            print(f"  Warning: Refractor unavailable ({e}), skipping scoring")
    else:
        print("\nSkipping Refractor (--skip-scoring)")

    # --- 8. Compute fitting + chromatic match ---
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

    scored_entries.sort(key=lambda e: e["chromatic_match"], reverse=True)

    # --- 9. Write candidate .txt files ---
    candidates_dir = melody_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    review = _load_or_init_review(melody_dir, meta, model, seed)

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

    # --- 10. Save review YAML ---
    review_path = melody_dir / LYRICS_REVIEW_FILENAME
    with open(review_path, "w") as f:
        yaml.dump(
            review, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    # --- 11. Summary ---
    print(f"\n{'=' * 60}")
    print("LYRIC GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Candidates: {len(new_entries)}")
    print(f"Review:     {review_path}")
    print()
    print(f"{'Rank':<5} {'ID':<12} {'Match':<8} {'Overall Fit'}")
    print("-" * 40)
    for entry in new_entries:
        chromatic = entry.get("chromatic") or {}
        match = chromatic.get("match", None)
        match_str = f"{match:.3f}" if match is not None else "n/a  "
        overall = entry["fitting"].get("overall", "?")
        print(f"  #{entry['rank']:<3} {entry['id']:<12} {match_str}    {overall}")

    print(f"\nNext: Edit {review_path} to approve a candidate")
    print(
        f"Then: python -m app.generators.midi.production.promote_part --review {review_path}"
    )

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
        help="Song production directory (must contain arrangement.txt)",
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
        help="Path to refractor.onnx (default: training/data/refractor.onnx)",
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip Refractor (useful when torch/DeBERTa unavailable locally)",
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
