#!/usr/bin/env python3
"""
Lyric Feedback Dataset Export

Walks production directories, collects (draft, edited) lyric pairs with
per-section fitting metrics and chromatic match scores, and writes a JSONL
file suitable for few-shot prompt engineering or future LoRA fine-tuning.

Usage:
    # Export all songs in a thread
    python -m app.generators.midi.lyric_feedback_export \
        --thread shrinkwrapped/white-the-breathing-machine-learns-to-sing \
        --output lyric_feedback.jsonl

    # Single song
    python -m app.generators.midi.lyric_feedback_export \
        --production-dir shrinkwrapped/.../production/yellow__... \
        --output lyric_feedback.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from app.generators.midi.lyric_pipeline import _compute_fitting, _read_vocal_sections
from app.generators.midi.production_plan import load_plan

LYRICS_REVIEW_FILENAME = "lyrics_review.yml"
EVALUATION_FILENAME = "song_evaluation.yml"
DEFAULT_OUTPUT = "lyric_feedback.jsonl"


# ---------------------------------------------------------------------------
# Record collection
# ---------------------------------------------------------------------------


def collect_song_record(production_dir: Path) -> dict | None:
    """Collect a feedback record for one production directory.

    Returns a dict (always, even for partial data) or None if the directory
    has no lyrics.txt at all.
    """
    production_dir = Path(production_dir)
    melody_dir = production_dir / "melody"
    lyrics_path = melody_dir / "lyrics.txt"
    draft_path = melody_dir / "lyrics_draft.txt"

    if not lyrics_path.exists():
        return None

    plan = load_plan(production_dir)
    song_slug = production_dir.name
    color = plan.color if plan else ""
    concept = plan.concept if plan else ""
    bpm = plan.bpm if plan else 0
    time_sig = plan.time_sig if plan else ""
    key = getattr(plan, "key", "") if plan else ""

    # Vocal sections + note counts (from lyric_pipeline helper)
    vocal_sections: list[dict] = []
    if plan:
        try:
            vocal_sections = _read_vocal_sections(plan, melody_dir)
        except Exception:
            pass

    # Singer — from lyrics_review.yml header (most reliable) or plan
    singer = ""
    review_data: dict = {}
    review_path = melody_dir / LYRICS_REVIEW_FILENAME
    if review_path.exists():
        with open(review_path) as f:
            review_data = yaml.safe_load(f) or {}
        singer = review_data.get("singer", "")

    # draft_chromatic_match — from the approved candidate in lyrics_review.yml
    draft_chromatic_match = None
    for candidate in review_data.get("candidates", []):
        status = str(candidate.get("status", "")).lower()
        if status in ("approved", "accepted"):
            ch = candidate.get("chromatic") or {}
            v = ch.get("match")
            if v is not None:
                draft_chromatic_match = float(v)
            break

    # edited_chromatic_match — from song_evaluation.yml if present
    edited_chromatic_match = None
    eval_path = production_dir / EVALUATION_FILENAME
    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = yaml.safe_load(f) or {}
        v = eval_data.get("lyrics_edited_chromatic_match")
        if v is not None:
            edited_chromatic_match = float(v)

    # Texts
    edited_text = lyrics_path.read_text(encoding="utf-8")

    if not draft_path.exists():
        print(f"  WARNING: no lyrics_draft.txt for {song_slug}")
        return {
            "song_slug": song_slug,
            "color": color,
            "concept": concept,
            "bpm": bpm,
            "time_sig": time_sig,
            "key": key,
            "singer": singer,
            "vocal_sections": [
                {k: v for k, v in s.items() if k != "approved_label"}
                for s in vocal_sections
            ],
            "draft_text": None,
            "edited_text": edited_text,
            "edited": None,
            "draft_chromatic_match": draft_chromatic_match,
            "edited_chromatic_match": edited_chromatic_match,
            "draft_fitting": None,
            "edited_fitting": _compute_fitting(edited_text, vocal_sections),
        }

    draft_text = draft_path.read_text(encoding="utf-8")
    is_edited = edited_text.strip() != draft_text.strip()

    if not is_edited:
        print(f"  note: no edits detected for {song_slug}")

    # Per-section fitting for both texts
    draft_fitting = _compute_fitting(draft_text, vocal_sections)
    edited_fitting = _compute_fitting(edited_text, vocal_sections)

    return {
        "song_slug": song_slug,
        "color": color,
        "concept": concept,
        "bpm": bpm,
        "time_sig": time_sig,
        "key": key,
        "singer": singer,
        "vocal_sections": [
            {k: v for k, v in s.items() if k != "approved_label"}
            for s in vocal_sections
        ],
        "draft_text": draft_text,
        "edited_text": edited_text,
        "edited": is_edited,
        "draft_chromatic_match": draft_chromatic_match,
        "edited_chromatic_match": edited_chromatic_match,
        "draft_fitting": draft_fitting,
        "edited_fitting": edited_fitting,
    }


# ---------------------------------------------------------------------------
# Thread export
# ---------------------------------------------------------------------------


def export_feedback(
    production_dirs: list[Path],
    output_path: Path,
) -> dict:
    """Collect records from production_dirs and write JSONL.

    Returns summary counts.
    """
    records = []
    n_null_draft = 0
    n_confirmed_edits = 0
    n_no_edits = 0

    for prod_dir in production_dirs:
        prod_dir = Path(prod_dir)
        record = collect_song_record(prod_dir)
        if record is None:
            continue
        records.append(record)
        if record["edited"] is None:
            n_null_draft += 1
        elif record["edited"]:
            n_confirmed_edits += 1
        else:
            n_no_edits += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "total": len(records),
        "confirmed_edits": n_confirmed_edits,
        "no_edits": n_no_edits,
        "null_drafts": n_null_draft,
    }

    print(f"\nExported {len(records)} songs to {output_path}")
    print(
        f"  {n_confirmed_edits} with confirmed edits, "
        f"{n_no_edits} unchanged, "
        f"{n_null_draft} with null drafts"
    )

    if n_confirmed_edits < 20:
        print(
            f"\n  {n_confirmed_edits} pairs collected — "
            "suggest 20+ for reliable few-shot injection, "
            "100+ for LoRA fine-tuning"
        )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Export lyric feedback dataset (draft/edited pairs) to JSONL"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--thread",
        help="Thread directory (walks production/ subdirectories)",
    )
    source.add_argument(
        "--production-dir",
        help="Single production directory",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL file (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if args.thread:
        thread_dir = Path(args.thread)
        if not thread_dir.exists():
            print(f"ERROR: Thread directory not found: {thread_dir}")
            sys.exit(1)
        production_dirs = sorted((thread_dir / "production").glob("*/"))
        if not production_dirs:
            print(f"No production directories found under {thread_dir}/production/")
            sys.exit(1)
    else:
        prod_path = Path(args.production_dir)
        if not prod_path.exists():
            print(f"ERROR: Production directory not found: {prod_path}")
            sys.exit(1)
        production_dirs = [prod_path]

    export_feedback(production_dirs, Path(args.output))


if __name__ == "__main__":
    main()
