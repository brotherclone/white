#!/usr/bin/env python3
"""
Migrate an existing production directory to song_context.yml.

Reads chords/review.yml to locate the original song proposal, then builds a
song_context.yml without re-running any MIDI generation phase. Existing files
are never modified.

Usage:
    python -m app.generators.midi.production.migrate_production_dir \
        --production-dir shrink_wrapped/.../production/red__my_song_v1

    # Preview without writing:
    python -m app.generators.midi.production.migrate_production_dir \
        --production-dir ... --dry-run
"""

from __future__ import annotations

import argparse
import sys
import yaml

from pathlib import Path

from app.generators.midi.production.init_production import (
    SONG_CONTEXT_FILENAME,
    _write_song_context,
    load_initial_proposal,
)
from app.generators.midi.production.production_plan import load_song_proposal


# Phase review filenames to check for completion
_PHASE_REVIEW_FILES = {
    "chords": "chords/review.yml",
    "drums": "drums/review.yml",
    "bass": "bass/review.yml",
    "melody": "melody/review.yml",
    "lyrics": "melody/lyrics_review.yml",
    "composition_proposal": "composition_proposal.yml",
}


def _detect_phase_status(production_dir: Path) -> dict[str, str]:
    """Check each phase review for approved candidates.

    Returns a dict mapping phase name to 'complete' or 'pending'.
    """
    statuses: dict[str, str] = {}
    for phase, rel_path in _PHASE_REVIEW_FILES.items():
        review_path = production_dir / rel_path
        if not review_path.exists():
            statuses[phase] = "pending"
            continue
        try:
            with open(review_path) as f:
                data = yaml.safe_load(f) or {}
            # composition_proposal.yml has no candidates list
            if phase == "composition_proposal":
                statuses[phase] = "complete"
                continue
            candidates = data.get("candidates") or []
            has_approved = any(c.get("status") == "approved" for c in candidates)
            statuses[phase] = "complete" if has_approved else "pending"
        except Exception:
            statuses[phase] = "pending"
    return statuses


def migrate_production_dir(
    production_dir: Path,
    dry_run: bool = False,
) -> dict | None:
    """Build song_context.yml for an existing production directory.

    Reads chords/review.yml to locate the original song proposal, loads
    concept/genres/mood from it, and picks up sounds_like from
    initial_proposal.yml if present.

    Returns the context dict that was (or would be) written.
    Returns None if migration cannot proceed (missing chord review).
    """
    production_dir = Path(production_dir)
    ctx_path = production_dir / SONG_CONTEXT_FILENAME

    if ctx_path.exists() and not dry_run:
        print(f"  song_context.yml already exists at {ctx_path} — skipping")
        return None

    # --- Load chord review ---
    chord_review_path = production_dir / "chords" / "review.yml"
    if not chord_review_path.exists():
        print(f"  ERROR: chords/review.yml not found at {chord_review_path}")
        print("  Cannot migrate without chord review. Run chord_pipeline first.")
        return None

    with open(chord_review_path) as f:
        chord_review = yaml.safe_load(f) or {}

    # --- Resolve song proposal ---
    thread = str(chord_review.get("thread", ""))
    song_proposal_name = str(chord_review.get("song_proposal", ""))
    proposal_path = None
    proposal_meta: dict = {}

    if thread and song_proposal_name:
        candidate = Path(thread) / "yml" / song_proposal_name
        if candidate.exists():
            proposal_path = candidate
            try:
                proposal_meta = load_song_proposal(proposal_path)
            except Exception as e:
                print(
                    f"  Warning: Could not load song proposal ({e}) — using chord review fallback"
                )

    # --- Build meta from chord review + proposal ---
    meta: dict = {
        "title": proposal_meta.get("title", ""),
        "song_slug": "",
        "song_proposal": song_proposal_name,
        "thread": thread,
        "color": str(chord_review.get("color", proposal_meta.get("color", ""))),
        "concept": proposal_meta.get("concept", ""),
        "key": str(chord_review.get("key", proposal_meta.get("key", ""))),
        "bpm": int(chord_review.get("bpm", proposal_meta.get("bpm") or 120)),
        "time_sig": str(
            chord_review.get("time_sig") or proposal_meta.get("time_sig") or "4/4"
        ),
        "singer": str(chord_review.get("singer", proposal_meta.get("singer", ""))),
        "genres": proposal_meta.get("genres") or [],
        "mood": proposal_meta.get("mood") or [],
    }

    # --- Derive song_slug from production dir name if not in proposal ---
    if not meta["song_slug"]:
        meta["song_slug"] = production_dir.name

    # --- sounds_like from initial_proposal.yml ---
    initial = load_initial_proposal(production_dir)
    sounds_like = initial.get("sounds_like") or []

    # --- Phase statuses ---
    meta["phases"] = _detect_phase_status(production_dir)

    if dry_run:
        print("  [dry-run] Would write song_context.yml with:")
        print(f"    title:      {meta['title']!r}")
        print(f"    color:      {meta['color']!r}")
        print(f"    concept:    {meta['concept'][:60]!r}")
        print(f"    key:        {meta['key']!r}")
        print(f"    bpm:        {meta['bpm']}")
        print(f"    singer:     {meta['singer']!r}")
        print(f"    sounds_like: {sounds_like}")
        print(f"    phases:     {meta['phases']}")
        return {**meta, "sounds_like": sounds_like}

    out_path = _write_song_context(production_dir, meta, sounds_like)
    print(f"  Written: {out_path}")
    return {**meta, "sounds_like": sounds_like}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate an existing production directory to song_context.yml."
    )
    parser.add_argument(
        "--production-dir",
        required=True,
        help="Production directory to migrate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without modifying any files",
    )
    args = parser.parse_args()

    prod_dir = Path(args.production_dir)
    if not prod_dir.exists():
        print(f"ERROR: Production directory not found: {prod_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Migrating: {prod_dir}")
    result = migrate_production_dir(prod_dir, dry_run=args.dry_run)
    if result is None and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
