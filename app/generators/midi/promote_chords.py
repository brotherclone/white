#!/usr/bin/env python3
"""
Promote approved chord candidates from review.yml to the approved/ directory.

Reads a review.yml file, finds candidates with status=approved, and copies
their MIDI files to the approved/ directory with label-based filenames.

Usage:
    python -m app.generators.midi.promote_chords \
        --review shrinkwrapped/.../production/.../chords/review.yml
"""

import argparse
import shutil
from pathlib import Path

import yaml


def promote_chords(review_path: str):
    """Read review.yml and promote approved candidates."""
    review_file = Path(review_path)
    if not review_file.exists():
        print(f"ERROR: Review file not found: {review_file}")
        return

    with open(review_file) as f:
        review = yaml.safe_load(f)

    chords_dir = review_file.parent
    candidates_dir = chords_dir / "candidates"
    approved_dir = chords_dir / "approved"
    approved_dir.mkdir(parents=True, exist_ok=True)

    candidates = review.get("candidates", [])
    approved = [
        c
        for c in candidates
        if str(c.get("status", "")).lower() in ("approved", "accepted")
    ]

    if not approved:
        print("No approved candidates found in review.yml")
        print(
            "Edit the review file and set status: approved on candidates you want to promote"
        )
        return

    # Group by label for numbering
    label_counts: dict[str, int] = {}
    promoted = []

    for candidate in approved:
        label = candidate.get("label") or "unlabeled"
        label_clean = label.replace("-", "_").replace(" ", "_").lower()
        midi_filename = candidate.get("midi_file", "")
        source = candidates_dir / Path(midi_filename).name

        if not source.exists():
            source = chords_dir / midi_filename
        if not source.exists():
            print(f"  WARNING: MIDI file not found: {source}")
            continue

        # Number duplicates: verse.mid, verse_2.mid, verse_3.mid
        count = label_counts.get(label_clean, 0) + 1
        label_counts[label_clean] = count

        if count == 1:
            dest_name = f"{label_clean}.mid"
        else:
            dest_name = f"{label_clean}_{count}.mid"

        dest = approved_dir / dest_name
        shutil.copy2(source, dest)
        promoted.append(
            {
                "id": candidate.get("id"),
                "label": label,
                "source": str(source.name),
                "dest": str(dest_name),
                "rank": candidate.get("rank"),
            }
        )

    # Summary
    print(f"\nPromoted {len(promoted)} candidates to {approved_dir}/")
    print("-" * 50)
    for p in promoted:
        print(f"  #{p['rank']} [{p['id']}] {p['label']:20s} → {p['dest']}")

    print(f"\nApproved directory: {approved_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Promote approved chord candidates from review.yml"
    )
    parser.add_argument("--review", required=True, help="Path to review.yml")
    args = parser.parse_args()
    promote_chords(args.review)


if __name__ == "__main__":
    main()
