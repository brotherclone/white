#!/usr/bin/env python3
"""
Promote approved part candidates from review.yml to the approved/ directory.

Reads a review.yml file, finds candidates with status=approved, and copies
their MIDI files to the approved/ directory with label-based filenames.

Usage:
    python -m app.generators.midi.promote_part \
        --review shrinkwrapped/.../production/.../chords/review.yml
"""

import argparse
import shutil
from pathlib import Path

import yaml


def promote_part(review_path: str):
    """Read review.yml and promote approved candidates."""
    review_file = Path(review_path)
    if not review_file.exists():
        print(f"ERROR: Review file not found: {review_file}")
        return

    with open(review_file) as f:
        review = yaml.safe_load(f)

    part_dir = review_file.parent
    candidates_dir = part_dir / "candidates"
    approved_dir = part_dir / "approved"
    approved_dir.mkdir(parents=True, exist_ok=True)

    candidates = review.get("candidates", [])
    approved = [
        c
        for c in candidates
        if str(c.get("status", "")).lower() in ("approved", "accepted")
        and not str(c.get("midi_file", "")).endswith("_scratch.mid")
    ]

    if not approved:
        print("No approved candidates found in review.yml")
        print(
            "Edit the review file and set status: approved on candidates you want to promote"
        )
        return

    # Enforce one approved per label
    label_to_candidates: dict[str, list] = {}
    for candidate in approved:
        label = (
            (candidate.get("label") or "unlabeled")
            .replace("-", "_")
            .replace(" ", "_")
            .lower()
        )
        label_to_candidates.setdefault(label, []).append(candidate)

    conflicts = {
        label: cands for label, cands in label_to_candidates.items() if len(cands) > 1
    }
    if conflicts:
        print("ERROR: Multiple approved candidates share the same label.")
        print("Resolve by rejecting all but one per label, then re-run promotion.\n")
        for label, cands in conflicts.items():
            ids = ", ".join(c.get("id", "?") for c in cands)
            print(f"  label '{label}': {ids}")
        return

    promoted = []
    for label_clean, cands in label_to_candidates.items():
        candidate = cands[0]
        label = candidate.get("label") or "unlabeled"
        midi_filename = candidate.get("midi_file", "")
        source = candidates_dir / Path(midi_filename).name

        if not source.exists():
            source = part_dir / midi_filename
        if not source.exists():
            print(f"  WARNING: MIDI file not found: {source}")
            continue

        dest_name = f"{label_clean}.mid"
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
        description="Promote approved part candidates from review.yml"
    )
    parser.add_argument("--review", required=True, help="Path to review.yml")
    args = parser.parse_args()
    promote_part(args.review)


if __name__ == "__main__":
    main()
