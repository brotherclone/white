#!/usr/bin/env python3
"""
Simple CLI tool for human-in-the-loop annotation of ambiguous segments.

Usage:
    python tools/annotation_cli.py data/training_data.parquet --output refined_annotations.json
    python tools/annotation_cli.py data/training_data.parquet --filter-ambiguous --threshold 0.4
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


TEMPORAL_MODES = ["past", "present", "future"]
SPATIAL_MODES = ["thing", "place", "person"]
ONTOLOGICAL_MODES = ["imagined", "forgotten", "known"]

ALBUM_MAPPING = {
    ("past", "thing", "imagined"): "Orange",
    ("past", "thing", "forgotten"): "Red",
    ("past", "thing", "known"): "Violet",
    ("past", "place", "imagined"): "Orange",
    ("past", "place", "forgotten"): "Red",
    ("past", "place", "known"): "Violet",
    ("past", "person", "imagined"): "Orange",
    ("past", "person", "forgotten"): "Red",
    ("past", "person", "known"): "Violet",
    ("present", "thing", "imagined"): "Yellow",
    ("present", "thing", "forgotten"): "Indigo",
    ("present", "thing", "known"): "Green",
    ("present", "place", "imagined"): "Yellow",
    ("present", "place", "forgotten"): "Indigo",
    ("present", "place", "known"): "Green",
    ("present", "person", "imagined"): "Yellow",
    ("present", "person", "forgotten"): "Indigo",
    ("present", "person", "known"): "Green",
    ("future", "thing", "imagined"): "Blue",
    ("future", "thing", "forgotten"): "Blue",
    ("future", "thing", "known"): "Blue",
    ("future", "place", "imagined"): "Blue",
    ("future", "place", "forgotten"): "Blue",
    ("future", "place", "known"): "Blue",
    ("future", "person", "imagined"): "Blue",
    ("future", "person", "forgotten"): "Blue",
    ("future", "person", "known"): "Blue",
}


def get_album(temporal: str, spatial: str, ontological: str) -> str:
    """Get album from mode combination."""
    key = (temporal.lower(), spatial.lower(), ontological.lower())
    return ALBUM_MAPPING.get(key, "Black")


def to_soft_target(label: str, modes: List[str], smoothing: float = 0.0) -> np.ndarray:
    """Convert label to soft target."""
    if (
        pd.isna(label)
        or label is None
        or str(label).lower() in ("none", "null", "", "black")
    ):
        return np.array([1 / 3, 1 / 3, 1 / 3])

    label = str(label).lower().strip()
    try:
        idx = modes.index(label)
    except ValueError:
        return np.array([1 / 3, 1 / 3, 1 / 3])

    target = np.zeros(len(modes))
    target[idx] = 1.0

    if smoothing > 0:
        target = (1 - smoothing) * target + smoothing * (1 / len(modes))

    return target


def display_sample(row: pd.Series, idx: int, text_col: str = "concept"):
    """Display a sample for annotation."""
    print("\n" + "=" * 70)
    print(f"Sample {idx}")
    print("=" * 70)

    # Show text
    text = row.get(text_col, row.get("lyric_text", ""))
    if text and not pd.isna(text):
        print(f"\nText: {text[:500]}{'...' if len(str(text)) > 500 else ''}")

    # Show current labels
    t_col = next((c for c in row.index if "temporal" in c.lower()), None)
    s_col = next(
        (c for c in row.index if "objectional" in c.lower() or "spatial" in c.lower()),
        None,
    )
    o_col = next((c for c in row.index if "ontological" in c.lower()), None)

    t_val = row.get(t_col) if t_col else None
    s_val = row.get(s_col) if s_col else None
    o_val = row.get(o_col) if o_col else None

    print("\nCurrent labels:")
    print(f"  Temporal:     {t_val or 'None'}")
    print(f"  Spatial:      {s_val or 'None'}")
    print(f"  Ontological:  {o_val or 'None'}")

    if t_val and s_val and o_val:
        album = get_album(str(t_val), str(s_val), str(o_val))
        print(f"  Album:        {album}")

    # Show soft targets
    t_target = to_soft_target(t_val, TEMPORAL_MODES, 0.1)
    s_target = to_soft_target(s_val, SPATIAL_MODES, 0.1)
    o_target = to_soft_target(o_val, ONTOLOGICAL_MODES, 0.1)

    print("\nSoft targets (with smoothing=0.1):")
    print(f"  Temporal:     {t_target.round(3)} [{'/'.join(TEMPORAL_MODES)}]")
    print(f"  Spatial:      {s_target.round(3)} [{'/'.join(SPATIAL_MODES)}]")
    print(f"  Ontological:  {o_target.round(3)} [{'/'.join(ONTOLOGICAL_MODES)}]")

    return t_val, s_val, o_val


def prompt_annotation() -> Dict:
    """Prompt user for annotation."""
    print("\n" + "-" * 40)
    print("Options:")
    print("  [Enter]     - Keep current labels")
    print("  [t/s/o]     - Modify temporal/spatial/ontological")
    print("  [c]         - Set custom soft targets")
    print("  [b]         - Mark as Black Album (uniform)")
    print("  [f]         - Flag for further review")
    print("  [s]         - Skip")
    print("  [q]         - Quit")
    print("-" * 40)

    choice = input("Action: ").strip().lower()

    if choice == "" or choice == "k":
        return {"action": "keep"}
    elif choice == "s":
        return {"action": "skip"}
    elif choice == "q":
        return {"action": "quit"}
    elif choice == "f":
        note = input("Review note: ").strip()
        return {"action": "flag", "note": note}
    elif choice == "b":
        return {"action": "black"}
    elif choice == "t":
        return prompt_mode_change("temporal", TEMPORAL_MODES)
    elif choice == "s" or choice == "sp":
        return prompt_mode_change("spatial", SPATIAL_MODES)
    elif choice == "o":
        return prompt_mode_change("ontological", ONTOLOGICAL_MODES)
    elif choice == "c":
        return prompt_custom_targets()
    else:
        print("Invalid option, keeping current labels")
        return {"action": "keep"}


def prompt_mode_change(dimension: str, modes: List[str]) -> Dict:
    """Prompt for mode change."""
    print(f"\n{dimension.capitalize()} modes:")
    for i, mode in enumerate(modes):
        print(f"  [{i+1}] {mode.capitalize()}")
    print("  [0] None (Black Album)")

    try:
        choice = int(input("Select: ").strip())
        if choice == 0:
            return {"action": "modify", "dimension": dimension, "value": None}
        elif 1 <= choice <= len(modes):
            return {
                "action": "modify",
                "dimension": dimension,
                "value": modes[choice - 1],
            }
    except ValueError:
        pass

    print("Invalid selection")
    return {"action": "keep"}


def prompt_custom_targets() -> Dict:
    """Prompt for custom soft target distribution."""
    print("\nEnter custom soft targets (3 values that sum to 1.0):")

    targets = {}
    for dim, modes in [
        ("temporal", TEMPORAL_MODES),
        ("spatial", SPATIAL_MODES),
        ("ontological", ONTOLOGICAL_MODES),
    ]:
        print(f"\n{dim.capitalize()} [{'/'.join(modes)}]:")
        try:
            values = input("  Values (comma-separated): ").strip()
            if values:
                parts = [float(x.strip()) for x in values.split(",")]
                if len(parts) == 3 and abs(sum(parts) - 1.0) < 0.01:
                    targets[dim] = parts
                else:
                    print(
                        f"  Invalid (must be 3 values summing to 1.0), skipping {dim}"
                    )
        except ValueError:
            print(f"  Parse error, skipping {dim}")

    if targets:
        return {"action": "custom", "targets": targets}
    return {"action": "keep"}


def find_ambiguous_samples(
    df: pd.DataFrame,
    threshold: float = 0.4,
    max_samples: int = 100,
) -> List[int]:
    """Find samples that might be ambiguous/hybrid."""
    # Look for samples where labels might be unclear
    t_col = next((c for c in df.columns if "temporal" in c.lower()), None)
    s_col = next(
        (c for c in df.columns if "objectional" in c.lower() or "spatial" in c.lower()),
        None,
    )
    o_col = next((c for c in df.columns if "ontological" in c.lower()), None)

    ambiguous_indices = []

    for idx in range(len(df)):
        # Check for None values (potential Black Album)
        row = df.iloc[idx]
        t_val = row.get(t_col) if t_col else None
        s_val = row.get(s_col) if s_col else None
        o_val = row.get(o_col) if o_col else None

        # Null check
        if pd.isna(t_val) or pd.isna(s_val) or pd.isna(o_val):
            ambiguous_indices.append(idx)
            continue

        # Could add more sophisticated ambiguity detection here
        # For now, just flag nulls

        if len(ambiguous_indices) >= max_samples:
            break

    return ambiguous_indices


def run_annotation_session(
    df: pd.DataFrame,
    output_path: str,
    indices: Optional[List[int]] = None,
    text_col: str = "concept",
):
    """Run interactive annotation session."""
    if indices is None:
        indices = list(range(len(df)))

    annotations = []

    print("\n" + "=" * 70)
    print("HUMAN-IN-THE-LOOP ANNOTATION SESSION")
    print(f"Samples to review: {len(indices)}")
    print("=" * 70)

    for i, idx in enumerate(indices):
        row = df.iloc[idx]

        print(f"\n[{i+1}/{len(indices)}]", end="")
        current_t, current_s, current_o = display_sample(row, idx, text_col)

        result = prompt_annotation()

        if result["action"] == "quit":
            print("\nSaving and exiting...")
            break
        elif result["action"] == "skip":
            continue
        elif result["action"] == "keep":
            annotations.append(
                {
                    "index": int(idx),
                    "action": "keep",
                    "temporal": current_t,
                    "spatial": current_s,
                    "ontological": current_o,
                }
            )
        elif result["action"] == "black":
            annotations.append(
                {
                    "index": int(idx),
                    "action": "black",
                    "temporal": None,
                    "spatial": None,
                    "ontological": None,
                    "soft_targets": {
                        "temporal": [1 / 3, 1 / 3, 1 / 3],
                        "spatial": [1 / 3, 1 / 3, 1 / 3],
                        "ontological": [1 / 3, 1 / 3, 1 / 3],
                    },
                }
            )
        elif result["action"] == "flag":
            annotations.append(
                {
                    "index": int(idx),
                    "action": "flag",
                    "note": result.get("note", ""),
                    "temporal": current_t,
                    "spatial": current_s,
                    "ontological": current_o,
                }
            )
        elif result["action"] == "modify":
            dim = result["dimension"]
            val = result["value"]
            annotation = {
                "index": int(idx),
                "action": "modify",
                "temporal": current_t,
                "spatial": current_s,
                "ontological": current_o,
            }
            annotation[dim] = val
            annotations.append(annotation)
        elif result["action"] == "custom":
            annotation = {
                "index": int(idx),
                "action": "custom",
                "temporal": current_t,
                "spatial": current_s,
                "ontological": current_o,
                "soft_targets": result.get("targets", {}),
            }
            annotations.append(annotation)

    # Save annotations
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(
            {
                "annotations": annotations,
                "total_reviewed": len(annotations),
                "source_rows": len(df),
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\nSaved {len(annotations)} annotations to {output_path}")
    return annotations


def main():
    parser = argparse.ArgumentParser(description="Human-in-the-loop annotation CLI")
    parser.add_argument("parquet_path", help="Path to parquet file")
    parser.add_argument(
        "--output", "-o", default="annotations.json", help="Output JSON path"
    )
    parser.add_argument(
        "--text-column", default="concept", help="Column with text content"
    )
    parser.add_argument(
        "--filter-ambiguous", action="store_true", help="Only show ambiguous samples"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.4, help="Ambiguity threshold"
    )
    parser.add_argument(
        "--max-samples", type=int, default=50, help="Max samples to review"
    )
    parser.add_argument("--start-index", type=int, default=0, help="Starting index")

    args = parser.parse_args()

    # Load data
    print(f"Loading {args.parquet_path}...")
    df = pd.read_parquet(args.parquet_path)
    print(f"Loaded {len(df)} rows")

    # Determine indices to review
    if args.filter_ambiguous:
        indices = find_ambiguous_samples(df, args.threshold, args.max_samples)
        print(f"Found {len(indices)} ambiguous samples")
    else:
        indices = list(
            range(args.start_index, min(args.start_index + args.max_samples, len(df)))
        )

    if not indices:
        print("No samples to review")
        return

    # Run session
    run_annotation_session(df, args.output, indices, args.text_column)


if __name__ == "__main__":
    main()
