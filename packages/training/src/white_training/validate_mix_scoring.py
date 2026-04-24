#!/usr/bin/env python3
"""
Validate chunked mix scoring accuracy against labeled staged_raw_material songs.

For each song directory that has a <id>.yml (with rainbow_color) and a
<id>_main.wav file, scores the mix with the Refractor, records the predicted
top-1 color, and compares to the ground-truth color label.

Writes a per-song YAML report to training/data/mix_scoring_validation.yml and
prints a per-color accuracy table.  Emits a prominent warning if overall
accuracy is below 70%.

Two-phase validation workflow:
    # Phase 1 (base model only):
    python training/validate_mix_scoring.py --no-cdm

    # Phase 2 (with CDM calibration head):
    python training/validate_mix_scoring.py

Usage:
    python training/validate_mix_scoring.py
    python training/validate_mix_scoring.py --artifacts-dir /path/to/staged_raw_material
    python training/validate_mix_scoring.py --no-cdm
    python training/validate_mix_scoring.py --chunk-size 30 --chunk-stride 5
    python training/validate_mix_scoring.py --output training/data/my_report.yml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml
from white_core.concepts.chromatic_targets import CHROMATIC_TARGETS

_COLOR_MAP = {
    "R": "Red",
    "O": "Orange",
    "Y": "Yellow",
    "G": "Green",
    "B": "Blue",
    "I": "Indigo",
    "V": "Violet",
    "Z": "White",
    "K": "Black",
}

_COLOR_ORDER = [
    "Red",
    "Orange",
    "Yellow",
    "Green",
    "Blue",
    "Indigo",
    "Violet",
    "White",
    "Black",
]


def _top1_color(score_result: dict) -> str:
    """Return the predicted color from a Refractor score result.

    When the CDM path was used, ``score_result`` carries ``predicted_color``
    directly (the CDM argmax) — use it to avoid the dot-product round-trip,
    which breaks for Indigo (soft-label ontological) and White/Black (identical
    uniform targets).

    Falls back to nearest-neighbor dot-product search for base-model results.
    """
    if "predicted_color" in score_result:
        return score_result["predicted_color"]

    t = [score_result["temporal"].get(m, 0.0) for m in ("past", "present", "future")]
    s = [score_result["spatial"].get(m, 0.0) for m in ("thing", "place", "person")]
    o = [
        score_result["ontological"].get(m, 0.0)
        for m in ("imagined", "forgotten", "known")
    ]

    best, best_score = "White", -1.0
    for color, targets in CHROMATIC_TARGETS.items():
        score = (
            float(np.dot(t, targets["temporal"]))
            + float(np.dot(s, targets["spatial"]))
            + float(np.dot(o, targets["ontological"]))
        ) / 3.0
        if score > best_score:
            best_score = score
            best = color
    return best


def validate(
    artifacts_dir: Path,
    output_path: Path,
    onnx_path: str | None = None,
    cdm_onnx_path: str | None = None,
    chunk_size_s: float = 30.0,
    chunk_stride_s: float = 5.0,
) -> None:
    """Score all labeled _main.wav files and write a per-song YAML report."""
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from white_analysis.refractor import Refractor

    from app.generators.midi.production.score_mix import (
        aggregate_chunk_scores,
        chunk_audio,
    )

    scorer = Refractor(onnx_path=onnx_path, cdm_onnx_path=cdm_onnx_path)
    using_cdm = scorer._cdm_session is not None

    print(
        f"Refractor CDM: {'enabled (' + scorer._cdm_onnx_path + ')' if using_cdm else 'disabled'}"
    )
    print(f"Chunk size: {chunk_size_s}s  stride: {chunk_stride_s}s\n")

    import librosa

    song_dirs = sorted(d for d in artifacts_dir.iterdir() if d.is_dir())
    print(f"Found {len(song_dirs)} song directories\n")

    per_song: list[dict] = []
    skipped = 0
    t0 = time.time()

    for i, song_dir in enumerate(song_dirs):
        song_id = song_dir.name
        yml_path = song_dir / f"{song_id}.yml"
        main_wav = song_dir / f"{song_id}_main.wav"

        if not yml_path.exists() or not main_wav.exists():
            skipped += 1
            continue

        with open(yml_path) as f:
            meta = yaml.safe_load(f)

        color_code = str(meta.get("rainbow_color", "")).strip().upper()
        color = _COLOR_MAP.get(color_code)
        if color is None:
            skipped += 1
            continue

        concept_text = meta.get("concept", "")
        concept_emb = (
            scorer.prepare_concept(concept_text)
            if concept_text
            else np.zeros(768, dtype=np.float32)
        )

        try:
            waveform, sr = librosa.load(str(main_wav), sr=None, mono=True)
            chunks = chunk_audio(
                waveform, int(sr), chunk_size_s=chunk_size_s, stride_s=chunk_stride_s
            )
            chunk_embs = [scorer.prepare_audio(c, sr=48000) for c in chunks]
            chunk_results = [
                scorer.score(audio_emb=emb, concept_emb=concept_emb)
                for emb in chunk_embs
            ]
            result = aggregate_chunk_scores(chunk_results)

            # When CDM was used, each chunk carries predicted_color directly.
            # Majority-vote across chunks to get the song-level prediction;
            # this bypasses the broken distribution round-trip in _top1_color
            # (Indigo soft-label and White/Black uniform-target tie).
            chunk_colors = [
                r["predicted_color"] for r in chunk_results if "predicted_color" in r
            ]
            if chunk_colors:
                from collections import Counter

                pred_color = Counter(chunk_colors).most_common(1)[0][0]
            else:
                pred_color = _top1_color(result)
            correct = pred_color == color

            per_song.append(
                {
                    "song_id": song_id,
                    "true_color": color,
                    "predicted_color": pred_color,
                    "correct": correct,
                    "confidence": round(float(result["confidence"]), 4),
                    "chunk_count": len(chunks),
                }
            )

            elapsed = time.time() - t0
            avg = elapsed / (i + 1 - skipped)
            remaining = avg * (len(song_dirs) - i - 1)
            mark = "+" if correct else "-"
            print(
                f"  [{mark}] {song_id:<40} true={color:<7} pred={pred_color:<7} "
                f"conf={result['confidence']:.3f}  chunks={len(chunks)}"
                f"  {elapsed:.0f}s / ~{remaining:.0f}s left"
            )

        except Exception as exc:
            print(f"  [ERR] {song_id}: {exc}", file=sys.stderr)
            skipped += 1

    # ------------------------------------------------------------------
    # Per-color accuracy table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Per-color accuracy:")
    by_color: dict[str, list[bool]] = {c: [] for c in _COLOR_ORDER}
    for entry in per_song:
        col = entry["true_color"]
        if col not in by_color:
            by_color[col] = []
        by_color[col].append(entry["correct"])

    total_correct = 0
    total_songs = 0
    for color in _COLOR_ORDER:
        results = by_color[color]
        if not results:
            continue
        n_correct = sum(results)
        n_total = len(results)
        acc = n_correct / n_total
        total_correct += n_correct
        total_songs += n_total
        print(f"  {color:<8} {n_correct:2d}/{n_total:2d}  {acc:.1%}")

    overall_acc = total_correct / max(1, total_songs)
    print(f"\n  OVERALL  {total_correct}/{total_songs}  {overall_acc:.1%}")
    print(f"  Skipped: {skipped}")

    if overall_acc < 0.70:
        print()
        print("  WARNING: accuracy below 70% threshold.")
        if not using_cdm:
            print("  Consider running Phase 2 CDM training:")
            print("    1. python training/extract_cdm_embeddings.py")
            print("    2. modal run training/modal_train_refractor_cdm.py")
            print("  Then re-validate with CDM enabled.")
        else:
            print("  Consider collecting more training data or increasing epochs.")

    # ------------------------------------------------------------------
    # Write YAML report
    # ------------------------------------------------------------------
    report = {
        "overall_accuracy": round(overall_acc, 4),
        "total_songs": total_songs,
        "total_correct": total_correct,
        "skipped": skipped,
        "chunk_size_s": chunk_size_s,
        "chunk_stride_s": chunk_stride_s,
        "refractor_cdm_enabled": using_cdm,
        "per_color": {
            color: {
                "n_correct": sum(by_color[color]),
                "n_total": len(by_color[color]),
                "accuracy": (
                    round(sum(by_color[color]) / len(by_color[color]), 4)
                    if by_color[color]
                    else None
                ),
            }
            for color in _COLOR_ORDER
            if by_color[color]
        },
        "per_song": per_song,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(
            report, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    print(f"\nReport written → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Refractor mix scoring against staged_raw_material ground truth."
    )
    parser.add_argument(
        "--artifacts-dir", default=None, help="staged_raw_material directory"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output YAML path (default: training/data/mix_scoring_validation.yml)",
    )
    parser.add_argument(
        "--onnx-path", default=None, help="Override Refractor ONNX path"
    )
    parser.add_argument(
        "--cdm-onnx-path",
        default=None,
        help="CDM ONNX path. Defaults to auto-detect; pass '' to disable.",
    )
    parser.add_argument(
        "--no-cdm",
        action="store_true",
        help="Disable CDM calibration head (A/B comparison mode)",
    )
    parser.add_argument("--chunk-size", type=float, default=30.0)
    parser.add_argument("--chunk-stride", type=float, default=5.0)
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    artifacts_dir = (
        Path(args.artifacts_dir)
        if args.artifacts_dir
        else repo_root / "staged_raw_material"
    )
    output_path = (
        Path(args.output)
        if args.output
        else repo_root / "training" / "data" / "mix_scoring_validation.yml"
    )

    if not artifacts_dir.exists():
        print(f"ERROR: {artifacts_dir} not found", file=sys.stderr)
        sys.exit(1)

    cdm_path = "" if args.no_cdm else args.cdm_onnx_path

    validate(
        artifacts_dir,
        output_path,
        onnx_path=args.onnx_path,
        cdm_onnx_path=cdm_path,
        chunk_size_s=args.chunk_size,
        chunk_stride_s=args.chunk_stride,
    )


if __name__ == "__main__":
    main()
