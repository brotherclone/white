#!/usr/bin/env python3
"""
Score a rendered audio bounce against the song's chromatic target.

Uses Refractor in audio-only mode (no MIDI, null concept embedding) to classify
the mix's perceived chromatic color, then computes drift vs. the target.

Writes melody/mix_score.yml with scores, chromatic match, and drift report.

Usage:
    python -m app.generators.midi.production.score_mix \
        --mix-file path/to/bounce.wav \
        --production-dir path/to/production/<song>
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

SUPPORTED_FORMATS = {".wav", ".aiff", ".aif", ".mp3"}
MIX_SCORE_FILENAME = "mix_score.yml"

# Mode order must match CHROMATIC_TARGETS arrays and Refractor output keys
_DIMS = [
    ("temporal", ["past", "present", "future"]),
    ("spatial", ["thing", "place", "person"]),
    ("ontological", ["imagined", "forgotten", "known"]),
]


# ---------------------------------------------------------------------------
# Audio encoding
# ---------------------------------------------------------------------------


def encode_audio_file(path: str | Path, scorer=None) -> np.ndarray:
    """Encode an audio file (WAV, AIFF, or MP3) to a 512-dim CLAP embedding.

    Args:
        path: Path to the audio file.
        scorer: Optional Refractor instance to reuse. Creates one if not provided.

    Returns:
        512-dim float32 numpy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported or unreadable.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported audio format '{suffix}'. Supported: WAV, AIFF, MP3."
        )

    import librosa

    try:
        waveform, sr = librosa.load(str(path), sr=None, mono=True)
    except Exception as exc:
        raise ValueError(f"Cannot read audio file {path}: {exc}") from exc

    if scorer is None:
        from training.refractor import Refractor

        scorer = Refractor()

    return scorer.prepare_audio(waveform, sr=int(sr))


# ---------------------------------------------------------------------------
# Per-dimension chromatic drift
# ---------------------------------------------------------------------------


def chromatic_drift_report(score_result: dict, target: dict) -> dict:
    """Compute per-dimension signed deltas between predicted and target distributions.

    For each dimension, the delta is:
        predicted_probability_of_target_peak_mode - target_probability_of_that_mode

    Negative means the mix under-predicts the target color; positive means it
    over-predicts. ``overall_drift`` is the mean absolute delta across all three
    dimensions.

    Args:
        score_result: Refractor output dict with "temporal", "spatial",
            "ontological" probability dicts.
        target: CHROMATIC_TARGETS entry — dict of lists, e.g.
            {"temporal": [0.8, 0.1, 0.1], "spatial": [...], "ontological": [...]}.

    Returns:
        Dict with temporal_delta, spatial_delta, ontological_delta, overall_drift.
    """
    deltas = {}
    abs_values = []

    for dim, modes in _DIMS:
        target_arr = np.array(target[dim], dtype=np.float32)
        peak_idx = int(np.argmax(target_arr))
        peak_mode = modes[peak_idx]

        pred_peak = float(score_result[dim][peak_mode])
        target_peak = float(target_arr[peak_idx])

        delta = round(pred_peak - target_peak, 4)
        deltas[f"{dim}_delta"] = delta
        abs_values.append(abs(delta))

    deltas["overall_drift"] = round(float(np.mean(abs_values)), 4)
    return deltas


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------


def write_mix_score(
    score_result: dict,
    drift: dict,
    melody_dir: str | Path,
    audio_path: Optional[str | Path] = None,
    onnx_path: Optional[str] = None,
) -> Path:
    """Write mix_score.yml to the song's melody directory.

    Overwrites any existing file.

    Returns:
        Path to the written file.
    """
    melody_dir = Path(melody_dir)
    melody_dir.mkdir(parents=True, exist_ok=True)

    # Default ONNX path for metadata (display only)
    if onnx_path is None:
        onnx_path = str(
            Path(__file__).parent.parent.parent.parent.parent
            / "training"
            / "data"
            / "refractor.onnx"
        )

    record = {
        "temporal": score_result["temporal"],
        "spatial": score_result["spatial"],
        "ontological": score_result["ontological"],
        "confidence": round(float(score_result["confidence"]), 4),
        "chromatic_match": round(float(score_result["chromatic_match"]), 4),
        "drift": drift,
        "metadata": {
            "audio_file": str(audio_path) if audio_path else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "onnx_path": onnx_path,
        },
    }

    out_path = melody_dir / MIX_SCORE_FILENAME
    with open(out_path, "w") as f:
        yaml.dump(
            record, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    return out_path


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------


def score_mix(
    audio_path: str | Path,
    production_dir: str | Path,
    onnx_path: Optional[str] = None,
    _scorer=None,
) -> tuple[dict, dict]:
    """Score a rendered audio bounce against the song's chromatic target.

    Encodes the audio with CLAP, runs Refractor in audio-only mode (no MIDI,
    null concept embedding), computes chromatic_match and per-dimension drift.

    Args:
        audio_path: Path to the rendered audio file (WAV, AIFF, or MP3).
        production_dir: Song production directory (must contain chords/review.yml).
        onnx_path: Optional override for Refractor ONNX path.
        _scorer: Injected Refractor instance (for testing).

    Returns:
        Tuple of (score_result, drift_report).
        score_result contains temporal/spatial/ontological dicts, confidence,
        and chromatic_match. drift_report contains per-dimension deltas and
        overall_drift.
    """
    from app.generators.midi.pipelines.lyric_pipeline import _find_and_load_proposal
    from app.generators.midi.pipelines.chord_pipeline import (
        compute_chromatic_match,
        get_chromatic_target,
    )

    production_dir = Path(production_dir)
    audio_path = Path(audio_path)

    # Load song color from proposal (via chords/review.yml)
    meta = _find_and_load_proposal(production_dir)
    color = meta.get("color", "White")
    target = get_chromatic_target(color)

    # Build or reuse scorer
    if _scorer is None:
        from training.refractor import Refractor

        _scorer = Refractor(onnx_path=onnx_path)

    # Encode audio
    audio_emb = encode_audio_file(audio_path, scorer=_scorer)

    # Encode concept — always present in training, never dropped; zero embedding is OOD
    concept_text = meta.get("concept", "")
    concept_emb = (
        _scorer.prepare_concept(concept_text)
        if concept_text
        else np.zeros(768, dtype=np.float32)
    )

    score_result = _scorer.score(audio_emb=audio_emb, concept_emb=concept_emb)

    # Chromatic match and drift
    score_result["chromatic_match"] = round(
        float(compute_chromatic_match(score_result, target)), 4
    )
    drift = chromatic_drift_report(score_result, target)

    return score_result, drift


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a rendered audio bounce against its chromatic target."
    )
    parser.add_argument(
        "--mix-file", required=True, help="Path to audio bounce (WAV/AIFF/MP3)"
    )
    parser.add_argument(
        "--production-dir", required=True, help="Song production directory"
    )
    parser.add_argument(
        "--onnx-path", default=None, help="Override Refractor ONNX path"
    )
    args = parser.parse_args()

    prod = Path(args.production_dir)
    if not prod.exists():
        print(f"ERROR: Production directory not found: {prod}", file=sys.stderr)
        sys.exit(1)

    mix_file = Path(args.mix_file)
    if not mix_file.exists():
        print(f"ERROR: Mix file not found: {mix_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Scoring mix: {mix_file.name}")
    print(f"Production:  {prod.name}")

    score_result, drift = score_mix(mix_file, prod, onnx_path=args.onnx_path)

    melody_dir = prod / "melody"
    out_path = write_mix_score(
        score_result, drift, melody_dir, audio_path=mix_file, onnx_path=args.onnx_path
    )

    # Human-readable summary
    print()
    print("Chromatic scores:")
    for dim in ("temporal", "spatial", "ontological"):
        dist = score_result[dim]
        top = max(dist, key=dist.get)
        print(f"  {dim:<14} {top} ({dist[top]:.3f})")
    print(f"  confidence     {score_result['confidence']:.3f}")
    print(f"  chromatic_match {score_result['chromatic_match']:.3f}")
    print()
    print("Drift vs target:")
    print(f"  temporal_delta    {drift['temporal_delta']:+.4f}")
    print(f"  spatial_delta     {drift['spatial_delta']:+.4f}")
    print(f"  ontological_delta {drift['ontological_delta']:+.4f}")
    print(f"  overall_drift     {drift['overall_drift']:.4f}")
    print()
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
