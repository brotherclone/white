#!/usr/bin/env python3
"""
Score a rendered audio bounce against the song's chromatic target.

Uses Refractor in audio-only mode (no MIDI, null concept embedding) to classify
the mix's perceived chromatic color, then computes drift vs. the target.

Audio is split into overlapping 30s chunks; per-chunk scores are aggregated
with confidence-weighted mean pooling.

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
# Chunked audio helpers
# ---------------------------------------------------------------------------


def chunk_audio(
    waveform: np.ndarray,
    sr: int,
    chunk_size_s: float = 30.0,
    stride_s: float = 5.0,
) -> list[np.ndarray]:
    """Split a waveform into overlapping fixed-length windows at 48 kHz.

    The waveform is first resampled to 48 kHz (CLAP native rate).  If the
    audio is shorter than one chunk window, a single zero-padded chunk is
    returned.

    Args:
        waveform: 1-D float32 audio array at any sample rate.
        sr: Sample rate of ``waveform``.
        chunk_size_s: Window length in seconds (default 30 s).
        stride_s: Hop between windows in seconds (default 5 s).

    Returns:
        List of float32 numpy arrays, each exactly ``chunk_size_s * 48000``
        samples long.
    """
    if chunk_size_s <= 0:
        raise ValueError(f"chunk_size_s must be > 0, got {chunk_size_s}")
    if stride_s <= 0:
        raise ValueError(f"stride_s must be > 0, got {stride_s}")

    import librosa

    TARGET_SR = 48000
    if sr != TARGET_SR:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)

    chunk_len = int(chunk_size_s * TARGET_SR)
    stride_len = int(stride_s * TARGET_SR)
    total_len = len(waveform)

    # Short audio: pad and return a single chunk
    if total_len <= chunk_len:
        chunk = np.zeros(chunk_len, dtype=np.float32)
        chunk[:total_len] = waveform[:total_len]
        return [chunk]

    chunks = []
    start = 0
    while start < total_len:
        end = start + chunk_len
        segment = waveform[start:end]
        if len(segment) < chunk_len:
            padded = np.zeros(chunk_len, dtype=np.float32)
            padded[: len(segment)] = segment
            segment = padded
        chunks.append(segment.astype(np.float32))
        start += stride_len

    return chunks


def aggregate_chunk_scores(results: list[dict]) -> dict:
    """Aggregate per-chunk Refractor scores via confidence-weighted mean.

    If all chunks have zero confidence, falls back to a uniform (unweighted)
    mean.  The aggregated confidence is the arithmetic mean of per-chunk
    confidences.

    Args:
        results: Non-empty list of Refractor score dicts, each containing
            ``temporal``, ``spatial``, ``ontological`` (mode-keyed probability
            dicts) and ``confidence`` (float).

    Returns:
        Single aggregated score dict with the same shape as an individual
        result, plus ``chunk_count`` (int).

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("results must be non-empty")

    weights = np.array([r["confidence"] for r in results], dtype=np.float64)
    total_w = weights.sum()
    if total_w == 0.0:
        weights = np.ones(len(results), dtype=np.float64)
        total_w = float(len(results))
    weights = weights / total_w

    agg: dict = {}
    for dim, modes in _DIMS:
        agg[dim] = {}
        for mode in modes:
            agg[dim][mode] = float(
                sum(w * r[dim][mode] for w, r in zip(weights, results))
            )

    agg["confidence"] = float(np.mean([r["confidence"] for r in results]))
    agg["chunk_count"] = len(results)
    return agg


# ---------------------------------------------------------------------------
# Audio encoding
# ---------------------------------------------------------------------------


def encode_audio_file(
    path: str | Path,
    scorer=None,
    chunk_size_s: float = 30.0,
    chunk_stride_s: float = 5.0,
) -> list[np.ndarray]:
    """Encode an audio file (WAV, AIFF, or MP3) to per-chunk CLAP embeddings.

    The file is chunked into overlapping 30 s windows (by default); each chunk
    is encoded to a 512-dim CLAP embedding.

    Args:
        path: Path to the audio file.
        scorer: Optional Refractor instance to reuse.
        chunk_size_s: Chunk window length in seconds.
        chunk_stride_s: Hop between chunks in seconds.

    Returns:
        List of 512-dim float32 numpy arrays (one per chunk).

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

    chunks = chunk_audio(
        waveform, int(sr), chunk_size_s=chunk_size_s, stride_s=chunk_stride_s
    )
    return [scorer.prepare_audio(c, sr=48000) for c in chunks]


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
    cdm_onnx_path: Optional[str] = None,
    chunk_size_s: float = 30.0,
    chunk_stride_s: float = 5.0,
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
        "chunk_count": score_result.get("chunk_count", 1),
        "chunk_size_s": chunk_size_s,
        "chunk_stride_s": chunk_stride_s,
        "drift": drift,
        "metadata": {
            "audio_file": str(audio_path) if audio_path else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "onnx_path": onnx_path,
            "refractor_cdm": cdm_onnx_path,
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
    cdm_onnx_path: Optional[str] = None,
    chunk_size_s: float = 30.0,
    chunk_stride_s: float = 5.0,
    _scorer=None,
) -> tuple[dict, dict]:
    """Score a rendered audio bounce against the song's chromatic target.

    Encodes the audio in overlapping 30 s chunks with CLAP, runs Refractor in
    audio-only mode for each chunk, aggregates with confidence-weighted mean,
    then computes chromatic_match and per-dimension drift.

    Args:
        audio_path: Path to the rendered audio file (WAV, AIFF, or MP3).
        production_dir: Song production directory (must contain chords/review.yml).
        onnx_path: Optional override for Refractor ONNX path.
        cdm_onnx_path: Optional CDM ONNX path. None = auto-detect; "" = disable.
        chunk_size_s: Audio chunk window length in seconds (default 30).
        chunk_stride_s: Hop between chunks in seconds (default 5).
        _scorer: Injected Refractor instance (for testing).

    Returns:
        Tuple of (score_result, drift_report).
        score_result contains temporal/spatial/ontological dicts, confidence,
        chromatic_match, and chunk_count. drift_report contains per-dimension
        deltas and overall_drift.
    """
    from app.generators.midi.pipelines.chord_pipeline import (
        compute_chromatic_match,
        get_chromatic_target,
    )
    from app.generators.midi.pipelines.lyric_pipeline import _find_and_load_proposal

    production_dir = Path(production_dir)
    audio_path = Path(audio_path)

    # Load song color from proposal (via chords/review.yml)
    meta = _find_and_load_proposal(production_dir)
    color = meta.get("color", "White")
    target = get_chromatic_target(color)

    # Build or reuse scorer
    if _scorer is None:
        from training.refractor import Refractor

        _scorer = Refractor(onnx_path=onnx_path, cdm_onnx_path=cdm_onnx_path)

    # Encode concept — always present in training, never dropped; zero embedding is OOD
    concept_text = meta.get("concept", "")
    concept_emb = (
        _scorer.prepare_concept(concept_text)
        if concept_text
        else np.zeros(768, dtype=np.float32)
    )

    # Encode audio into per-chunk CLAP embeddings
    chunk_embs = encode_audio_file(
        audio_path,
        scorer=_scorer,
        chunk_size_s=chunk_size_s,
        chunk_stride_s=chunk_stride_s,
    )

    # Score each chunk
    chunk_results = [
        _scorer.score(audio_emb=emb, concept_emb=concept_emb) for emb in chunk_embs
    ]

    # Aggregate
    score_result = aggregate_chunk_scores(chunk_results)

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
    parser.add_argument(
        "--cdm-onnx-path",
        default=None,
        help="Refractor CDM ONNX path. Defaults to auto-detect training/data/refractor_cdm.onnx; pass '' to disable.",
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=30.0,
        help="Audio chunk window length in seconds (default: 30)",
    )
    parser.add_argument(
        "--chunk-stride",
        type=float,
        default=5.0,
        help="Hop between chunk windows in seconds (default: 5)",
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

    cdm_path = args.cdm_onnx_path  # None = auto-detect, "" = disabled

    score_result, drift = score_mix(
        mix_file,
        prod,
        onnx_path=args.onnx_path,
        cdm_onnx_path=cdm_path,
        chunk_size_s=args.chunk_size,
        chunk_stride_s=args.chunk_stride,
    )

    melody_dir = prod / "melody"
    out_path = write_mix_score(
        score_result,
        drift,
        melody_dir,
        audio_path=mix_file,
        onnx_path=args.onnx_path,
        cdm_onnx_path=cdm_path,
        chunk_size_s=args.chunk_size,
        chunk_stride_s=args.chunk_stride,
    )

    # Human-readable summary
    print()
    print(f"Chunks scored: {score_result.get('chunk_count', 1)}")
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
