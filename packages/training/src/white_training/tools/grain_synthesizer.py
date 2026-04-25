#!/usr/bin/env python3
"""
Granular grain synthesizer — chromatic audio texture from Refractor-scored corpus segments.

Retrieves the top-N segments for a target color using retrieve_by_color(), draws
random 1-second grains from the source audio files, crossfades them with a Hann
window, and writes a WAV + grain_map.yml.

The output is a chromatic collage from the White corpus: not new synthesis, but a
targeted texture that carries the chromatic signature of the selected segments.

Usage:
    python -m training.tools.grain_synthesizer \
        --color Red --duration 30 --output ./red_texture.wav

    python -m training.tools.grain_synthesizer \
        --color Violet --duration 60 --top-n 30 --seed 7 --crossfade-ms 80
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import yaml

# ---------------------------------------------------------------------------
# Grain pool loading
# ---------------------------------------------------------------------------


def load_grain_pool(
    color: str,
    top_n: int = 20,
    clap_parquet: Optional[str] = None,
    meta_parquet: Optional[str] = None,
) -> list[dict]:
    """Retrieve top-N Refractor-scored segments for a color.

    Returns a list of dicts with keys:
        segment_id, source_audio_file, start_seconds, end_seconds, match, song_slug
    Only segments with a reachable source_audio_file are included.
    """
    from white_composition.retrieve_samples import (
        load_clap_index,
        retrieve_by_color,
    )

    df = load_clap_index(parquet_path=clap_parquet, meta_parquet_path=meta_parquet)
    results = retrieve_by_color(df, color, top_n=top_n)

    pool = []
    skipped = 0
    for r in results:
        src = r.get("source_audio_file")
        start = r.get("start_seconds")
        end = r.get("end_seconds")
        if not src or start is None or end is None:
            skipped += 1
            continue
        if not Path(src).exists():
            skipped += 1
            continue
        pool.append(
            {
                "segment_id": r["segment_id"],
                "source_audio_file": src,
                "start_seconds": float(start),
                "end_seconds": float(end),
                "match": float(r["match"]),
                "song_slug": r.get("song_slug", ""),
            }
        )

    if skipped:
        print(f"  Note: {skipped} segment(s) skipped (no reachable audio)")

    return pool


# ---------------------------------------------------------------------------
# Grain extraction
# ---------------------------------------------------------------------------


def extract_grain(
    source_path: str,
    segment_start: float,
    segment_end: float,
    grain_dur: float = 1.0,
    rng: Optional[random.Random] = None,
) -> tuple[np.ndarray, int]:
    """Load a single grain from a source audio file.

    Picks a random offset within [segment_start, segment_end - grain_dur].
    Returns (grain_float32_array, sample_rate).

    Raises ValueError if the segment is shorter than grain_dur.
    """
    if rng is None:
        rng = random.Random()

    usable = segment_end - grain_dur - segment_start
    if usable < 0:
        # Segment shorter than grain_dur — use whole segment
        offset = segment_start
    else:
        offset = segment_start + rng.uniform(0, usable)

    data, sr = sf.read(str(source_path), dtype="float32", always_2d=True)
    s = int(offset * sr)
    e = s + int(grain_dur * sr)
    grain = data[s:e]

    # Pad if we hit the end of file
    if len(grain) < int(grain_dur * sr):
        pad = np.zeros(
            (int(grain_dur * sr) - len(grain), grain.shape[1]), dtype=np.float32
        )
        grain = np.concatenate([grain, pad], axis=0)

    return grain, sr


# ---------------------------------------------------------------------------
# Crossfade + assembly
# ---------------------------------------------------------------------------


def _to_stereo(grain: np.ndarray) -> np.ndarray:
    """Convert mono (N,1) or (N,) to stereo (N,2). Stereo passes through."""
    if grain.ndim == 1:
        grain = grain[:, np.newaxis]
    if grain.shape[1] == 1:
        return np.concatenate([grain, grain], axis=1)
    return grain


def hann_crossfade(
    grains: list[np.ndarray],
    sr: int,
    crossfade_ms: float = 50,
) -> np.ndarray:
    """Concatenate grains with Hann-windowed crossfades.

    All grains must be (N, 2) float32 stereo arrays with the same sample rate.
    crossfade_ms: length of the overlap region in milliseconds.

    Returns a single (M, 2) float32 array.
    """
    if not grains:
        return np.zeros((0, 2), dtype=np.float32)

    cf_samples = int(sr * crossfade_ms / 1000)
    # Ensure crossfade doesn't exceed half a grain
    if grains:
        cf_samples = min(cf_samples, len(grains[0]) // 2)

    if cf_samples < 2 or len(grains) == 1:
        return np.concatenate(grains, axis=0)

    # Hann fade-out and fade-in curves
    fade_out = np.hanning(cf_samples * 2)[:cf_samples, np.newaxis].astype(np.float32)
    fade_in = np.hanning(cf_samples * 2)[cf_samples:, np.newaxis].astype(np.float32)

    result = grains[0].copy()
    for grain in grains[1:]:
        # Apply fade to tail of result and head of incoming grain
        result[-cf_samples:] *= fade_out
        incoming = grain.copy()
        incoming[:cf_samples] *= fade_in
        # Overlap-add: trim result tail by cf_samples and add overlap
        body = result[:-cf_samples]
        overlap = result[-cf_samples:] + incoming[:cf_samples]
        tail = incoming[cf_samples:]
        result = np.concatenate([body, overlap, tail], axis=0)

    return result


# ---------------------------------------------------------------------------
# End-to-end synthesis
# ---------------------------------------------------------------------------


def synthesize(
    color: str,
    duration_s: float = 30.0,
    top_n: int = 20,
    output_path: Optional[str] = None,
    clap_parquet: Optional[str] = None,
    meta_parquet: Optional[str] = None,
    seed: int = 42,
    grain_dur_s: float = 1.0,
    crossfade_ms: float = 50.0,
    grain_pool: Optional[list[dict]] = None,
) -> tuple[Path, Path]:
    """End-to-end granular synthesis.

    Loads grain pool, draws grains randomly with replacement until duration_s is
    reached, crossfades them, writes WAV + grain_map.yml.

    Args:
        color: Target color name.
        duration_s: Target texture duration in seconds.
        top_n: Number of Refractor-scored segments in the grain pool.
        output_path: Path for the output WAV (default: ./grain_output/<color>_texture.wav).
        clap_parquet: Override CLAP parquet path.
        meta_parquet: Override metadata parquet path.
        seed: Random seed for reproducibility.
        grain_dur_s: Duration of each grain in seconds.
        crossfade_ms: Crossfade length in milliseconds.
        grain_pool: Pre-loaded pool (skips retrieve_by_color; useful for testing).

    Returns:
        (wav_path, grain_map_path)

    Raises:
        ValueError: If the grain pool is empty.
    """
    rng = random.Random(seed)

    # Load pool
    if grain_pool is None:
        print(f"Retrieving top-{top_n} {color} segments...")
        grain_pool = load_grain_pool(color, top_n, clap_parquet, meta_parquet)

    if not grain_pool:
        raise ValueError(
            f"No reachable audio segments found for color '{color}'. "
            "Check that source audio files are accessible locally."
        )

    print(f"  Pool: {len(grain_pool)} segments, top match={grain_pool[0]['match']:.4f}")

    # Resolve output path
    if output_path is None:
        out_dir = Path("./grain_output")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{color.lower()}_texture.wav")
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Determine sample rate from first pool entry
    _, sr = sf.read(str(grain_pool[0]["source_audio_file"]), stop=1, dtype="float32")

    n_grains_needed = (
        int(np.ceil(duration_s / grain_dur_s)) + 1
    )  # +1 for crossfade overlap

    print(
        f"Extracting {n_grains_needed} grains ({grain_dur_s}s each, {crossfade_ms}ms crossfade)..."
    )

    grains_used = []
    grain_arrays = []

    for i in range(n_grains_needed):
        seg = rng.choice(grain_pool)
        try:
            grain, grain_sr = extract_grain(
                seg["source_audio_file"],
                seg["start_seconds"],
                seg["end_seconds"],
                grain_dur=grain_dur_s,
                rng=rng,
            )
        except Exception as exc:
            print(f"  Warning: skipping grain from {seg['segment_id']}: {exc}")
            continue

        # Normalise to stereo
        grain = _to_stereo(grain)

        # Resample if SR mismatch (rare but possible)
        if grain_sr != sr:
            # Simple integer-ratio resampling fallback
            ratio = sr / grain_sr
            new_len = int(len(grain) * ratio)
            indices = np.linspace(0, len(grain) - 1, new_len)
            grain = np.stack(
                [
                    np.interp(indices, np.arange(len(grain)), grain[:, c])
                    for c in range(grain.shape[1])
                ],
                axis=1,
            ).astype(np.float32)

        grain_arrays.append(grain)
        offset_s = seg["start_seconds"] + rng.uniform(
            0, max(0, seg["end_seconds"] - grain_dur_s - seg["start_seconds"])
        )
        grains_used.append(
            {
                "grain_index": i,
                "segment_id": seg["segment_id"],
                "song_slug": seg.get("song_slug", ""),
                "source": seg["source_audio_file"],
                "offset_s": round(offset_s, 4),
                "match": seg["match"],
            }
        )

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_grains_needed} grains extracted...")

    if not grain_arrays:
        raise ValueError("All grain extractions failed — no audio written.")

    print("Crossfading...")
    texture = hann_crossfade(grain_arrays, sr, crossfade_ms=crossfade_ms)

    # Trim to exact duration
    target_samples = int(duration_s * sr)
    texture = texture[:target_samples]

    # Normalise peak to -1 dBFS
    peak = np.abs(texture).max()
    if peak > 0:
        texture = texture * (0.891 / peak)  # -1 dBFS

    # Write WAV
    wav_path = Path(output_path)
    sf.write(str(wav_path), texture, sr, subtype="PCM_24")
    actual_dur = len(texture) / sr
    print(f"Written: {wav_path}  ({actual_dur:.1f}s, {sr}Hz stereo)")

    # Write grain map
    grain_map_path = wav_path.with_name(wav_path.stem + "_grain_map.yml")
    grain_map = {
        "color": color,
        "generated": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(actual_dur, 3),
        "grain_dur_s": grain_dur_s,
        "crossfade_ms": crossfade_ms,
        "seed": seed,
        "pool_size": len(grain_pool),
        "grains": grains_used,
    }
    with open(grain_map_path, "w") as f:
        yaml.dump(
            grain_map, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
    print(f"Grain map: {grain_map_path}")

    return wav_path, grain_map_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Granular grain synthesizer — chromatic texture from Refractor-scored corpus."
    )
    parser.add_argument(
        "--color", required=True, help="Target color (Red, Blue, Violet, ...)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Texture duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--top-n", type=int, default=20, help="Segments in grain pool (default: 20)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output WAV path (default: ./grain_output/<color>_texture.wav)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--grain-dur",
        type=float,
        default=1.0,
        help="Grain duration in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--crossfade-ms",
        type=float,
        default=50.0,
        help="Crossfade length in ms (default: 50)",
    )
    parser.add_argument("--parquet", default=None, help="Override CLAP parquet path")

    args = parser.parse_args()

    print("=" * 60)
    print("GRANULAR GRAIN SYNTHESIZER")
    print("=" * 60)

    try:
        wav_path, map_path = synthesize(
            color=args.color,
            duration_s=args.duration,
            top_n=args.top_n,
            output_path=args.output,
            clap_parquet=args.parquet,
            seed=args.seed,
            grain_dur_s=args.grain_dur,
            crossfade_ms=args.crossfade_ms,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
