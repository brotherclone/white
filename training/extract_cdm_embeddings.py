#!/usr/bin/env python3
"""
Extract per-chunk CLAP embeddings from staged_raw_material/_main.wav files.

Runs locally — reads audio from disk, encodes each 30s chunk with CLAP via
Refractor, and saves all embeddings + metadata to a single .npz cache file
at training/data/refractor_cdm_embeddings.npz.

This is a one-time preprocessing step. The resulting .npz is consumed by
modal_train_refractor_cdm.py (and optionally local training).

Run time: ~30–60 min on CPU (depends on CLAP speed per chunk).

Usage:
    python training/extract_cdm_embeddings.py
    python training/extract_cdm_embeddings.py --artifacts-dir /path/to/staged_raw_material
    python training/extract_cdm_embeddings.py --output training/data/refractor_cdm_embeddings.npz
    python training/extract_cdm_embeddings.py --chunk-size 30 --chunk-stride 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

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


def extract(
    artifacts_dir: Path,
    output_path: Path,
    chunk_size_s: float = 30.0,
    chunk_stride_s: float = 5.0,
) -> None:
    """Extract CLAP embeddings for all labeled _main.wav files.

    Saves an .npz with keys:
        clap_embs   : float32 (N, 512) — per-chunk CLAP embeddings
        concept_embs: float32 (N, 768) — per-chunk concept embeddings (song-level broadcast)
        colors      : str array (N,)   — color name per chunk
        song_ids    : str array (N,)   — song id per chunk (for train/val split by song)
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from app.generators.midi.production.score_mix import chunk_audio
    from training.refractor import Refractor

    scorer = Refractor()

    song_dirs = sorted(d for d in artifacts_dir.iterdir() if d.is_dir())
    print(f"Found {len(song_dirs)} song directories")
    print(f"Chunking: {chunk_size_s}s windows, {chunk_stride_s}s stride\n")

    all_clap: list[np.ndarray] = []
    all_concept: list[np.ndarray] = []
    all_colors: list[str] = []
    all_song_ids: list[str] = []

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

        try:
            import librosa

            waveform, sr = librosa.load(str(main_wav), sr=None, mono=True)
            chunks = chunk_audio(
                waveform, int(sr), chunk_size_s=chunk_size_s, stride_s=chunk_stride_s
            )

            concept_emb = (
                scorer.prepare_concept(concept_text)
                if concept_text
                else np.zeros(768, dtype=np.float32)
            )

            song_clap = [scorer.prepare_audio(c, sr=48000) for c in chunks]

            all_clap.extend(song_clap)
            all_concept.extend([concept_emb] * len(chunks))
            all_colors.extend([color] * len(chunks))
            all_song_ids.extend([song_id] * len(chunks))

            elapsed = time.time() - t0
            avg = elapsed / (i + 1 - skipped)
            remaining = avg * (len(song_dirs) - i - 1)
            print(
                f"  [{i+1:2d}/{len(song_dirs)}] {song_id} ({color:7s}) "
                f"{len(chunks)} chunks — {elapsed:.0f}s elapsed, ~{remaining:.0f}s left"
            )

        except Exception as exc:
            print(f"  [ERR] {song_id}: {exc}", file=sys.stderr)
            skipped += 1

    clap_arr = np.array(all_clap, dtype=np.float32)
    concept_arr = np.array(all_concept, dtype=np.float32)
    colors_arr = np.array(all_colors)
    ids_arr = np.array(all_song_ids)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        clap_embs=clap_arr,
        concept_embs=concept_arr,
        colors=colors_arr,
        song_ids=ids_arr,
    )

    total_time = time.time() - t0
    print(f"\nExtracted {len(all_clap)} chunks from {len(song_dirs) - skipped} songs")
    print(f"Skipped: {skipped}")
    print(f"clap_embs shape:    {clap_arr.shape}")
    print(f"concept_embs shape: {concept_arr.shape}")
    print(f"Saved → {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Total time: {total_time/60:.1f} min")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract CLAP embeddings from staged_raw_material for CDM training."
    )
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npz path (default: training/data/refractor_cdm_embeddings.npz)",
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
        else repo_root / "training" / "data" / "refractor_cdm_embeddings.npz"
    )

    if not artifacts_dir.exists():
        print(f"ERROR: {artifacts_dir} not found", file=sys.stderr)
        sys.exit(1)

    extract(artifacts_dir, output_path, args.chunk_size, args.chunk_stride)


if __name__ == "__main__":
    main()
