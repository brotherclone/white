#!/usr/bin/env python3
"""
Chromatic sample retrieval — find corpus segments that best match a color target.

Loads precomputed CLAP embeddings from the training parquet, scores each segment
against the requested color using Refractor in audio-only mode, and writes a
ranked `sample_map.yml`. Optionally copies matched audio files to a local directory.

Usage:
    python -m app.generators.midi.production.retrieve_samples \
        --color Red --top-n 10 --output-dir ./sample_retrieval

    python -m app.generators.midi.production.retrieve_samples \
        --color Blue --top-n 5 --copy-audio
"""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import yaml

if TYPE_CHECKING:
    import pandas as pd

# Default paths
_ROOT = Path(__file__).parent.parent.parent.parent.parent
DEFAULT_CLAP_PARQUET = str(
    _ROOT / "training" / "data" / "training_data_clap_embeddings.parquet"
)
DEFAULT_META_PARQUET = str(
    _ROOT / "training" / "data" / "training_data_with_embeddings.parquet"
)

HF_REPO_ID = "earthlyframes/white-training-data"
HF_CLAP_FILENAME = "data/training_data_clap_embeddings.parquet"

# Valid colors come from CHROMATIC_TARGETS in chord_pipeline
VALID_COLORS = [
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


# ---------------------------------------------------------------------------
# Parquet loading
# ---------------------------------------------------------------------------


def load_clap_index(
    parquet_path: Optional[str] = None,
    meta_parquet_path: Optional[str] = None,
) -> "pd.DataFrame":
    """Load CLAP embeddings merged with segment metadata.

    Tries the local path first; falls back to hf_hub_download if not found.
    Merges with the metadata parquet to provide color, song_slug, and audio_path.

    Returns a DataFrame with columns:
        segment_id, song_slug, color, clap_embedding, audio_path
    (plus temporal/spatial/ontological/confidence if present in the parquet).
    """
    import pandas as pd

    # --- Load CLAP embeddings ---
    clap_path = parquet_path or DEFAULT_CLAP_PARQUET
    if not Path(clap_path).exists():
        print(
            f"CLAP parquet not found locally ({clap_path}), downloading from HuggingFace..."
        )
        try:
            from huggingface_hub import hf_hub_download

            clap_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_CLAP_FILENAME,
                repo_type="dataset",
            )
        except Exception as exc:
            raise FileNotFoundError(
                f"CLAP parquet not found at {clap_path} and HuggingFace download failed: {exc}"
            ) from exc

    clap_df = pd.read_parquet(clap_path)

    # Normalise: the column may be audio_embedding or clap_embedding
    if "audio_embedding" in clap_df.columns and "clap_embedding" not in clap_df.columns:
        clap_df = clap_df.rename(columns={"audio_embedding": "clap_embedding"})

    # Drop rows with no embedding
    if "has_audio_embedding" in clap_df.columns:
        clap_df = clap_df[clap_df["has_audio_embedding"]].drop(
            columns=["has_audio_embedding"], errors="ignore"
        )

    # --- Load metadata for color / song title / audio path ---
    meta_path = meta_parquet_path or DEFAULT_META_PARQUET
    if Path(meta_path).exists():
        meta_df = pd.read_parquet(
            meta_path,
            columns=[
                "segment_id",
                "rainbow_color",
                "title",
                "segment_audio_file",
                "source_audio_file",
                "start_seconds",
                "end_seconds",
            ],
        )
        meta_df = meta_df.rename(
            columns={
                "rainbow_color": "color",
                "title": "song_slug",
                "segment_audio_file": "audio_path",
            }
        )
        clap_df = clap_df.merge(meta_df, on="segment_id", how="left")
    else:
        warnings.warn(
            f"Metadata parquet not found at {meta_path}; color and audio_path will be missing.",
            stacklevel=2,
        )
        clap_df["color"] = None
        clap_df["song_slug"] = None
        clap_df["audio_path"] = None

    # Ensure score columns exist as NaN if not present
    for col in ("temporal", "spatial", "ontological", "confidence"):
        if col not in clap_df.columns:
            clap_df[col] = None

    return clap_df


# ---------------------------------------------------------------------------
# Chromatic scoring helpers
# ---------------------------------------------------------------------------


def _compute_chromatic_match_for_df(
    df: "pd.DataFrame",
    color: str,
    _scorer=None,
) -> "pd.Series":
    """Compute chromatic_match scores for all rows in df.

    Prefers precomputed Refractor columns (temporal/spatial/ontological).
    Falls back to running Refractor in audio-only mode on each CLAP embedding.
    """
    from app.generators.midi.pipelines.chord_pipeline import (
        compute_chromatic_match,
        get_chromatic_target,
    )

    target = get_chromatic_target(color)

    # Check if precomputed distributions are present (all rows non-null in temporal col)
    has_precomputed = (
        "temporal" in df.columns
        and df["temporal"].notna().any()
        and isinstance(df["temporal"].dropna().iloc[0], dict)
    )

    if has_precomputed:
        return df["temporal"].apply(
            lambda t: (
                compute_chromatic_match(
                    {
                        "temporal": t,
                        "spatial": df.loc[df["temporal"] == t, "spatial"].iloc[0],
                        "ontological": df.loc[df["temporal"] == t, "ontological"].iloc[
                            0
                        ],
                        "confidence": 1.0,
                    },
                    target,
                )
                if isinstance(t, dict)
                else 0.0
            )
        )

    # Fall back: run Refractor in audio-only mode
    if _scorer is None:
        try:
            from training.refractor import Refractor

            _scorer = Refractor()
        except Exception as exc:
            warnings.warn(
                f"Refractor unavailable, chromatic_match set to 0.0: {exc}",
                stacklevel=2,
            )
            import pandas as pd

            return pd.Series(0.0, index=df.index)

    scores = []
    null_concept = np.zeros(768, dtype=np.float32)
    for emb in df["clap_embedding"]:
        try:
            result = _scorer.score(
                audio_emb=np.array(emb, dtype=np.float32), concept_emb=null_concept
            )
            scores.append(float(compute_chromatic_match(result, target)))
        except Exception:
            scores.append(0.0)

    import pandas as pd

    return pd.Series(scores, index=df.index)


# ---------------------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------------------


def retrieve_by_color(
    df: "pd.DataFrame",
    color: str,
    top_n: int = 10,
    _scorer=None,
) -> list[dict]:
    """Return top-N segments matching a color target, sorted by chromatic_match.

    Args:
        df: DataFrame from load_clap_index.
        color: Target color name (Red, Blue, Violet, etc.).
        top_n: Maximum results to return.
        _scorer: Optional injected Refractor instance (for testing).

    Returns:
        List of dicts with: rank, segment_id, song_slug, color, match, audio_path.

    Raises:
        ValueError: If color is not in VALID_COLORS.
    """
    if color not in VALID_COLORS:
        raise ValueError(
            f"Unknown color '{color}'. Valid colors: {', '.join(VALID_COLORS)}"
        )

    filtered = df[df["color"] == color].copy()
    if filtered.empty:
        return []

    filtered["chromatic_match"] = _compute_chromatic_match_for_df(
        filtered, color, _scorer
    )
    filtered = filtered.sort_values("chromatic_match", ascending=False)
    top = filtered.head(top_n)

    results = []
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        results.append(
            {
                "rank": rank,
                "segment_id": str(row["segment_id"]),
                "song_slug": str(row.get("song_slug") or ""),
                "color": str(row.get("color") or color),
                "match": round(float(row["chromatic_match"]), 4),
                "audio_path": (
                    str(row["audio_path"])
                    if row.get("audio_path") is not None
                    else None
                ),
                "source_audio_file": (
                    str(row["source_audio_file"])
                    if row.get("source_audio_file") is not None
                    else None
                ),
                "start_seconds": (
                    float(row["start_seconds"])
                    if row.get("start_seconds") is not None
                    else None
                ),
                "end_seconds": (
                    float(row["end_seconds"])
                    if row.get("end_seconds") is not None
                    else None
                ),
            }
        )

    return results


def retrieve_by_clap_similarity(
    df: "pd.DataFrame",
    query_embedding: np.ndarray,
    top_n: int = 10,
) -> list[dict]:
    """Return top-N segments by cosine similarity to a query CLAP embedding.

    Cross-color retrieval — color label is preserved in results but not used for filtering.

    Args:
        df: DataFrame from load_clap_index.
        query_embedding: 512-dim float32 numpy array.
        top_n: Maximum results to return.

    Returns:
        List of dicts with: rank, segment_id, song_slug, color, similarity, audio_path.
    """

    embeddings = np.stack(df["clap_embedding"].values).astype(np.float32)

    # Normalise
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings_norm = embeddings / emb_norms

    similarities = embeddings_norm @ query_norm  # shape (N,)

    top_indices = np.argsort(similarities)[::-1][:top_n]
    top_df = df.iloc[top_indices].copy()
    top_sims = similarities[top_indices]

    results = []
    for rank, ((_, row), sim) in enumerate(zip(top_df.iterrows(), top_sims), start=1):
        results.append(
            {
                "rank": rank,
                "segment_id": str(row["segment_id"]),
                "song_slug": str(row.get("song_slug") or ""),
                "color": str(row.get("color") or ""),
                "similarity": round(float(sim), 4),
                "audio_path": (
                    str(row["audio_path"])
                    if row.get("audio_path") is not None
                    else None
                ),
                "source_audio_file": (
                    str(row["source_audio_file"])
                    if row.get("source_audio_file") is not None
                    else None
                ),
                "start_seconds": (
                    float(row["start_seconds"])
                    if row.get("start_seconds") is not None
                    else None
                ),
                "end_seconds": (
                    float(row["end_seconds"])
                    if row.get("end_seconds") is not None
                    else None
                ),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_sample_map(
    results: list[dict],
    output_dir: Path,
    color: str,
) -> Path:
    """Write sample_map.yml to output_dir.

    Args:
        results: From retrieve_by_color or retrieve_by_clap_similarity.
        output_dir: Directory to write into (created if missing).
        color: Color used for retrieval (written to YAML header).

    Returns:
        Path to the written sample_map.yml.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "color": color,
        "generated": datetime.now(timezone.utc).isoformat(),
        "count": len(results),
        "results": results,
    }

    out_path = output_dir / "sample_map.yml"
    with open(out_path, "w") as f:
        yaml.dump(
            data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    return out_path


def copy_audio_files(results: list[dict], output_dir: Path) -> int:
    """Copy matched audio files into output_dir/audio/.

    Tries the precomputed segment WAV first. If missing, falls back to cutting
    the segment from the source audio file using start_seconds/end_seconds.
    Skips silently if neither is available.

    Returns:
        Number of files successfully written.
    """
    import soundfile as sf

    audio_dir = Path(output_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for entry in results:
        seg_id = entry.get("segment_id", "segment")
        dest = audio_dir / f"{seg_id}.wav"

        # 1. Try precomputed segment WAV
        seg_path = entry.get("audio_path")
        if seg_path and Path(seg_path).exists():
            shutil.copy2(seg_path, dest)
            copied += 1
            continue

        # 2. Fall back: cut from source audio at start/end timestamps
        src = entry.get("source_audio_file")
        start = entry.get("start_seconds")
        end = entry.get("end_seconds")
        if src and start is not None and end is not None and Path(src).exists():
            try:
                data, sr = sf.read(str(src))
                s = int(start * sr)
                e = int(end * sr)
                sf.write(str(dest), data[s:e], sr)
                copied += 1
                continue
            except Exception as exc:
                print(f"  Warning: could not cut {seg_id} from source: {exc}")

        skipped += 1

    if skipped:
        print(f"  Warning: {skipped} audio file(s) not found in local cache — skipped")

    return copied


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve corpus segments matching a chromatic color target."
    )
    parser.add_argument(
        "--color",
        required=True,
        help=f"Target color ({', '.join(VALID_COLORS)})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of results to return (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        default="./sample_retrieval",
        help="Output directory for sample_map.yml (default: ./sample_retrieval)",
    )
    parser.add_argument(
        "--parquet",
        default=None,
        help="Override path to training_data_clap_embeddings.parquet",
    )
    parser.add_argument(
        "--copy-audio",
        action="store_true",
        help="Copy matched audio files to <output-dir>/audio/ if available",
    )

    args = parser.parse_args()

    if args.color not in VALID_COLORS:
        print(
            f"ERROR: Unknown color '{args.color}'. Valid: {', '.join(VALID_COLORS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)

    print(f"Color:      {args.color}")
    print(f"Top-N:      {args.top_n}")
    print(f"Output:     {output_dir}")
    print()

    print("Loading CLAP index...")
    df = load_clap_index(parquet_path=args.parquet)
    total = len(df)
    color_count = (df["color"] == args.color).sum()
    print(f"  {total} segments total, {color_count} labeled {args.color}")

    print(f"Scoring and ranking top {args.top_n} {args.color} segments...")
    results = retrieve_by_color(df, args.color, top_n=args.top_n)

    if not results:
        print(f"No segments found for color '{args.color}'.")
        sys.exit(0)

    # Print ranked table
    print()
    print(f"{'Rank':<5} {'Segment ID':<28} {'Song':<35} {'Match'}")
    print("-" * 75)
    for r in results:
        print(
            f"  #{r['rank']:<3} {r['segment_id']:<28} "
            f"{r['song_slug'][:33]:<35} {r['match']:.4f}"
        )

    # Write sample_map.yml
    out_path = write_sample_map(results, output_dir, args.color)
    print(f"\nWritten: {out_path}")

    # Optional audio copy
    if args.copy_audio:
        print("Copying audio files...")
        n_copied = copy_audio_files(results, output_dir)
        print(f"  Copied {n_copied} file(s) to {output_dir / 'audio'}/")


if __name__ == "__main__":
    main()
