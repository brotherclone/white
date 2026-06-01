"""Chromatic sample retrieval from the precomputed CLAP embedding index.

Usage::
    df = load_clap_index()
    results = retrieve_by_color(df, "Red", top_n=20)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).parents[4] / "training" / "data"
_DEFAULT_CLAP_PARQUET = _DATA_DIR / "training_data_clap_embeddings.parquet"
_DEFAULT_META_PARQUET = _DATA_DIR / "training_data_with_embeddings.parquet"

_META_COLUMNS = [
    "segment_id",
    "rainbow_color",
    "source_audio_file",
    "song_id",
    "start_seconds",
    "end_seconds",
]


def load_clap_index(
    parquet_path: str | None = None,
    meta_parquet_path: str | None = None,
) -> pd.DataFrame:
    """Load and join CLAP embeddings with segment metadata.

    Args:
        parquet_path: Path to training_data_clap_embeddings.parquet.
            Defaults to the project-local copy when None.
        meta_parquet_path: Path to a metadata parquet that includes
            segment_id, rainbow_color, source_audio_file, song_id,
            start_seconds, end_seconds.  Defaults to the project-local
            training_data_with_embeddings.parquet when None.

    Returns:
        DataFrame with columns: segment_id, audio_embedding_arr (np.ndarray),
        rainbow_color, source_audio_file, song_slug, start_seconds, end_seconds.
    """
    clap_path = (
        Path(parquet_path) if parquet_path is not None else _DEFAULT_CLAP_PARQUET
    )
    meta_path = (
        Path(meta_parquet_path)
        if meta_parquet_path is not None
        else _DEFAULT_META_PARQUET
    )

    if not clap_path.exists():
        raise FileNotFoundError(f"CLAP parquet not found: {clap_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta parquet not found: {meta_path}")

    clap = pd.read_parquet(clap_path)
    # Only request columns that actually exist in the meta parquet
    meta_all_cols = pd.read_parquet(meta_path, columns=None).columns.tolist()
    load_cols = [c for c in _META_COLUMNS if c in meta_all_cols]
    meta = pd.read_parquet(meta_path, columns=load_cols)

    df = clap[clap["has_audio_embedding"]].merge(meta, on="segment_id", how="inner")
    df = df.dropna(subset=["rainbow_color", "source_audio_file"])

    df["audio_embedding_arr"] = df["audio_embedding"].apply(
        lambda v: np.asarray(v, dtype=np.float32)
    )
    df["song_slug"] = df["song_id"]

    # Ensure start/end columns exist even if the meta parquet omits them
    for col in ("start_seconds", "end_seconds"):
        if col not in df.columns:
            df[col] = None

    return df.reset_index(drop=True)


def _cosine_sim(mat: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = mat / norms
    c_norm = np.linalg.norm(centroid)
    c_unit = centroid / c_norm if c_norm > 0 else centroid
    return normed @ c_unit


def retrieve_by_color(df: pd.DataFrame, color: str, top_n: int = 20) -> list[dict]:
    """Return the top-N segments whose CLAP audio most matches the given color.

    Matching is cosine similarity to the centroid of all embeddings for that
    color in the index.  Results are sorted descending by match score.

    Args:
        df: DataFrame returned by load_clap_index.
        color: Rainbow color string (e.g. "Red"). Case-insensitive.
        top_n: Maximum number of results to return (must be >= 1).

    Returns:
        List of dicts with keys: segment_id, source_audio_file, match,
        song_slug, color, start_seconds, end_seconds.
    """
    top_n = max(1, int(top_n))
    color_lower = color.lower()

    if color_lower == "black":
        return []

    if color_lower == "white":
        subset = df
    else:
        subset = df[df["rainbow_color"].str.lower() == color_lower]

    if subset.empty:
        return []

    mat = np.stack(subset["audio_embedding_arr"].to_numpy())
    centroid = mat.mean(axis=0)
    scores = _cosine_sim(mat, centroid)

    # Shift from [-1,1] to [0,1]
    scores_01 = (scores + 1.0) / 2.0

    top_idx = np.argsort(scores_01)[::-1][:top_n]
    rows = subset.iloc[top_idx]
    score_vals = scores_01[top_idx]

    results = []
    for i, (_, row) in enumerate(rows.iterrows()):
        start = row.get("start_seconds")
        end = row.get("end_seconds")
        results.append(
            {
                "segment_id": row["segment_id"],
                "source_audio_file": row["source_audio_file"],
                "match": float(round(score_vals[i], 4)),
                "song_slug": row["song_slug"],
                "color": row["rainbow_color"],
                "start_seconds": float(start) if start is not None else None,
                "end_seconds": float(end) if end is not None else None,
            }
        )
    return results
