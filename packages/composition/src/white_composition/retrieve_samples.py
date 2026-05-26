"""Chromatic sample retrieval from the precomputed CLAP embedding index.

Usage::
    df = load_clap_index(clap_parquet_path, meta_parquet_path)
    results = retrieve_by_color(df, "Red", top_n=20)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_clap_index(parquet_path: str, meta_parquet_path: str) -> pd.DataFrame:
    """Load and join CLAP embeddings with segment metadata.

    Args:
        parquet_path: Path to training_data_clap_embeddings.parquet
            (columns: segment_id, audio_embedding, has_audio_embedding)
        meta_parquet_path: Path to a metadata parquet with at least
            (segment_id, rainbow_color, source_audio_file, song_id)

    Returns:
        DataFrame with columns: segment_id, audio_embedding_arr (np.ndarray),
        rainbow_color, source_audio_file, song_slug.
    """
    clap = pd.read_parquet(parquet_path)
    meta = pd.read_parquet(
        meta_parquet_path,
        columns=["segment_id", "rainbow_color", "source_audio_file", "song_id"],
    )

    df = clap[clap["has_audio_embedding"]].merge(meta, on="segment_id", how="inner")
    df = df.dropna(subset=["rainbow_color", "source_audio_file"])

    df["audio_embedding_arr"] = df["audio_embedding"].apply(
        lambda v: np.asarray(v, dtype=np.float32)
    )
    df["song_slug"] = df["song_id"]
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
        top_n: Maximum number of results to return.

    Returns:
        List of dicts with keys: segment_id, source_audio_file, match,
        song_slug, color.
    """
    mask = df["rainbow_color"].str.lower() == color.lower()
    subset = df[mask]
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
        results.append(
            {
                "segment_id": row["segment_id"],
                "source_audio_file": row["source_audio_file"],
                "match": float(round(score_vals[i], 4)),
                "song_slug": row["song_slug"],
                "color": row["rainbow_color"],
            }
        )
    return results
