#!/usr/bin/env python3
"""
Build sounds-like embeddings for Phase 5 training.

Reads the DeBERTa parquet (which has sounds_like strings and segment_id),
looks up each artist in the artist catalog, embeds descriptions via DeBERTa,
mean-pools across artists per song, and broadcasts to all segments.

Output parquet schema:
    segment_id       str
    song_slug        str
    sounds_like_raw  str
    artists_found    int
    artists_total    int
    has_sounds_like  bool
    sounds_like_emb  list[float32] (768-dim)

Usage:
    python training/build_sounds_like_embeddings.py
    python training/build_sounds_like_embeddings.py \\
        --training-parquet training/data/training_data_with_embeddings.parquet \\
        --catalog app/reference/music/artist_catalog.yml \\
        --output training/data/sounds_like_embeddings.parquet
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_sounds_like(raw: str) -> list[str]:
    """Strip ', discogs_id: <digits>' fragments and return clean artist name list.

    >>> parse_sounds_like("David Bowie, discogs_id: 10263, Broadcast, discogs_id: 955")
    ['David Bowie', 'Broadcast']
    >>> parse_sounds_like("")
    []
    >>> parse_sounds_like("The Beatles")
    ['The Beatles']
    """
    if not raw or not raw.strip():
        return []
    # Remove all ", discogs_id: <digits>" fragments (including leading comma/space)
    cleaned = re.sub(r",?\s*discogs_id:\s*\d+", "", raw)
    # Split by comma, strip whitespace, drop empties
    artists = [a.strip() for a in cleaned.split(",") if a.strip()]
    return artists


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------


def load_catalog(catalog_path: str | Path) -> dict[str, dict]:
    """Read artist_catalog.yml, return {artist_name: entry}.

    Prefers 'reviewed' entries over 'draft' when both would somehow exist
    under the same key (defensive). Returns empty dict if file is empty or
    has no artist entries.
    """
    catalog_path = Path(catalog_path)
    if not catalog_path.exists():
        return {}

    with catalog_path.open() as f:
        data = yaml.safe_load(f)

    if not data or not isinstance(data, dict):
        return {}

    catalog = {}
    for name, entry in data.items():
        if not isinstance(entry, dict):
            continue
        # Only include entries with a description
        if not entry.get("description"):
            continue
        status = entry.get("status", "draft")
        if name not in catalog:
            catalog[name] = entry
        elif status == "reviewed" and catalog[name].get("status") != "reviewed":
            # Prefer reviewed
            catalog[name] = entry

    return catalog


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def embed_descriptions(
    artist_names: list[str],
    catalog: dict[str, dict],
    deberta_tokenizer,
    deberta_model,
) -> tuple[np.ndarray, int, int]:
    """Embed found artist descriptions, mean-pool into one 768-dim vector.

    Args:
        artist_names: List of clean artist names for one song.
        catalog: {artist_name: catalog_entry} dict.
        deberta_tokenizer: Loaded DeBERTa tokenizer.
        deberta_model: Loaded DeBERTa model (eval mode).

    Returns:
        (vector_768, found_count, total_count)
        vector_768 is zero-filled when found_count == 0.
    """
    import torch

    total = len(artist_names)
    embeddings = []

    for name in artist_names:
        entry = catalog.get(name)
        if entry is None:
            continue
        description = entry.get("description", "").strip()
        if not description:
            continue

        tokens = deberta_tokenizer(
            description,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            output = deberta_model(**tokens)

        # Mean pooling over token dimension
        attention_mask = tokens["attention_mask"].unsqueeze(-1)
        hidden = output.last_hidden_state * attention_mask
        pooled = hidden.sum(dim=1) / attention_mask.sum(dim=1)
        embeddings.append(pooled.squeeze(0).numpy().astype(np.float32))

    found = len(embeddings)
    if found == 0:
        return np.zeros(768, dtype=np.float32), 0, total

    return np.mean(embeddings, axis=0).astype(np.float32), found, total


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_sounds_like_parquet(
    training_parquet_path: str | Path,
    catalog_path: str | Path,
    output_path: str | Path,
) -> None:
    """Build sounds_like_embeddings.parquet from training data + catalog.

    Groups segments by song (using 'song_id' or 'title'), computes one
    embedding per song, then broadcasts to all segments of that song.
    """
    from transformers import AutoModel, AutoTokenizer

    training_parquet_path = Path(training_parquet_path)
    output_path = Path(output_path)

    print(f"Loading training parquet: {training_parquet_path}")
    df = pd.read_parquet(training_parquet_path)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    print(f"Loading catalog: {catalog_path}")
    catalog = load_catalog(catalog_path)
    print(f"  {len(catalog)} artists with descriptions")

    # Determine song grouping column
    song_col = "song_id" if "song_id" in df.columns else "title"
    print(f"  Song grouping column: {song_col}")

    print("Loading DeBERTa model...")
    model_name = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    print("  DeBERTa loaded")

    # Build per-song embedding (one pass, deduplicated)
    # song_id -> (sounds_like_raw, artists_found, artists_total, emb_768)
    songs_seen: dict[str, tuple] = {}
    song_to_sounds_like_raw: dict[str, str] = {}

    for _, row in df.drop_duplicates(subset=[song_col]).iterrows():
        sid = str(row[song_col])
        raw = str(row.get("sounds_like", "") or "")
        song_to_sounds_like_raw[sid] = raw

    total_songs = len(song_to_sounds_like_raw)
    songs_with_coverage = 0
    total_artists_found = 0
    total_artists_total = 0

    print(f"\nEmbedding {total_songs} songs...")
    for i, (sid, raw) in enumerate(song_to_sounds_like_raw.items(), 1):
        artists = parse_sounds_like(raw)
        emb, found, total = embed_descriptions(artists, catalog, tokenizer, model)
        songs_seen[sid] = (raw, found, total, emb)

        if found > 0:
            songs_with_coverage += 1
        total_artists_found += found
        total_artists_total += total

        if i % 10 == 0 or i == total_songs:
            print(f"  [{i}/{total_songs}] {found}/{total} artists found for '{sid}'")

    # Broadcast to all segments
    print(f"\nBroadcasting to {len(df)} segments...")
    rows = []
    for _, row in df.iterrows():
        sid = str(row[song_col])
        slug = str(row.get("song_id", row.get("title", sid)))
        raw, found, total, emb = songs_seen[sid]
        rows.append(
            {
                "segment_id": str(row["segment_id"]),
                "song_slug": slug,
                "sounds_like_raw": raw,
                "artists_found": found,
                "artists_total": total,
                "has_sounds_like": found > 0,
                "sounds_like_emb": emb.tolist(),
            }
        )

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(str(output_path), index=False)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")

    # Coverage summary
    total_segs = len(out_df)
    segs_with = out_df["has_sounds_like"].sum()
    pct_segs = 100 * segs_with / total_segs if total_segs else 0
    pct_songs = 100 * songs_with_coverage / total_songs if total_songs else 0
    artist_pct = (
        100 * total_artists_found / total_artists_total if total_artists_total else 0
    )

    print("\n--- Coverage Summary ---")
    print(
        f"Songs:    {songs_with_coverage}/{total_songs} ({pct_songs:.1f}%) with catalog coverage"
    )
    print(
        f"Segments: {segs_with}/{total_segs} ({pct_segs:.1f}%) have sounds_like embedding"
    )
    print(
        f"Artists:  {total_artists_found}/{total_artists_total} ({artist_pct:.1f}%) found in catalog"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Build sounds-like embeddings parquet")
    parser.add_argument(
        "--training-parquet",
        default="training/data/training_data_with_embeddings.parquet",
        help="Path to DeBERTa parquet (has segment_id + sounds_like columns)",
    )
    parser.add_argument(
        "--catalog",
        default="app/reference/music/artist_catalog.yml",
        help="Path to artist_catalog.yml",
    )
    parser.add_argument(
        "--output",
        default="training/data/sounds_like_embeddings.parquet",
        help="Output parquet path",
    )
    args = parser.parse_args()

    build_sounds_like_parquet(
        training_parquet_path=args.training_parquet,
        catalog_path=args.catalog,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
