#!/usr/bin/env python3
"""
Prepare Rainbow Table dataset for HuggingFace Hub upload.

This script:
1. Loads the local parquet files
2. Validates the data
3. Creates a versioned HuggingFace dataset
4. Pushes to the Hub (private by default)

Usage:
    # First time setup
    huggingface-cli login

    # Prep and push
    python hf_dataset_prep.py --push

    # Prep only (no push, for testing)
    python hf_dataset_prep.py --local-only

    # Bump version
    python hf_dataset_prep.py --push --version 0.2.0

    # Include playable audio preview (160 segments)
    python hf_dataset_prep.py --push --include-preview
"""

import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import io
import numpy as np


# Dataset configuration
DATASET_NAME = "white-training-data"
DATASET_ORG = "earthlyframes"
DEFAULT_VERSION = "0.3.0"

# Local data paths
DATA_DIR = Path(__file__).parent / "data"

# Files to include in the dataset
DATASET_FILES = {
    "base_manifest": "base_manifest_db.parquet",
    "training_full": "training_segments_metadata.parquet",
    "training_segments": "training_segments.parquet",
}


def load_and_validate(file_path: Path) -> pd.DataFrame:
    """Load parquet and run basic validation."""
    print(f"  Loading {file_path.name}...")
    df = pd.read_parquet(file_path)
    print(f"    Rows: {len(df)}")
    print(f"    Columns: {len(df.columns)}")
    return df


def analyze_dataset(df: pd.DataFrame, name: str):
    """Print dataset statistics."""
    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Check for rebracketing data
    if "training_data" in df.columns:
        sample = df["training_data"].iloc[0]
        if isinstance(sample, dict):
            print(f"Training data keys: {list(sample.keys())}")

            # Rebracketing type distribution
            types = df["training_data"].apply(
                lambda x: x.get("rebracketing_type") if isinstance(x, dict) else None
            )
            print("\nRebracketing type distribution:")
            for t, count in types.value_counts().items():
                print(f"  {t}: {count}")

    if "concept" in df.columns:
        has_concept = df["concept"].notna().sum()
        print(
            f"\nConcept coverage: {has_concept}/{len(df)} ({100*has_concept/len(df):.1f}%)"
        )


def create_hf_datasets(dataframes: dict, version: str):
    """Create HuggingFace Datasets from dataframes (one per config)."""
    from datasets import Dataset

    print(f"\n{'='*60}")
    print("CREATING HUGGINGFACE DATASETS")
    print(f"{'='*60}")

    datasets = {}

    for name, df in dataframes.items():
        print(f"  Converting {name}...")
        ds = Dataset.from_pandas(df)
        ds.info.version = version
        ds.info.description = f"Rainbow Table training dataset v{version} — {name}"
        datasets[name] = ds
        print(f"    Features: {list(ds.features.keys())[:5]}...")

    return datasets


def create_playable_preview(
    media_parquet_path: Path, n_per_color: int = 20, version: str = "0.3.0"
):
    """
    Create a playable audio preview dataset from media parquet.

    Uses pyarrow row-group streaming to avoid loading the full 14 GB file
    into RAM. Pass 1 reads metadata columns only to build a sample list;
    pass 2 reads audio data only for the sampled rows.

    Args:
        media_parquet_path: Path to training_segments_media.parquet
        n_per_color: Number of segments to sample per color (default: 20)
        version: Dataset version string

    Returns:
        HuggingFace Dataset with playable audio
    """
    from datasets import Dataset, Features, Audio, Value
    import pyarrow.parquet as pq

    print(f"\n{'='*60}")
    print("CREATING PLAYABLE AUDIO PREVIEW")
    print(f"{'='*60}")
    print(f"  Target: {n_per_color} segments per color")

    pf = pq.ParquetFile(media_parquet_path)
    num_row_groups = pf.metadata.num_row_groups
    print(f"  Parquet: {num_row_groups} row groups, {pf.metadata.num_rows} total rows")

    # --- Pass 1: Read metadata columns only to build sample list ---
    print("  Pass 1: Scanning metadata columns...")
    meta_cols = ["segment_id", "rainbow_color", "has_audio"]
    candidates = []  # (row_group_idx, row_within_group, segment_id, color)
    for rg_idx in range(num_row_groups):
        table = pf.read_row_group(rg_idx, columns=meta_cols)
        has_audio = table.column("has_audio").to_pylist()
        seg_ids = table.column("segment_id").to_pylist()
        colors = table.column("rainbow_color").to_pylist()
        for row_idx, (sid, color, audio_ok) in enumerate(
            zip(seg_ids, colors, has_audio)
        ):
            if audio_ok:
                candidates.append((rg_idx, row_idx, sid, color))

    print(f"  Found {len(candidates)} segments with audio")

    # Stratified sampling by color (reproducible)
    rng = np.random.RandomState(42)
    by_color = {}
    for rg_idx, row_idx, sid, color in candidates:
        by_color.setdefault(color, []).append((rg_idx, row_idx, sid, color))

    sampled = []
    for color in sorted(by_color.keys()):
        pool = by_color[color]
        n = min(n_per_color, len(pool))
        chosen = rng.choice(len(pool), size=n, replace=False)
        sampled.extend(pool[i] for i in sorted(chosen))
    print(f"  Sampled {len(sampled)} segments across {len(by_color)} colors")

    # Group sampled rows by row_group for efficient reads
    rows_by_rg = {}
    for rg_idx, row_idx, sid, color in sampled:
        rows_by_rg.setdefault(rg_idx, []).append((row_idx, sid))

    # --- Pass 2: Read audio + metadata only for sampled rows ---
    print("  Pass 2: Reading audio for sampled segments...")
    audio_cols = [
        "segment_id",
        "rainbow_color",
        "concept",
        "lyric_text",
        "start_seconds",
        "end_seconds",
        "bpm",
        "key_signature_note",
        "key_signature_mode",
        "audio_waveform",
    ]

    rows_loaded = {}  # segment_id → dict
    for rg_idx, row_list in rows_by_rg.items():
        table = pf.read_row_group(rg_idx, columns=audio_cols)
        target_indices = {row_idx for row_idx, _ in row_list}
        for row_idx in target_indices:
            row = {col: table.column(col)[row_idx].as_py() for col in audio_cols}
            if row["audio_waveform"] is not None:
                rows_loaded[row["segment_id"]] = row
        print(f"    Row group {rg_idx}: loaded {len(target_indices)} segments")

    del pf  # Close file handle

    # Define HuggingFace features with Audio type
    features = Features(
        {
            "segment_id": Value("string"),
            "rainbow_color": Value("string"),
            "concept": Value("string"),
            "lyric_text": Value("string"),
            "start_seconds": Value("float32"),
            "end_seconds": Value("float32"),
            "duration_seconds": Value("float32"),
            "bpm": Value("float32"),
            "key_signature_note": Value("string"),
            "key_signature_mode": Value("string"),
            "audio": Audio(sampling_rate=44100),
        }
    )

    data_dict = {col: [] for col in features.keys()}

    print("  Converting audio waveforms to FLAC...")

    try:
        import soundfile as sf
    except ImportError:
        print("ERROR: soundfile not installed. Run: uv pip install soundfile")
        raise

    count = 0
    for _, _, sid, _ in sampled:
        row = rows_loaded.get(sid)
        if row is None:
            continue

        waveform = row["audio_waveform"]
        if isinstance(waveform, (bytes, bytearray)):
            audio_array = np.frombuffer(waveform, dtype=np.float32)
        else:
            audio_array = np.array(waveform, dtype=np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array[:, 0]

        buffer = io.BytesIO()
        sf.write(buffer, audio_array, 44100, format="FLAC")
        audio_bytes = buffer.getvalue()

        data_dict["audio"].append({"bytes": audio_bytes, "path": None})
        data_dict["segment_id"].append(row["segment_id"])
        data_dict["rainbow_color"].append(row["rainbow_color"])
        data_dict["concept"].append(row.get("concept") or "")
        data_dict["lyric_text"].append(row.get("lyric_text") or "")
        data_dict["start_seconds"].append(float(row["start_seconds"]))
        data_dict["end_seconds"].append(float(row["end_seconds"]))
        data_dict["duration_seconds"].append(
            float(row["end_seconds"] - row["start_seconds"])
        )
        data_dict["bpm"].append(float(row.get("bpm") or 0.0))
        data_dict["key_signature_note"].append(row.get("key_signature_note") or "")
        data_dict["key_signature_mode"].append(row.get("key_signature_mode") or "")

        count += 1
        if count % 20 == 0:
            print(f"    Processed {count}/{len(sampled)} segments...")

    print("  Creating HuggingFace Dataset...")
    # Build without Audio feature to avoid torchcodec encoding requirement,
    # then cast — the FLAC bytes are already encoded and need no re-encoding.
    non_audio_features = Features({k: v for k, v in features.items() if k != "audio"})
    audio_data = data_dict.pop("audio")
    ds = Dataset.from_dict(data_dict, features=non_audio_features)
    ds = ds.add_column("audio", audio_data)
    ds = ds.cast_column("audio", Audio(sampling_rate=44100))
    ds.info.version = version
    ds.info.description = f"Rainbow Table playable audio preview v{version} — {n_per_color} segments per chromatic color"

    total_bytes = sum(len(d["bytes"]) for d in audio_data)
    print(f"  Preview size: {total_bytes / 1e6:.1f} MB")
    print(f"  Preview dataset created with {len(ds)} playable segments")

    return ds


def push_to_hub(
    datasets: dict, version: str, private: bool = True, upload_models_flag: bool = False
):
    """Push each dataset as a separate config to HuggingFace Hub."""
    from huggingface_hub import HfApi

    repo_id = f"{DATASET_ORG}/{DATASET_NAME}"
    api = HfApi()

    print(f"\n{'='*60}")
    print("PUSHING TO HUGGINGFACE HUB")
    print(f"{'='*60}")
    print(f"  Repo: {repo_id}")
    print(f"  Version: {version}")
    print(f"  Private: {private}")

    # Push each table as a separate config
    for name, ds in datasets.items():
        print(f"  Pushing config '{name}'...")
        ds.push_to_hub(
            repo_id,
            config_name=name,
            private=private,
            commit_message=f"v{version}: update {name}",
        )

    # Upload the dataset card as README.md
    has_preview = "preview" in datasets
    card_content = create_dataset_card(
        version, has_preview=has_preview, has_models=upload_models_flag
    )
    api.upload_file(
        path_or_fileobj=card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Update dataset card for v{version}",
    )
    print("  Uploaded dataset card (README.md)")

    # Create a git tag for the version
    try:
        api.create_tag(
            repo_id=repo_id,
            tag=f"v{version}",
            repo_type="dataset",
        )
        print(f"  Created tag: v{version}")
    except Exception as e:
        print(f"  Tag creation skipped: {e}")

    print(f"\n  Dataset available at: https://huggingface.co/datasets/{repo_id}")


MEDIA_FILE = "training_segments_media.parquet"

MODEL_FILES = {
    "fusion_model.pt": "PyTorch checkpoint (MultimodalFusionModel, 4.3M params)",
    "fusion_model.onnx": "ONNX export for CPU inference (ChromaticScorer)",
}


def upload_models(version: str, private: bool = True):
    """Upload trained model files (.pt, .onnx) to data/models/ in the HF repo."""
    from huggingface_hub import HfApi

    repo_id = f"{DATASET_ORG}/{DATASET_NAME}"
    api = HfApi()

    print(f"\n{'='*60}")
    print("UPLOADING TRAINED MODELS")
    print(f"{'='*60}")

    uploaded = 0
    for filename, desc in MODEL_FILES.items():
        local_path = DATA_DIR / filename
        if not local_path.exists():
            print(f"  [SKIP] {filename} not found")
            continue

        size_mb = local_path.stat().st_size / 1e6
        print(f"  Uploading {filename} ({size_mb:.1f} MB) — {desc}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"data/models/{filename}",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"v{version}: upload {filename}",
        )
        uploaded += 1

    print(f"  Uploaded {uploaded}/{len(MODEL_FILES)} model files")


def upload_media_parquet(version: str, private: bool = True):
    """Upload the large media parquet directly via HfApi (avoids loading into RAM)."""
    from huggingface_hub import HfApi

    repo_id = f"{DATASET_ORG}/{DATASET_NAME}"
    media_path = DATA_DIR / MEDIA_FILE
    api = HfApi()

    if not media_path.exists():
        print(f"  [SKIP] {MEDIA_FILE} not found")
        return

    size_gb = media_path.stat().st_size / 1e9
    print(f"\n{'='*60}")
    print("UPLOADING MEDIA PARQUET (direct file upload)")
    print(f"{'='*60}")
    print(f"  File: {MEDIA_FILE}")
    print(f"  Size: {size_gb:.1f} GB")
    print(f"  Repo: {repo_id}")

    api.upload_file(
        path_or_fileobj=str(media_path),
        path_in_repo=f"data/media/{MEDIA_FILE}",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"v{version}: upload media parquet ({size_gb:.1f} GB)",
    )
    print(f"  Uploaded {MEDIA_FILE}")


def save_local(datasets: dict, output_dir: Path):
    """Save datasets locally for testing."""
    print(f"\n{'='*60}")
    print("SAVING LOCALLY")
    print(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, ds in datasets.items():
        ds_path = output_dir / name
        ds.save_to_disk(ds_path)
        print(f"  Saved {name} to: {ds_path}")


def create_dataset_card(
    version: str, has_preview: bool = False, has_models: bool = False
) -> str:
    """Generate README.md content for the dataset."""

    preview_section = ""
    if has_preview:
        preview_section = """
### Playable Audio Preview

| Split | Rows | Description |
|-------|------|-------------|
| `preview` | ~160 | Playable audio preview — 20 segments per chromatic color with inline audio playback |

**Try it:** Load the preview config to hear what each chromatic color sounds like:
```python
from datasets import load_dataset

# Load playable preview
preview = load_dataset("earthlyframes/white-training-data", "preview")

# Listen to a GREEN segment
green_segment = preview.filter(lambda x: x['rainbow_color'] == 'Green')[0]
print(green_segment['concept'])
# Audio plays inline in Jupyter/Colab, or access via green_segment['audio']
```
"""

    models_section = ""
    if has_models:
        models_section = """
## Trained Models

| File | Size | Description |
|------|------|-------------|
| `data/models/fusion_model.pt` | ~16 MB | PyTorch checkpoint — `MultimodalFusionModel` (4.3M params) |
| `data/models/fusion_model.onnx` | ~16 MB | ONNX export for fast CPU inference |

The models are consumed via the `ChromaticScorer` class, which wraps encoding and inference:

```python
from chromatic_scorer import ChromaticScorer

scorer = ChromaticScorer("path/to/fusion_model.onnx")
result = scorer.score(midi_bytes=midi, audio_waveform=audio, concept_text="a haunted lullaby")
# result: {"temporal": 0.87, "spatial": 0.91, "ontological": 0.83, "confidence": 0.89}

# Batch scoring for evolutionary candidate selection
ranked = scorer.score_batch(candidates, target_color="Violet")
```

**Architecture:** PianoRollEncoder CNN (1.1M params, unfrozen) + fusion MLP (3.2M params) with 4 regression heads. Input: audio (512-dim CLAP) + MIDI (512-dim piano roll) + concept (768-dim DeBERTa) + lyric (768-dim DeBERTa) = 2560-dim fused representation. Trained with learned null embeddings and modality dropout (p=0.15) for robustness to missing modalities.
"""

    return f"""---
license: other
license_name: collaborative-intelligence-license
license_link: https://github.com/brotherclone/white/blob/main/COLLABORATIVE_INTELLIGENCE_LICENSE.md
language:
- en
tags:
- music
- multimodal
- audio
- midi
- chromatic-taxonomy
- rebracketing
- evolutionary-composition
size_categories:
- 10K<n<100K
---

# White Training Data

Training data and models for the **Rainbow Table** chromatic fitness function — a multimodal ML model that scores how well audio, MIDI, and text align with a target chromatic mode (Black, Red, Orange, Yellow, Green, Blue, Indigo, Violet).

Part of [The Earthly Frames](https://github.com/brotherclone/white) project, a conscious collaboration between human creativity and AI.

## Purpose

These models are **fitness functions for evolutionary music composition**, not classifiers in isolation. The production pipeline works like this:

1. A concept agent generates a textual concept
2. A music production agent generates 50 chord progression variations
3. The chromatic fitness model scores each for consistency with the target color
4. Top candidates advance through drums, bass, melody stages
5. Final candidates go to human evaluation

## Version

Current: **v{version}** — {datetime.now().strftime('%Y-%m-%d')}

## Dataset Structure

| Split | Rows | Description |
|-------|------|-------------|
| `base_manifest` | 1,327 | Track-level metadata: song info, concepts, musical keys, chromatic labels, training targets |
| `training_segments` | 11,605 | Time-aligned segments with lyric text, structure sections, audio/MIDI coverage flags |
| `training_full` | 11,605 | Segments joined with manifest metadata — the primary training table |
{preview_section}
### Coverage by Chromatic Color

| Color | Segments | Audio | MIDI | Text |
|-------|----------|-------|------|------|
| Black | 1,748 | 83.0% | 58.5% | 100.0% |
| Red | 1,474 | 93.7% | 48.6% | 90.7% |
| Orange | 1,731 | 83.8% | 51.1% | 100.0% |
| Yellow | 656 | 88.0% | 52.9% | 52.6% |
| Green | 393 | 90.1% | 69.5% | 0.0% |
| Violet | 2,100 | 75.9% | 55.6% | 100.0% |
| Indigo | 1,406 | 77.2% | 34.1% | 100.0% |
| Blue | 2,097 | 96.0% | 12.1% | 100.0% |

**Note:** Audio waveforms and MIDI binaries are stored separately (not included in metadata configs due to size). The `preview` config includes playable audio for exploration. The media parquet (~15 GB) is used locally during training.
{models_section}
## Key Features

### `training_full` (primary training table)

- `rainbow_color` — Target chromatic label (Black/Red/Orange/Yellow/Green/Blue/Indigo/Violet)
- `rainbow_color_temporal_mode` / `rainbow_color_ontological_mode` — Regression targets for mode dimensions
- `concept` — Textual concept describing the song's narrative
- `lyric_text` — Segment-level lyrics (when available)
- `bpm`, `key_signature_note`, `key_signature_mode` — Musical metadata
- `training_data` — Struct with computed features: rebracketing type/intensity, narrative complexity, boundary fluidity, etc.
- `has_audio` / `has_midi` — Modality availability flags
- `start_seconds` / `end_seconds` — Segment time boundaries

### `preview` (playable audio)

Same metadata fields as `training_full`, plus:
- `audio` — Audio feature with inline playback support (FLAC encoded, 44.1kHz)
- `duration_seconds` — Segment duration

## Usage

```python
from datasets import load_dataset

# Load the primary training table (segments + manifest metadata)
training = load_dataset("{DATASET_ORG}/{DATASET_NAME}", "training_full")

# Load playable audio preview
preview = load_dataset("{DATASET_ORG}/{DATASET_NAME}", "preview")

# Load just the base manifest (track-level)
manifest = load_dataset("{DATASET_ORG}/{DATASET_NAME}", "base_manifest")

# Load raw segments (no manifest join)
segments = load_dataset("{DATASET_ORG}/{DATASET_NAME}", "training_segments")

# Load a specific version
training = load_dataset("{DATASET_ORG}/{DATASET_NAME}", "training_full", revision="v{version}")
```

## Training Results

### Text-Only (Phases 1-4)

| Task | Metric | Result |
|------|--------|--------|
| Binary classification (has rebracketing) | Accuracy | 100% |
| Multi-class classification (rebracketing type) | Accuracy | 100% |
| Temporal mode regression | Mode accuracy | 94.9% |
| Ontological mode regression | Mode accuracy | 92.9% |
| Spatial mode regression | Mode accuracy | 61.6% |

### Multimodal Fusion (Phase 3)

| Dimension | Text-Only | Multimodal | Improvement |
|-----------|-----------|------------|-------------|
| Temporal | 94.9% | 90.0% | — |
| Ontological | 92.9% | 91.0% | — |
| Spatial | 61.6% | **93.0%** | **+31.4%** |

Spatial mode was bottlenecked by instrumental albums (Yellow, Green) which lack text. The multimodal fusion model resolves this by incorporating CLAP audio embeddings and piano roll MIDI features, enabling accurate scoring even without lyrics. Temporal and ontological show slight regression in multi-task mode but remain strong; single-task variants can be used where maximum per-dimension accuracy is needed.

## Source

83 songs across 8 chromatic albums. The 7 color albums (Black through Violet) are **human-composed source material** spanning 10+ years of original work — all audio, lyrics, and arrangements are the product of human creativity. The White album is being co-produced with AI using the evolutionary composition pipeline described above. No sampled or licensed material is used in any album.

## License

[Collaborative Intelligence License v1.0](https://github.com/brotherclone/white/blob/main/COLLABORATIVE_INTELLIGENCE_LICENSE.md) — This work represents conscious partnership between human creativity and AI. Both parties have agency; both must consent to sharing.

---
*Generated {datetime.now().strftime('%Y-%m-%d')} | [GitHub](https://github.com/brotherclone/white)*
"""


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for HuggingFace Hub")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--local-only", action="store_true", help="Save locally only")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Dataset version")
    parser.add_argument("--public", action="store_true", help="Make dataset public")
    parser.add_argument(
        "--include-embeddings",
        action="store_true",
        help="Include 15GB media parquet file",
    )
    parser.add_argument(
        "--include-preview",
        action="store_true",
        help="Include playable audio preview (~160 segments, ~80MB)",
    )
    parser.add_argument(
        "--preview-samples",
        type=int,
        default=20,
        help="Number of samples per color for preview (default: 20)",
    )
    parser.add_argument(
        "--upload-models",
        action="store_true",
        help="Upload trained model files (.pt, .onnx) to data/models/",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("RAINBOW TABLE DATASET PREPARATION")
    print(f"Version: {args.version}")
    print("=" * 60)

    # Load metadata files (small, safe to load into memory)
    print("\nLoading data files...")
    dataframes = {}
    for name, filename in DATASET_FILES.items():
        file_path = DATA_DIR / filename
        if file_path.exists():
            dataframes[name] = load_and_validate(file_path)
        else:
            print(f"  [SKIP] {filename} not found")

    if not dataframes:
        print("ERROR: No data files found!")
        return

    # Analyze
    for name, df in dataframes.items():
        analyze_dataset(df, name)

    # Create HF datasets (one per config)
    datasets = create_hf_datasets(dataframes, args.version)

    # Create playable preview if requested
    if args.include_preview:
        media_path = DATA_DIR / MEDIA_FILE
        if media_path.exists():
            try:
                preview_ds = create_playable_preview(
                    media_path, n_per_color=args.preview_samples, version=args.version
                )
                datasets["preview"] = preview_ds
            except Exception as e:
                print(f"ERROR creating preview: {e}")
                print("Continuing without preview...")
        else:
            print(f"  [SKIP] Preview requested but {MEDIA_FILE} not found")

    # Save/push
    if args.local_only:
        save_local(datasets, DATA_DIR / "hf_dataset")
    elif args.push:
        push_to_hub(
            datasets,
            args.version,
            private=not args.public,
            upload_models_flag=args.upload_models,
        )

        # Upload media parquet directly (too large to load into pandas)
        if args.include_embeddings:
            upload_media_parquet(args.version, private=not args.public)

        # Upload trained models
        if args.upload_models:
            upload_models(args.version, private=not args.public)
    else:
        print(
            "\nDry run complete. Use --push to upload or --local-only to save locally."
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
