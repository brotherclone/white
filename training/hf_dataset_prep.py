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
"""

import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd


# Dataset configuration
DATASET_NAME = "white-training-data"
DATASET_ORG = "earthlyframes"
DEFAULT_VERSION = "0.2.0"

# Local data paths
DATA_DIR = Path(__file__).parent / "data"

# Files to include in the dataset
DATASET_FILES = {
    "base_manifest": "base_manifest_db.parquet",
    "training_full": "training_segments_metadata.parquet",
    "training_segments": "training_segments.parquet",
    # "embeddings": "training_segments_media.parquet",  # Add for Phase 3
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


def push_to_hub(datasets: dict, version: str, private: bool = True):
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
    card_content = create_dataset_card(version)
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


def create_dataset_card(version: str) -> str:
    """Generate README.md content for the dataset."""
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

Training data for the **Rainbow Table** chromatic fitness function — a multimodal ML model that scores how well audio, MIDI, and text align with a target chromatic mode (Black, Red, Orange, Yellow, Green, Blue, Indigo, Violet).

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

### Coverage by Chromatic Color

| Color | Segments | Audio | MIDI | Text |
|-------|----------|-------|------|------|
| Black | 1,748 | 83.0% | 58.5% | 100.0% |
| Red | 1,474 | 93.7% | 48.6% | 90.7% |
| Orange | 1,731 | 83.8% | 51.1% | 100.0% |
| Yellow | 656 | 88.0% | 52.9% | 52.6% |
| Green | 393 | 90.1% | 69.5% | 0.0% |
| Blue | 2,097 | 96.0% | 12.1% | 100.0% |
| Indigo | 1,406 | 77.2% | 34.1% | 100.0% |
| Violet | 2,100 | 75.9% | 55.6% | 100.0% |

**Note:** Audio waveforms and MIDI binaries are stored separately (not included in this dataset due to size). This dataset contains the metadata, segment boundaries, lyric text, and computed training features needed for model training. The media parquet (~15 GB) is used locally during training.

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

## Usage

```python
from datasets import load_dataset

# Load the primary training table (segments + manifest metadata)
training = load_dataset("{DATASET_ORG}/{DATASET_NAME}", "training_full")

# Load just the base manifest (track-level)
manifest = load_dataset("{DATASET_ORG}/{DATASET_NAME}", "base_manifest")

# Load raw segments (no manifest join)
segments = load_dataset("{DATASET_ORG}/{DATASET_NAME}", "training_segments")

# Load a specific version
training = load_dataset("{DATASET_ORG}/{DATASET_NAME}", "training_full", revision="v{version}")
```

## Training Results (Text-Only, Phases 1-4)

| Task | Metric | Result |
|------|--------|--------|
| Binary classification (has rebracketing) | Accuracy | 100% |
| Multi-class classification (rebracketing type) | Accuracy | 100% |
| Temporal mode regression | Mode accuracy | 94.9% |
| Ontological mode regression | Mode accuracy | 92.9% |
| Spatial mode regression | Mode accuracy | 61.6% |

Spatial mode is bottlenecked by instrumental albums (Yellow, Green) which lack text. The multimodal fusion model (Phase 3, in progress) will incorporate audio and MIDI embeddings to address this.

## Source

83 songs across 8 chromatic albums, each composed as a conscious human-AI collaboration. All source audio is original — no sampled or licensed material.

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
        help="Include 69GB embeddings file (Phase 3)",
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

    # Save/push
    if args.local_only:
        save_local(datasets, DATA_DIR / "hf_dataset")
    elif args.push:
        push_to_hub(datasets, args.version, private=not args.public)

        # Upload media parquet directly (too large to load into pandas)
        if args.include_embeddings:
            upload_media_parquet(args.version, private=not args.public)
    else:
        print(
            "\nDry run complete. Use --push to upload or --local-only to save locally."
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
