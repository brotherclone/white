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
DATASET_NAME = "rainbow-table-training"
DATASET_ORG = "brotherclone"  # Your HF username or org
DEFAULT_VERSION = "0.1.0"

# Local data paths
DATA_DIR = Path(__file__).parent / "data"

# Files to include in the dataset
DATASET_FILES = {
    "base_manifest": "base_manifest_db.parquet",
    "training_full": "training_data_full.parquet",
    "training_segments": "training_segments.parquet",
    # "embeddings": "training_data_embedded.parquet",  # Add for Phase 3
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


def create_hf_dataset(dataframes: dict, version: str):
    """Create a HuggingFace Dataset from dataframes."""
    from datasets import Dataset, DatasetDict

    print(f"\n{'='*60}")
    print("CREATING HUGGINGFACE DATASET")
    print(f"{'='*60}")

    dataset_dict = {}

    for name, df in dataframes.items():
        print(f"  Converting {name}...")

        # Handle dict columns (like training_data) - flatten or keep as-is
        # HF datasets can handle dicts but let's be explicit
        dataset_dict[name] = Dataset.from_pandas(df)
        print(f"    Features: {list(dataset_dict[name].features.keys())[:5]}...")

    ds = DatasetDict(dataset_dict)

    # Add metadata
    ds.info.version = version
    ds.info.description = f"Rainbow Table training dataset v{version}"

    return ds


def push_to_hub(dataset, version: str, private: bool = True):
    """Push dataset to HuggingFace Hub."""
    repo_id = f"{DATASET_ORG}/{DATASET_NAME}"

    print(f"\n{'='*60}")
    print("PUSHING TO HUGGINGFACE HUB")
    print(f"{'='*60}")
    print(f"  Repo: {repo_id}")
    print(f"  Version: {version}")
    print(f"  Private: {private}")

    dataset.push_to_hub(
        repo_id,
        private=private,
        commit_message=f"Dataset version {version}",
    )

    # Create a git tag for the version
    from huggingface_hub import HfApi

    api = HfApi()

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


def save_local(dataset, output_dir: Path):
    """Save dataset locally for testing."""
    print(f"\n{'='*60}")
    print("SAVING LOCALLY")
    print(f"{'='*60}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_dir)
    print(f"  Saved to: {output_dir}")


def create_dataset_card(version: str) -> str:
    """Generate README.md content for the dataset."""
    return f"""---
license: cc-by-nc-4.0
language:
- en
tags:
- music
- rebracketing
- rainbow-table
- multimodal
size_categories:
- 1K<n<10K
---

# Rainbow Table Training Dataset

Training data for the Rainbow Pipeline - multiclass rebracketing classification and beyond.

## Version

Current version: **{version}**

## Dataset Structure

- **base_manifest**: Core track metadata with concepts and training targets ({DATASET_FILES['base_manifest']})
- **training_full**: Full training data with computed features
- **training_segments**: Temporal segments for sequence modeling

## Usage

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("{DATASET_ORG}/{DATASET_NAME}")

# Access splits
manifest = dataset["base_manifest"]
segments = dataset["training_segments"]

# Load specific version
dataset = load_dataset("{DATASET_ORG}/{DATASET_NAME}", revision="v{version}")
```

## Training Targets

### Phase 2: Multiclass Classification
- **Target column**: `training_data.rebracketing_type`
- **Classes**: spatial, temporal, causal, perceptual, memory, ontological, narrative, identity

### Rebracketing Type Distribution
| Type | Count |
|------|-------|
| spatial | ~803 |
| perceptual | ~53 |
| causal | ~31 |
| temporal | ~5 |

## Features

Key columns in `base_manifest`:
- `concept`: Text description for classification
- `training_data`: Dict with rebracketing labels and computed features
- `rainbow_color`: Chromatic mode (BLACK, RED, ORANGE, etc.)
- `lyrics`, `lrc_lyrics`: Song text content

## License

CC-BY-NC-4.0 (Non-commercial use)

## Citation

Part of the White Album Project - Rainbow Table corpus.

---
*Dataset version {version} - Generated {datetime.now().strftime('%Y-%m-%d')}*
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

    # Determine which files to include
    files_to_load = DATASET_FILES.copy()
    if args.include_embeddings:
        files_to_load["embeddings"] = "training_data_embedded.parquet"

    # Load all files
    print("\nLoading data files...")
    dataframes = {}
    for name, filename in files_to_load.items():
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

    # Create HF dataset
    dataset = create_hf_dataset(dataframes, args.version)

    # Save/push
    if args.local_only:
        save_local(dataset, DATA_DIR / "hf_dataset")
    elif args.push:
        push_to_hub(dataset, args.version, private=not args.public)

        # Also save the dataset card
        card_content = create_dataset_card(args.version)
        print("\nDataset card (copy to repo README.md if needed):")
        print("-" * 40)
        print(card_content[:500] + "...")
    else:
        print(
            "\nDry run complete. Use --push to upload or --local-only to save locally."
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
