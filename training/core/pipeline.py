"""
Data pipeline for Rainbow Table training.

Handles loading, preprocessing, and batching of training data.
"""

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer


class RainbowDataset(Dataset):
    """Dataset for Rainbow Table tracks."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        target_column: str,
        max_length: int = 512,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.target_column = target_column
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Get concept text
        concept = str(row["concept"])

        # Tokenize
        encoding = self.tokenizer(
            concept,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Get target from training_data dict
        training_data = row["training_data"]
        if isinstance(training_data, dict):
            target = training_data.get(self.target_column)
        else:
            # Fallback to direct column access
            target = row[self.target_column]

        # Convert to float (handle boolean and string)
        if isinstance(target, bool):
            label = float(target)
        else:
            label = float(target == "True" or target)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float32),
            "track_id": row["id"],
        }


def load_manifest(
    manifest_path: str,
    require_concept: bool = True,
    min_concept_length: int = 50,
) -> pd.DataFrame:
    """Load and filter the manifest."""

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print(f"Loading manifest from {manifest_path}")
    df = pd.read_parquet(manifest_path)
    print(f"  Loaded {len(df)} tracks")

    # Filter
    if require_concept:
        df = df[df["concept"].notna()]
        print(f"  After concept filter: {len(df)} tracks")

    if min_concept_length > 0:
        df = df[df["concept"].str.len() >= min_concept_length]
        print(f"  After length filter: {len(df)} tracks")

    return df


def build_dataloaders(
    manifest_path: str,
    tokenizer: AutoTokenizer,
    target_column: str,
    train_split: float = 0.8,
    val_split: float = 0.2,
    batch_size: int = 16,
    num_workers: int = 4,
    random_seed: int = 42,
    require_concept: bool = True,
    min_concept_length: int = 50,
    max_length: int = 512,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Build train and validation dataloaders.

    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        stats: Dictionary with dataset statistics
    """

    # Load data
    df = load_manifest(
        manifest_path,
        require_concept=require_concept,
        min_concept_length=min_concept_length,
    )

    # Check target distribution
    # Extract from training_data dict if needed
    if target_column in df.columns:
        target_values = df[target_column]
    else:
        target_values = df["training_data"].apply(
            lambda x: x.get(target_column) if isinstance(x, dict) else None
        )

    target_dist = target_values.value_counts()
    print(f"\nTarget distribution ({target_column}):")
    for val, count in target_dist.items():
        pct = 100 * count / len(df)
        print(f"  {val}: {count} ({pct:.1f}%)")

    # Create full dataset
    full_dataset = RainbowDataset(
        df=df,
        tokenizer=tokenizer,
        target_column=target_column,
        max_length=max_length,
    )

    # Split
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    print("\nDataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Compute class weights for imbalanced data
    # Extract target values properly
    if target_column in df.columns:
        target_values = df[target_column].values
    else:
        target_values = (
            df["training_data"]
            .apply(lambda x: x.get(target_column) if isinstance(x, dict) else None)
            .values
        )

    # Convert to float - FIXED: handle boolean arrays properly
    target_array = np.array(target_values)

    # Check dtype and convert appropriately
    if target_array.dtype == bool or target_array.dtype == np.bool_:
        # Direct boolean array: True → 1.0, False → 0.0
        target_float = target_array.astype(float)
    elif target_array.dtype == object:
        # Object array (might be strings or mixed types)
        # Try to convert each element
        converted = []
        for val in target_array:
            if isinstance(val, bool):
                converted.append(float(val))
            elif isinstance(val, str):
                # String "True"/"False"
                converted.append(float(val == "True"))
            else:
                # Fallback - try to interpret as truthy
                converted.append(float(bool(val)))
        target_float = np.array(converted, dtype=float)
    else:
        # Numeric or other - just cast to float
        target_float = target_array.astype(float)

    # Compute pos_weight
    num_positive = target_float.sum()
    num_negative = len(target_float) - num_positive

    if num_positive == 0:
        print("⚠️  WARNING: No positive samples! Setting pos_weight=1.0")
        pos_weight = 1.0
    elif num_negative == 0:
        print("⚠️  WARNING: No negative samples! Setting pos_weight=1.0")
        pos_weight = 1.0
    else:
        pos_weight = num_negative / num_positive

    print(f"Class balance: {num_positive:.0f} positive, {num_negative:.0f} negative")

    stats = {
        "total_samples": len(df),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "pos_weight": pos_weight,
        "target_distribution": target_dist.to_dict(),
    }

    return train_loader, val_loader, stats


if __name__ == "__main__":
    # Quick test
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    train_loader, val_loader, stats = build_dataloaders(
        manifest_path="/workspace/data/base_manifest_db.parquet",
        tokenizer=tokenizer,
        target_column="has_rebracketing_markers",
        batch_size=4,
        num_workers=0,
    )

    print("\n" + "=" * 60)
    print("TESTING DATALOADER")
    print("=" * 60)

    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Labels: {batch['labels']}")
