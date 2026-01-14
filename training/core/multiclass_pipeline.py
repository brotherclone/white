"""
Multi-class data pipeline for Rainbow Table training.

Extends the binary pipeline to support:
- Multi-class classification (single label per segment)
- Multi-label classification (multiple labels per segment)
- Label encoding and decoding
- Class weight computation
- Stratified splitting
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from collections import Counter


class LabelEncoder:
    """Encodes and decodes rebracketing type labels."""

    def __init__(self, class_mapping: Dict[str, int]):
        """
        Initialize label encoder.

        Args:
            class_mapping: Dictionary mapping label names to class indices
                          e.g., {"spatial": 0, "temporal": 1, ...}
        """
        self.class_mapping = class_mapping
        self.inverse_mapping = {v: k for k, v in class_mapping.items()}
        self.num_classes = len(class_mapping)

    def encode(
        self, label: Union[str, List[str]], multi_label: bool = False
    ) -> Union[int, np.ndarray]:
        """
        Encode label(s) to class index/indices.

        Args:
            label: String label or list of labels
            multi_label: If True, return binary vector for multi-label

        Returns:
            - For single-label: class index (int)
            - For multi-label: binary vector [num_classes]
        """
        if multi_label:
            # Multi-label: return binary vector
            vector = np.zeros(self.num_classes, dtype=np.float32)
            if isinstance(label, str):
                labels = [label]
            else:
                labels = label

            for lbl in labels:
                if lbl in self.class_mapping:
                    vector[self.class_mapping[lbl]] = 1.0

            return vector
        else:
            # Single-label: return class index
            if isinstance(label, list):
                # Take first label if multiple provided
                label = label[0] if label else None

            if label in self.class_mapping:
                return self.class_mapping[label]
            else:
                raise ValueError(f"Unknown label: {label}")

    def decode(
        self, index: Union[int, np.ndarray, torch.Tensor], multi_label: bool = False
    ) -> Union[str, List[str]]:
        """
        Decode class index/indices to label(s).

        Args:
            index: Class index or binary vector
            multi_label: If True, expect binary vector and return list of labels

        Returns:
            - For single-label: label string
            - For multi-label: list of label strings
        """
        if multi_label:
            # Multi-label: decode binary vector
            if isinstance(index, torch.Tensor):
                index = index.cpu().numpy()

            labels = []
            for i, val in enumerate(index):
                if val > 0.5:  # Threshold at 0.5
                    if i in self.inverse_mapping:
                        labels.append(self.inverse_mapping[i])

            return labels
        else:
            # Single-label: decode class index
            if isinstance(index, (np.ndarray, torch.Tensor)):
                index = int(index)

            if index in self.inverse_mapping:
                return self.inverse_mapping[index]
            else:
                raise ValueError(f"Unknown class index: {index}")

    def get_class_distribution(
        self, labels: List[Union[str, List[str]]], multi_label: bool = False
    ) -> Dict[str, int]:
        """
        Compute class distribution from labels.

        Args:
            labels: List of labels (strings or lists of strings)
            multi_label: Whether labels are multi-label

        Returns:
            Dictionary mapping class names to counts
        """
        counts = Counter()

        for label in labels:
            if multi_label:
                # Multi-label: count each label separately
                if isinstance(label, str):
                    label = [label]
                for lbl in label:
                    counts[lbl] += 1
            else:
                # Single-label: count single label
                if isinstance(label, list):
                    label = label[0] if label else None
                if label:
                    counts[label] += 1

        return dict(counts)


class MultiClassRainbowDataset(Dataset):
    """Dataset for multi-class Rainbow Table classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        label_encoder: LabelEncoder,
        target_column: str,
        multi_label: bool = False,
        max_length: int = 512,
    ):
        """
        Initialize dataset.

        Args:
            df: DataFrame with track data
            tokenizer: Tokenizer for text encoding
            label_encoder: Label encoder for class encoding
            target_column: Name of target column
            multi_label: Whether this is multi-label classification
            max_length: Maximum sequence length
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.target_column = target_column
        self.multi_label = multi_label
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

        # Get target from training_data dict or direct column
        training_data = row.get("training_data")
        if isinstance(training_data, dict) and self.target_column in training_data:
            target = training_data[self.target_column]
        else:
            target = row.get(self.target_column)

        # Encode label
        try:
            encoded = self.label_encoder.encode(target, multi_label=self.multi_label)
        except ValueError as e:
            # Handle unknown labels gracefully
            print(f"Warning: {e} for track {row.get('id')}. Using default.")
            if self.multi_label:
                encoded = np.zeros(self.label_encoder.num_classes, dtype=np.float32)
            else:
                encoded = 0

        # Convert to tensor
        if self.multi_label:
            label = torch.tensor(encoded, dtype=torch.float32)
        else:
            label = torch.tensor(encoded, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label,
            "track_id": row.get("id", idx),
        }


def build_multiclass_dataloaders(
    manifest_path: str,
    tokenizer: AutoTokenizer,
    class_mapping: Dict[str, int],
    target_column: str,
    train_split: float = 0.8,
    val_split: float = 0.2,
    batch_size: int = 16,
    num_workers: int = 4,
    random_seed: int = 42,
    require_concept: bool = True,
    min_concept_length: int = 50,
    max_length: int = 512,
    multi_label: bool = False,
    stratified: bool = True,
    filter_unknown_types: bool = True,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Build train and validation dataloaders for multi-class classification.

    Args:
        manifest_path: Path to parquet manifest
        tokenizer: Tokenizer for text encoding
        class_mapping: Dictionary mapping label names to class indices
        target_column: Name of target column
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        batch_size: Batch size
        num_workers: Number of data loading workers
        random_seed: Random seed for reproducibility
        require_concept: Filter rows without concept text
        min_concept_length: Minimum concept text length
        max_length: Maximum sequence length
        multi_label: Whether this is multi-label classification
        stratified: Whether to use stratified splitting (single-label only)
        filter_unknown_types: Filter out rows with unknown label types

    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        stats: Dictionary with dataset statistics
    """
    # Load data
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

    # Extract labels
    if target_column in df.columns:
        labels = df[target_column].tolist()
    else:
        labels = (
            df["training_data"]
            .apply(lambda x: x.get(target_column) if isinstance(x, dict) else None)
            .tolist()
        )

    # Filter unknown types if requested
    if filter_unknown_types:
        valid_indices = []
        for i, label in enumerate(labels):
            if multi_label:
                # Multi-label: check if any label is valid
                if isinstance(label, str):
                    label = [label]
                if label and any(lbl in class_mapping for lbl in label):
                    valid_indices.append(i)
            else:
                # Single-label: check if label is valid
                if isinstance(label, list):
                    label = label[0] if label else None
                if label in class_mapping:
                    valid_indices.append(i)

        df = df.iloc[valid_indices].reset_index(drop=True)
        labels = [labels[i] for i in valid_indices]
        print(f"  After unknown type filter: {len(df)} tracks")

    # Initialize label encoder
    label_encoder = LabelEncoder(class_mapping)

    # Analyze label distribution
    dist = label_encoder.get_class_distribution(labels, multi_label=multi_label)
    print(f"\nLabel distribution ({target_column}):")
    for label, count in sorted(dist.items()):
        pct = (
            100 * count / len(labels) if not multi_label else 100 * count / len(labels)
        )
        print(f"  {label}: {count} ({pct:.1f}%)")

    # Split data
    if stratified and not multi_label:
        # Stratified split for single-label
        # Encode labels for stratification
        encoded_labels = [
            label_encoder.encode(lbl, multi_label=False) for lbl in labels
        ]

        train_indices, val_indices = train_test_split(
            np.arange(len(df)),
            train_size=train_split,
            test_size=val_split,
            stratify=encoded_labels,
            random_state=random_seed,
        )

        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True)
    else:
        # Random split (for multi-label or non-stratified)
        train_size = int(train_split * len(df))
        train_df = df.iloc[:train_size].reset_index(drop=True)
        val_df = df.iloc[train_size:].reset_index(drop=True)

    print("\nDataset split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")

    # Create datasets
    train_dataset = MultiClassRainbowDataset(
        df=train_df,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        target_column=target_column,
        multi_label=multi_label,
        max_length=max_length,
    )

    val_dataset = MultiClassRainbowDataset(
        df=val_df,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        target_column=target_column,
        multi_label=multi_label,
        max_length=max_length,
    )

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

    # Compute class weights
    class_counts = {}
    for label, count in dist.items():
        if label in class_mapping:
            class_counts[class_mapping[label]] = count

    from models.multiclass_classifier import MultiClassRebracketingClassifier

    class_weights = MultiClassRebracketingClassifier.compute_class_weights(
        class_counts=class_counts,
        num_classes=len(class_mapping),
        mode="balanced",
    )

    print("\nClass weights (balanced):")
    for label, idx in sorted(class_mapping.items(), key=lambda x: x[1]):
        print(f"  {label}: {class_weights[idx]:.3f}")

    stats = {
        "total_samples": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "num_classes": len(class_mapping),
        "class_weights": class_weights.tolist(),
        "class_distribution": dist,
        "label_encoder": label_encoder,
    }

    return train_loader, val_loader, stats
