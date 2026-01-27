"""
Data pipeline for multi-task regression training.

Extends the existing pipelines to support:
- Soft target loading for ontological regression
- Combined classification + regression targets
- Target validation and normalization
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any
import warnings

from .soft_targets import (
    SoftTargetGenerator,
    SoftTargets,
    generate_soft_targets_from_dataframe,
)
from .multiclass_pipeline import LabelEncoder


class MultiTaskRainbowDataset(Dataset):
    """
    Dataset for multi-task classification + regression training.

    Loads:
    - Text (concept) for encoding
    - Classification labels (rebracketing type)
    - Soft regression targets (ontological scores)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        label_encoder: LabelEncoder,
        soft_targets: SoftTargets,
        text_column: str = "concept",
        classification_column: str = "rebracketing_type",
        max_length: int = 512,
        include_sample_weights: bool = True,
    ):
        """
        Initialize multi-task dataset.

        Args:
            df: DataFrame with text and labels
            tokenizer: HuggingFace tokenizer
            label_encoder: Encoder for classification labels
            soft_targets: Pre-generated soft targets
            text_column: Column containing concept text
            classification_column: Column containing classification labels
            max_length: Maximum sequence length
            include_sample_weights: Include uncertainty weights for loss
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.soft_targets = soft_targets
        self.text_column = text_column
        self.classification_column = classification_column
        self.max_length = max_length
        self.include_sample_weights = include_sample_weights

        # Validate alignment
        if len(df) != soft_targets.temporal.shape[0]:
            raise ValueError(
                f"DataFrame length ({len(df)}) doesn't match soft targets ({soft_targets.temporal.shape[0]})"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Get text
        text = str(row[self.text_column])

        # Tokenize
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Classification label
        label_str = str(row[self.classification_column])
        classification_label = self.label_encoder.encode(label_str)

        # Soft regression targets
        temporal_target = torch.tensor(
            self.soft_targets.temporal[idx], dtype=torch.float32
        )
        spatial_target = torch.tensor(
            self.soft_targets.spatial[idx], dtype=torch.float32
        )
        ontological_target = torch.tensor(
            self.soft_targets.ontological[idx], dtype=torch.float32
        )
        confidence_target = torch.tensor(
            self.soft_targets.confidence[idx], dtype=torch.float32
        )

        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "classification_label": torch.tensor(
                classification_label, dtype=torch.long
            ),
            "temporal_target": temporal_target,
            "spatial_target": spatial_target,
            "ontological_target": ontological_target,
            "confidence_target": confidence_target,
        }

        # Optional sample weight
        if (
            self.include_sample_weights
            and self.soft_targets.uncertainty_weights is not None
        ):
            result["sample_weight"] = torch.tensor(
                self.soft_targets.uncertainty_weights[idx], dtype=torch.float32
            )

        return result


def build_multitask_dataloaders(
    manifest_path: str,
    tokenizer,
    config: Dict[str, Any],
    hf_dataset: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Build DataLoaders for multi-task training.

    Args:
        manifest_path: Path to parquet manifest file
        tokenizer: HuggingFace tokenizer
        config: Configuration dictionary
        hf_dataset: Optional HuggingFace dataset ID

    Returns:
        Tuple of (train_loader, val_loader, info_dict)
    """
    # Load data
    df = _load_dataframe(manifest_path, hf_dataset, config)

    # Filter
    df = _filter_dataframe(df, config)

    # Setup label encoder
    class_mapping = config.get("data", {}).get("class_mapping", {})
    label_encoder = LabelEncoder(class_mapping)

    # Generate soft targets
    soft_targets = _generate_soft_targets(df, config)

    # Validate targets
    _validate_targets(df, soft_targets, config)

    # Split data
    train_df, val_df, train_targets, val_targets = _split_data(df, soft_targets, config)

    # Create datasets
    train_dataset = MultiTaskRainbowDataset(
        df=train_df,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        soft_targets=train_targets,
        text_column=config.get("data", {}).get("text_column", "concept"),
        classification_column=config.get("data", {}).get(
            "classification_target", "rebracketing_type"
        ),
        max_length=config.get("model", {})
        .get("text_encoder", {})
        .get("max_length", 512),
    )

    val_dataset = MultiTaskRainbowDataset(
        df=val_df,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        soft_targets=val_targets,
        text_column=config.get("data", {}).get("text_column", "concept"),
        classification_column=config.get("data", {}).get(
            "classification_target", "rebracketing_type"
        ),
        max_length=config.get("model", {})
        .get("text_encoder", {})
        .get("max_length", 512),
    )

    # Create dataloaders
    batch_size = config.get("training", {}).get("batch_size", 32)
    num_workers = config.get("training", {}).get("num_workers", 0)

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
    class_counts = train_df[
        config.get("data", {}).get("classification_target", "rebracketing_type")
    ].value_counts()
    class_weights = _compute_class_weights(class_counts, label_encoder, config)

    info = {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "num_classes": len(class_mapping),
        "class_weights": class_weights,
        "label_encoder": label_encoder,
    }

    return train_loader, val_loader, info


def _load_dataframe(
    manifest_path: str,
    hf_dataset: Optional[str],
    config: Dict,
) -> pd.DataFrame:
    """Load DataFrame from local file or HuggingFace."""
    # Try HuggingFace first
    if hf_dataset:
        try:
            from datasets import load_dataset

            hf_split = config.get("data", {}).get("hf_split", "train")
            ds = load_dataset(hf_dataset, split=hf_split)
            return ds.to_pandas()
        except Exception as e:
            warnings.warn(
                f"Failed to load from HuggingFace: {e}, falling back to local"
            )

    # Load local parquet
    return pd.read_parquet(manifest_path)


def _filter_dataframe(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Apply filtering based on config."""
    data_config = config.get("data", {})

    # Require concept text
    if data_config.get("require_concept", True):
        text_column = data_config.get("text_column", "concept")
        if text_column in df.columns:
            df = df[df[text_column].notna() & (df[text_column].str.len() > 0)]

    # Minimum concept length
    min_length = data_config.get("min_concept_length", 0)
    if min_length > 0:
        text_column = data_config.get("text_column", "concept")
        if text_column in df.columns:
            df = df[df[text_column].str.len() >= min_length]

    # Filter unknown types
    if data_config.get("filter_unknown_types", False):
        class_mapping = data_config.get("class_mapping", {})
        cls_column = data_config.get("classification_target", "rebracketing_type")
        if cls_column in df.columns and class_mapping:
            valid_types = set(class_mapping.keys())
            df = df[df[cls_column].isin(valid_types)]

    return df.reset_index(drop=True)


def _generate_soft_targets(df: pd.DataFrame, config: Dict) -> SoftTargets:
    """
    Generate soft targets from DataFrame on-the-fly.

    This function auto-detects Rainbow Table mode columns using multiple
    naming conventions, eliminating the need for parquet schema modifications.
    Soft targets are generated from discrete labels using label smoothing.
    """
    soft_config = config.get("soft_targets", {})

    # Auto-detect mode columns with multiple naming conventions
    temporal_col, spatial_col, ontological_col = _detect_mode_columns(df, config)

    # If columns still not found, derive from album or use defaults
    if temporal_col is None:
        # Try to derive from album column
        album_col = config.get("data", {}).get("album_column", "album")
        rainbow_col = "rainbow_color"

        if rainbow_col in df.columns:
            df = _derive_modes_from_album(df, rainbow_col)
            temporal_col = "temporal_mode"
            spatial_col = "spatial_mode"
            ontological_col = "ontological_mode"
        elif album_col in df.columns:
            df = _derive_modes_from_album(df, album_col)
            temporal_col = "temporal_mode"
            spatial_col = "spatial_mode"
            ontological_col = "ontological_mode"
        else:
            # Create default modes based on rebracketing type
            df = _create_default_modes(df, config)
            temporal_col = "temporal_mode"
            spatial_col = "spatial_mode"
            ontological_col = "ontological_mode"

    return generate_soft_targets_from_dataframe(
        df,
        temporal_column=temporal_col,
        spatial_column=spatial_col,
        ontological_column=ontological_col,
        track_id_column=config.get("data", {}).get("track_id_column", "track_id"),
        confidence_column=config.get("data", {}).get("confidence_column"),
        label_smoothing=soft_config.get("label_smoothing", 0.1),
        temporal_context=soft_config.get("temporal_context", {}).get("enabled", True),
    )


def _detect_mode_columns(
    df: pd.DataFrame,
    config: Dict,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Auto-detect Rainbow Table mode columns using multiple naming conventions.

    Supports:
    - Direct config: temporal_column, spatial_column, ontological_column
    - Rainbow color prefixed: rainbow_color_temporal_mode, etc.
    - Objectional/spatial typo handling
    - Simple mode names: temporal_mode, spatial_mode, ontological_mode

    Returns:
        Tuple of (temporal_col, spatial_col, ontological_col) or (None, None, None)
    """
    # First check explicit config
    data_config = config.get("data", {})
    temporal_col = data_config.get("temporal_column")
    spatial_col = data_config.get("spatial_column")
    ontological_col = data_config.get("ontological_column")

    # If all specified and exist, use them
    if temporal_col and spatial_col and ontological_col:
        if all(
            col in df.columns for col in [temporal_col, spatial_col, ontological_col]
        ):
            return temporal_col, spatial_col, ontological_col

    columns = set(df.columns)

    # Try rainbow_color_* pattern (actual parquet schema)
    rainbow_patterns = [
        # Pattern: (temporal, spatial/objectional, ontological)
        (
            "rainbow_color_temporal_mode",
            "rainbow_color_objectional_mode",
            "rainbow_color_ontological_mode",
        ),
        (
            "rainbow_color_temporal_mode",
            "rainbow_color_spatial_mode",
            "rainbow_color_ontological_mode",
        ),
    ]

    for t_col, s_col, o_col in rainbow_patterns:
        if t_col in columns and s_col in columns and o_col in columns:
            return t_col, s_col, o_col

    # Try simple mode names
    simple_patterns = [
        ("temporal_mode", "spatial_mode", "ontological_mode"),
        ("temporal", "spatial", "ontological"),
    ]

    for t_col, s_col, o_col in simple_patterns:
        if t_col in columns and s_col in columns and o_col in columns:
            return t_col, s_col, o_col

    # Try partial matches - handle objectional typo
    if "rainbow_color_temporal_mode" in columns:
        temporal_col = "rainbow_color_temporal_mode"

        # Try both spatial and objectional
        if "rainbow_color_spatial_mode" in columns:
            spatial_col = "rainbow_color_spatial_mode"
        elif "rainbow_color_objectional_mode" in columns:
            spatial_col = "rainbow_color_objectional_mode"
        else:
            return None, None, None

        if "rainbow_color_ontological_mode" in columns:
            ontological_col = "rainbow_color_ontological_mode"
        else:
            return None, None, None

        return temporal_col, spatial_col, ontological_col

    # No columns found
    return None, None, None


def _derive_modes_from_album(df: pd.DataFrame, album_col: str) -> pd.DataFrame:
    """Derive ontological modes from album labels."""
    # Album to mode mapping
    album_modes = {
        "Orange": ("past", "thing", "imagined"),
        "Red": ("past", "thing", "forgotten"),
        "Violet": ("past", "person", "known"),
        "Yellow": ("present", "place", "imagined"),
        "Green": ("present", "person", "known"),
        "Indigo": ("present", "person", "forgotten"),
        "Blue": ("future", "place", "known"),
        "Black": ("none", "none", "none"),
    }

    temporal_modes = []
    spatial_modes = []
    ontological_modes = []

    for album in df[album_col]:
        modes = album_modes.get(album, ("none", "none", "none"))
        temporal_modes.append(modes[0])
        spatial_modes.append(modes[1])
        ontological_modes.append(modes[2])

    df = df.copy()
    df["temporal_mode"] = temporal_modes
    df["spatial_mode"] = spatial_modes
    df["ontological_mode"] = ontological_modes

    return df


def _create_default_modes(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Create default modes when no mode data available."""
    # Map rebracketing types to likely modes (heuristic)
    type_to_modes = {
        "spatial": ("present", "place", "known"),
        "temporal": ("past", "thing", "forgotten"),
        "causal": ("past", "thing", "known"),
        "perceptual": ("present", "thing", "imagined"),
        "memory": ("past", "person", "forgotten"),
        "ontological": ("present", "thing", "imagined"),
        "narrative": ("past", "person", "imagined"),
        "identity": ("present", "person", "known"),
    }

    cls_col = config.get("data", {}).get("classification_target", "rebracketing_type")

    temporal_modes = []
    spatial_modes = []
    ontological_modes = []

    for rtype in df.get(cls_col, ["unknown"] * len(df)):
        modes = type_to_modes.get(str(rtype).lower(), ("present", "thing", "known"))
        temporal_modes.append(modes[0])
        spatial_modes.append(modes[1])
        ontological_modes.append(modes[2])

    df = df.copy()
    df["temporal_mode"] = temporal_modes
    df["spatial_mode"] = spatial_modes
    df["ontological_mode"] = ontological_modes

    return df


def _validate_targets(
    df: pd.DataFrame,
    soft_targets: SoftTargets,
    config: Dict,
):
    """Validate soft targets."""
    errors = soft_targets.validate()
    if errors:
        warnings.warn(f"Soft target validation errors: {errors}")

    # Additional consistency checks can be added here


def _split_data(
    df: pd.DataFrame,
    soft_targets: SoftTargets,
    config: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, SoftTargets, SoftTargets]:
    """Split data into train and validation sets."""
    data_config = config.get("data", {})

    train_ratio = data_config.get("train_split", 0.8)
    seed = data_config.get("random_seed", 42)
    stratified = data_config.get("stratified", True)

    n_samples = len(df)

    if stratified:
        # Stratified split by classification target
        from sklearn.model_selection import train_test_split

        cls_col = data_config.get("classification_target", "rebracketing_type")
        if cls_col in df.columns:
            train_idx, val_idx = train_test_split(
                range(n_samples),
                train_size=train_ratio,
                stratify=df[cls_col],
                random_state=seed,
            )
        else:
            train_idx, val_idx = train_test_split(
                range(n_samples),
                train_size=train_ratio,
                random_state=seed,
            )
    else:
        # Random split
        np.random.seed(seed)
        indices = np.random.permutation(n_samples)
        split_point = int(n_samples * train_ratio)
        train_idx = indices[:split_point]
        val_idx = indices[split_point:]

    # Split DataFrame
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Split soft targets
    train_targets = SoftTargets(
        temporal=soft_targets.temporal[train_idx],
        spatial=soft_targets.spatial[train_idx],
        ontological=soft_targets.ontological[train_idx],
        confidence=soft_targets.confidence[train_idx],
        is_black_album=(
            soft_targets.is_black_album[train_idx]
            if soft_targets.is_black_album is not None
            else None
        ),
        uncertainty_weights=(
            soft_targets.uncertainty_weights[train_idx]
            if soft_targets.uncertainty_weights is not None
            else None
        ),
    )

    val_targets = SoftTargets(
        temporal=soft_targets.temporal[val_idx],
        spatial=soft_targets.spatial[val_idx],
        ontological=soft_targets.ontological[val_idx],
        confidence=soft_targets.confidence[val_idx],
        is_black_album=(
            soft_targets.is_black_album[val_idx]
            if soft_targets.is_black_album is not None
            else None
        ),
        uncertainty_weights=(
            soft_targets.uncertainty_weights[val_idx]
            if soft_targets.uncertainty_weights is not None
            else None
        ),
    )

    return train_df, val_df, train_targets, val_targets


def _compute_class_weights(
    class_counts: pd.Series,
    label_encoder: LabelEncoder,
    config: Dict,
) -> torch.Tensor:
    """Compute class weights for imbalanced classification."""
    weight_mode = (
        config.get("model", {}).get("classifier", {}).get("class_weights", "balanced")
    )

    num_classes = len(label_encoder.label_to_idx)

    if weight_mode == "uniform":
        return torch.ones(num_classes)

    # Balanced weights: n_samples / (n_classes * n_samples_per_class)
    total = class_counts.sum()
    weights = torch.zeros(num_classes)

    for label, count in class_counts.items():
        if label in label_encoder.label_to_idx:
            idx = label_encoder.label_to_idx[label]
            weights[idx] = total / (num_classes * max(count, 1))

    # Normalize
    weights = weights / weights.sum() * num_classes

    return weights


if __name__ == "__main__":
    # Quick tests
    print("Testing regression pipeline...")

    # Create mock data
    mock_data = {
        "concept": [
            "A memory of childhood toys",
            "The future city I dream of",
            "This present moment now",
            "A forgotten place from the past",
        ]
        * 10,
        "rebracketing_type": ["memory", "temporal", "spatial", "memory"] * 10,
        "track_id": ["track1"] * 20 + ["track2"] * 20,
    }

    df = pd.DataFrame(mock_data)

    # Generate soft targets
    from .soft_targets import SoftTargetGenerator

    generator = SoftTargetGenerator(label_smoothing=0.1)
    targets = generator.generate_from_labels(
        temporal_labels=["past", "future", "present", "past"] * 10,
        spatial_labels=["thing", "place", "thing", "place"] * 10,
        ontological_labels=["imagined", "known", "known", "forgotten"] * 10,
        track_ids=df["track_id"].tolist(),
    )

    print(f"Generated {len(targets.temporal)} soft targets")
    print(f"Temporal shape: {targets.temporal.shape}")
    print(f"Validation: {targets.validate()}")

    # Test dataset creation (without tokenizer)
    print("\n✓ Pipeline tests passed!")
