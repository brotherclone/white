"""
Core training components for Rainbow Pipeline.

This module provides:
- Binary pipeline: build_dataloaders
- Multiclass pipeline: build_multiclass_dataloaders, LabelEncoder
- Metrics: MultiClassMetrics, MultiLabelMetrics, top_k_accuracy
"""

from core.pipeline import build_dataloaders
from core.multiclass_pipeline import (
    build_multiclass_dataloaders,
    LabelEncoder,
    MultiClassRainbowDataset,
)
from core.multiclass_metrics import (
    MultiClassMetrics,
    MultiLabelMetrics,
    top_k_accuracy,
)

__all__ = [
    # Binary pipeline
    "build_dataloaders",
    # Multiclass pipeline
    "build_multiclass_dataloaders",
    "LabelEncoder",
    "MultiClassRainbowDataset",
    # Metrics
    "MultiClassMetrics",
    "MultiLabelMetrics",
    "top_k_accuracy",
]
