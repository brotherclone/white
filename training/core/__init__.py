"""
Core training components for Rainbow Pipeline.

This module provides:
- Binary pipeline: build_dataloaders
- Multiclass pipeline: build_multiclass_dataloaders, LabelEncoder
- Metrics: MultiClassMetrics, MultiLabelMetrics, top_k_accuracy
- Regression losses: MSELoss, HuberLoss, OntologicalRegressionLoss, etc.
- Regression metrics: MAE, RMSE, R², calibration metrics, etc.
"""

from core.multiclass_metrics import (
    MultiClassMetrics,
    MultiLabelMetrics,
    top_k_accuracy,
)
from core.multiclass_pipeline import (
    LabelEncoder,
    MultiClassRainbowDataset,
    build_multiclass_dataloaders,
)
from core.pipeline import build_dataloaders
from core.regression_losses import (
    CombinedClassificationRegressionLoss,
    DistributionMSELoss,
    HuberLoss,
    KLDivergenceLoss,
    MSELoss,
    OntologicalRegressionLoss,
    PerTargetWeightedLoss,
    RegressionLossOutput,
    SmoothL1Loss,
)
from core.regression_metrics import (
    CalibrationMetrics,
    DistributionMetrics,
    OntologicalRegressionEvaluator,
    RegressionMetrics,
    UncertaintyCalibration,
    compute_all_metrics,
    compute_mae,
    compute_mse,
    compute_pearson_correlation,
    compute_per_target_metrics,
    compute_r2,
    compute_rmse,
    compute_spearman_correlation,
)
from core.regression_pipeline import (
    MultiTaskRainbowDataset,
    build_multitask_dataloaders,
)
from core.soft_targets import (
    SoftTargetGenerator,
    SoftTargets,
    TargetConsistencyValidator,
    generate_soft_targets_from_dataframe,
)

__all__ = [
    # Binary pipeline
    "build_dataloaders",
    # Multiclass pipeline
    "build_multiclass_dataloaders",
    "LabelEncoder",
    "MultiClassRainbowDataset",
    # Classification metrics
    "MultiClassMetrics",
    "MultiLabelMetrics",
    "top_k_accuracy",
    # Regression losses
    "MSELoss",
    "HuberLoss",
    "SmoothL1Loss",
    "KLDivergenceLoss",
    "DistributionMSELoss",
    "PerTargetWeightedLoss",
    "OntologicalRegressionLoss",
    "CombinedClassificationRegressionLoss",
    "RegressionLossOutput",
    # Regression metrics
    "RegressionMetrics",
    "CalibrationMetrics",
    "compute_mae",
    "compute_mse",
    "compute_rmse",
    "compute_r2",
    "compute_pearson_correlation",
    "compute_spearman_correlation",
    "compute_all_metrics",
    "compute_per_target_metrics",
    "DistributionMetrics",
    "UncertaintyCalibration",
    "OntologicalRegressionEvaluator",
    # Soft targets
    "SoftTargets",
    "SoftTargetGenerator",
    "TargetConsistencyValidator",
    "generate_soft_targets_from_dataframe",
    # Regression pipeline
    "MultiTaskRainbowDataset",
    "build_multitask_dataloaders",
]
