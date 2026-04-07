"""
Visualization utilities for Rainbow Pipeline.

This module provides plotting and visualization tools for:
- Regression target distributions
- Soft vs hard target comparisons
- Album balance analysis
- Calibration curves
- Ontological state embeddings
"""

from visualization.regression_plots import (
    RegressionVisualizer,
    plot_album_balance,
    plot_calibration_curve,
    plot_ontological_space,
    plot_per_dimension_metrics,
    plot_prediction_intervals,
    plot_soft_vs_hard_targets,
    plot_target_distribution,
)

__all__ = [
    "plot_target_distribution",
    "plot_soft_vs_hard_targets",
    "plot_album_balance",
    "plot_calibration_curve",
    "plot_ontological_space",
    "plot_prediction_intervals",
    "plot_per_dimension_metrics",
    "RegressionVisualizer",
]
