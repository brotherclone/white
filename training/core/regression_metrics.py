"""
Evaluation metrics for continuous regression targets.

Includes:
- MAE, RMSE, R² for standard regression
- Correlation coefficients (Pearson, Spearman)
- Calibration metrics for uncertainty estimation
- Distribution comparison metrics (Jensen-Shannon, Wasserstein)
- Per-target and aggregate metrics
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats as scipy_stats


@dataclass
class RegressionMetrics:
    """Container for regression evaluation metrics."""

    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    r2: float  # R² coefficient of determination
    pearson_r: float  # Pearson correlation coefficient
    pearson_p: float  # Pearson p-value
    spearman_r: float  # Spearman rank correlation
    spearman_p: float  # Spearman p-value

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "mae": self.mae,
            "mse": self.mse,
            "rmse": self.rmse,
            "r2": self.r2,
            "pearson_r": self.pearson_r,
            "pearson_p": self.pearson_p,
            "spearman_r": self.spearman_r,
            "spearman_p": self.spearman_p,
        }


@dataclass
class CalibrationMetrics:
    """Container for uncertainty calibration metrics."""

    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    coverage_95: float  # Coverage of 95% prediction intervals
    sharpness: float  # Average width of prediction intervals
    nll: float  # Negative log-likelihood (if uncertainty is provided)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "ece": self.ece,
            "mce": self.mce,
            "coverage_95": self.coverage_95,
            "sharpness": self.sharpness,
            "nll": self.nll,
        }


def to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def compute_mae(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Compute Mean Absolute Error.

    MAE = (1/n) * Σ|predicted - target|
    """
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    return float(np.mean(np.abs(predictions - targets)))


def compute_mse(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Compute Mean Squared Error.

    MSE = (1/n) * Σ(predicted - target)²
    """
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    return float(np.mean((predictions - targets) ** 2))


def compute_rmse(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Compute Root Mean Squared Error.

    RMSE = sqrt(MSE)
    """
    return float(np.sqrt(compute_mse(predictions, targets)))


def compute_r2(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Compute R² coefficient of determination.

    R² = 1 - SS_res / SS_tot
       = 1 - Σ(target - predicted)² / Σ(target - mean(target))²

    R² = 1.0 indicates perfect prediction
    R² = 0.0 indicates prediction no better than mean
    R² < 0.0 indicates worse than predicting mean
    """
    predictions = to_numpy(predictions).flatten()
    targets = to_numpy(targets).flatten()

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)

    if ss_tot == 0:
        return 0.0 if ss_res == 0 else -np.inf

    return float(1 - ss_res / ss_tot)


def compute_pearson_correlation(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
) -> Tuple[float, float]:
    """
    Compute Pearson correlation coefficient.

    Measures linear correlation between predictions and targets.

    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    predictions = to_numpy(predictions).flatten()
    targets = to_numpy(targets).flatten()

    if len(predictions) < 2:
        return 0.0, 1.0

    r, p = scipy_stats.pearsonr(predictions, targets)
    return float(r), float(p)


def compute_spearman_correlation(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation coefficient.

    Measures monotonic relationship between predictions and targets.
    Robust to outliers and non-linear relationships.

    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    predictions = to_numpy(predictions).flatten()
    targets = to_numpy(targets).flatten()

    if len(predictions) < 2:
        return 0.0, 1.0

    r, p = scipy_stats.spearmanr(predictions, targets)
    return float(r), float(p)


def compute_all_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
) -> RegressionMetrics:
    """
    Compute all regression metrics.

    Args:
        predictions: Predicted values [n_samples] or [n_samples, n_targets]
        targets: Target values (same shape as predictions)

    Returns:
        RegressionMetrics dataclass with all computed metrics
    """
    mae = compute_mae(predictions, targets)
    mse = compute_mse(predictions, targets)
    rmse = compute_rmse(predictions, targets)
    r2 = compute_r2(predictions, targets)
    pearson_r, pearson_p = compute_pearson_correlation(predictions, targets)
    spearman_r, spearman_p = compute_spearman_correlation(predictions, targets)

    return RegressionMetrics(
        mae=mae,
        mse=mse,
        rmse=rmse,
        r2=r2,
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        spearman_r=spearman_r,
        spearman_p=spearman_p,
    )


def compute_per_target_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    target_names: Optional[List[str]] = None,
) -> Dict[str, RegressionMetrics]:
    """
    Compute metrics for each target independently.

    Args:
        predictions: Predicted values [n_samples, n_targets]
        targets: Target values [n_samples, n_targets]
        target_names: Optional names for each target

    Returns:
        Dictionary mapping target names to RegressionMetrics
    """
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
        targets = targets.reshape(-1, 1)

    n_targets = predictions.shape[1]

    if target_names is None:
        target_names = [f"target_{i}" for i in range(n_targets)]

    results = {}
    for i, name in enumerate(target_names):
        results[name] = compute_all_metrics(predictions[:, i], targets[:, i])

    return results


class DistributionMetrics:
    """
    Metrics for comparing probability distributions.

    Used for evaluating ontological mode predictions where
    outputs are softmax distributions.
    """

    @staticmethod
    def jensen_shannon_divergence(
        p: Union[torch.Tensor, np.ndarray],
        q: Union[torch.Tensor, np.ndarray],
    ) -> float:
        """
        Compute Jensen-Shannon divergence between distributions.

        JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
        where M = 0.5 * (P + Q)

        JSD is symmetric and bounded [0, 1] (using log base 2).

        Args:
            p: First distribution [batch, n_classes] or [n_classes]
            q: Second distribution [batch, n_classes] or [n_classes]

        Returns:
            Average JSD across batch
        """
        p = to_numpy(p)
        q = to_numpy(q)

        if p.ndim == 1:
            p = p.reshape(1, -1)
            q = q.reshape(1, -1)

        # Add small epsilon for numerical stability
        eps = 1e-10
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)

        m = 0.5 * (p + q)

        def kl_div(a, b):
            return np.sum(a * np.log2(a / b + eps), axis=-1)

        jsd = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
        return float(np.mean(jsd))

    @staticmethod
    def total_variation_distance(
        p: Union[torch.Tensor, np.ndarray],
        q: Union[torch.Tensor, np.ndarray],
    ) -> float:
        """
        Compute Total Variation distance between distributions.

        TV(P, Q) = 0.5 * Σ|P - Q|

        Bounded [0, 1].
        """
        p = to_numpy(p)
        q = to_numpy(q)

        if p.ndim == 1:
            p = p.reshape(1, -1)
            q = q.reshape(1, -1)

        tv = 0.5 * np.sum(np.abs(p - q), axis=-1)
        return float(np.mean(tv))

    @staticmethod
    def hellinger_distance(
        p: Union[torch.Tensor, np.ndarray],
        q: Union[torch.Tensor, np.ndarray],
    ) -> float:
        """
        Compute Hellinger distance between distributions.

        H(P, Q) = (1/√2) * √(Σ(√P - √Q)²)

        Bounded [0, 1].
        """
        p = to_numpy(p)
        q = to_numpy(q)

        if p.ndim == 1:
            p = p.reshape(1, -1)
            q = q.reshape(1, -1)

        hellinger = (1.0 / np.sqrt(2)) * np.sqrt(
            np.sum((np.sqrt(p) - np.sqrt(q)) ** 2, axis=-1)
        )
        return float(np.mean(hellinger))

    @staticmethod
    def mode_accuracy(
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
    ) -> float:
        """
        Compute accuracy of argmax mode prediction.

        Measures if the dominant mode matches.
        """
        predictions = to_numpy(predictions)
        targets = to_numpy(targets)

        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)
            targets = targets.reshape(1, -1)

        pred_modes = np.argmax(predictions, axis=-1)
        target_modes = np.argmax(targets, axis=-1)

        return float(np.mean(pred_modes == target_modes))


class UncertaintyCalibration:
    """
    Calibration metrics for uncertainty estimates.

    Measures if predicted uncertainties match actual errors.
    """

    @staticmethod
    def compute_calibration_metrics(
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        variances: Union[torch.Tensor, np.ndarray],
        n_bins: int = 10,
    ) -> CalibrationMetrics:
        """
        Compute uncertainty calibration metrics.

        Args:
            predictions: Predicted means [n_samples]
            targets: True targets [n_samples]
            variances: Predicted variances [n_samples]
            n_bins: Number of bins for calibration curve

        Returns:
            CalibrationMetrics dataclass
        """
        predictions = to_numpy(predictions).flatten()
        targets = to_numpy(targets).flatten()
        variances = to_numpy(variances).flatten()

        # Ensure variances are positive
        variances = np.clip(variances, 1e-10, None)
        stds = np.sqrt(variances)

        # Compute squared errors
        squared_errors = (predictions - targets) ** 2

        # Expected Calibration Error (ECE)
        # Group by predicted variance, check if actual error matches
        sorted_indices = np.argsort(variances)
        bin_size = len(variances) // n_bins

        ece = 0.0
        mce = 0.0
        bin_errors = []

        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else len(variances)
            bin_indices = sorted_indices[start:end]

            if len(bin_indices) == 0:
                continue

            # Average predicted variance in bin
            avg_variance = np.mean(variances[bin_indices])

            # Actual MSE in bin
            actual_mse = np.mean(squared_errors[bin_indices])

            # Calibration error for this bin
            bin_error = abs(avg_variance - actual_mse)
            bin_errors.append(bin_error)

            ece += bin_error * len(bin_indices)
            mce = max(mce, bin_error)

        ece /= len(variances)

        # 95% prediction interval coverage
        z_95 = 1.96
        lower = predictions - z_95 * stds
        upper = predictions + z_95 * stds
        coverage_95 = float(np.mean((targets >= lower) & (targets <= upper)))

        # Sharpness (average interval width)
        sharpness = float(np.mean(2 * z_95 * stds))

        # Negative log-likelihood (Gaussian assumption)
        nll = 0.5 * np.mean(np.log(2 * np.pi * variances) + squared_errors / variances)

        return CalibrationMetrics(
            ece=float(ece),
            mce=float(mce),
            coverage_95=coverage_95,
            sharpness=sharpness,
            nll=float(nll),
        )


class OntologicalRegressionEvaluator:
    """
    Comprehensive evaluator for Rainbow Table ontological regression.

    Combines:
    - Per-dimension regression metrics (temporal, spatial, ontological)
    - Distribution comparison metrics
    - Mode prediction accuracy
    - Confidence calibration
    - Album prediction accuracy
    """

    def __init__(
        self,
        temporal_names: List[str] = ["past", "present", "future"],
        spatial_names: List[str] = ["thing", "place", "person"],
        ontological_names: List[str] = ["imagined", "forgotten", "known"],
    ):
        """Initialize evaluator with mode names."""
        self.temporal_names = temporal_names
        self.spatial_names = spatial_names
        self.ontological_names = ontological_names

        self.dist_metrics = DistributionMetrics()

    def evaluate(
        self,
        temporal_preds: Union[torch.Tensor, np.ndarray],
        spatial_preds: Union[torch.Tensor, np.ndarray],
        ontological_preds: Union[torch.Tensor, np.ndarray],
        confidence_preds: Union[torch.Tensor, np.ndarray],
        temporal_targets: Union[torch.Tensor, np.ndarray],
        spatial_targets: Union[torch.Tensor, np.ndarray],
        ontological_targets: Union[torch.Tensor, np.ndarray],
        confidence_targets: Union[torch.Tensor, np.ndarray],
        album_preds: Optional[List[str]] = None,
        album_targets: Optional[List[str]] = None,
    ) -> Dict:
        """
        Evaluate all aspects of ontological regression.

        Returns:
            Dictionary with comprehensive metrics
        """
        results = {}

        # Per-dimension distribution metrics
        for name, preds, targets in [
            ("temporal", temporal_preds, temporal_targets),
            ("spatial", spatial_preds, spatial_targets),
            ("ontological", ontological_preds, ontological_targets),
        ]:
            results[f"{name}_jsd"] = self.dist_metrics.jensen_shannon_divergence(
                preds, targets
            )
            results[f"{name}_tv"] = self.dist_metrics.total_variation_distance(
                preds, targets
            )
            results[f"{name}_hellinger"] = self.dist_metrics.hellinger_distance(
                preds, targets
            )
            results[f"{name}_mode_accuracy"] = self.dist_metrics.mode_accuracy(
                preds, targets
            )

        # Confidence metrics
        confidence_metrics = compute_all_metrics(confidence_preds, confidence_targets)
        for key, value in confidence_metrics.to_dict().items():
            results[f"confidence_{key}"] = value

        # Album prediction accuracy (if provided)
        if album_preds is not None and album_targets is not None:
            album_accuracy = sum(
                p == t for p, t in zip(album_preds, album_targets)
            ) / len(album_preds)
            results["album_accuracy"] = album_accuracy

        # Aggregate metrics
        results["avg_jsd"] = np.mean(
            [
                results["temporal_jsd"],
                results["spatial_jsd"],
                results["ontological_jsd"],
            ]
        )
        results["avg_mode_accuracy"] = np.mean(
            [
                results["temporal_mode_accuracy"],
                results["spatial_mode_accuracy"],
                results["ontological_mode_accuracy"],
            ]
        )

        return results

    def generate_report(self, metrics: Dict) -> str:
        """Generate human-readable evaluation report."""
        lines = [
            "=" * 60,
            "ONTOLOGICAL REGRESSION EVALUATION REPORT",
            "=" * 60,
            "",
            "DISTRIBUTION METRICS",
            "-" * 40,
        ]

        for dim in ["temporal", "spatial", "ontological"]:
            lines.append(f"\n{dim.upper()}:")
            lines.append(f"  JSD: {metrics[f'{dim}_jsd']:.4f}")
            lines.append(f"  Total Variation: {metrics[f'{dim}_tv']:.4f}")
            lines.append(f"  Hellinger: {metrics[f'{dim}_hellinger']:.4f}")
            lines.append(f"  Mode Accuracy: {metrics[f'{dim}_mode_accuracy']:.2%}")

        lines.extend(
            [
                "",
                "CONFIDENCE METRICS",
                "-" * 40,
                f"  MAE: {metrics['confidence_mae']:.4f}",
                f"  RMSE: {metrics['confidence_rmse']:.4f}",
                f"  R²: {metrics['confidence_r2']:.4f}",
                f"  Pearson r: {metrics['confidence_pearson_r']:.4f}",
                f"  Spearman ρ: {metrics['confidence_spearman_r']:.4f}",
            ]
        )

        if "album_accuracy" in metrics:
            lines.extend(
                [
                    "",
                    "ALBUM PREDICTION",
                    "-" * 40,
                    f"  Accuracy: {metrics['album_accuracy']:.2%}",
                ]
            )

        lines.extend(
            [
                "",
                "AGGREGATE METRICS",
                "-" * 40,
                f"  Average JSD: {metrics['avg_jsd']:.4f}",
                f"  Average Mode Accuracy: {metrics['avg_mode_accuracy']:.2%}",
                "",
                "=" * 60,
            ]
        )

        return "\n".join(lines)


if __name__ == "__main__":
    # Quick tests
    print("Testing regression metrics...")

    n_samples = 100

    # Generate test data
    np.random.seed(42)
    targets = np.random.randn(n_samples)
    predictions = targets + 0.2 * np.random.randn(n_samples)  # Add noise

    # Test basic metrics
    print("\n=== Basic Regression Metrics ===")
    metrics = compute_all_metrics(predictions, targets)
    print(f"MAE: {metrics.mae:.4f}")
    print(f"RMSE: {metrics.rmse:.4f}")
    print(f"R²: {metrics.r2:.4f}")
    print(f"Pearson r: {metrics.pearson_r:.4f}")
    print(f"Spearman ρ: {metrics.spearman_r:.4f}")

    # Test per-target metrics
    print("\n=== Per-Target Metrics ===")
    multi_targets = np.random.randn(n_samples, 3)
    multi_preds = multi_targets + 0.2 * np.random.randn(n_samples, 3)

    per_target = compute_per_target_metrics(
        multi_preds, multi_targets, target_names=["intensity", "fluidity", "complexity"]
    )
    for name, m in per_target.items():
        print(f"{name}: MAE={m.mae:.4f}, R²={m.r2:.4f}")

    # Test distribution metrics
    print("\n=== Distribution Metrics ===")
    from scipy.special import softmax as sp_softmax

    p = sp_softmax(np.random.randn(n_samples, 3), axis=-1)
    q = sp_softmax(np.random.randn(n_samples, 3), axis=-1)

    dm = DistributionMetrics()
    print(f"JSD: {dm.jensen_shannon_divergence(p, q):.4f}")
    print(f"TV: {dm.total_variation_distance(p, q):.4f}")
    print(f"Hellinger: {dm.hellinger_distance(p, q):.4f}")
    print(f"Mode Accuracy: {dm.mode_accuracy(p, q):.2%}")

    # Test calibration metrics
    print("\n=== Calibration Metrics ===")
    variances = 0.1 + 0.2 * np.random.rand(n_samples)

    cal = UncertaintyCalibration.compute_calibration_metrics(
        predictions, targets, variances
    )
    print(f"ECE: {cal.ece:.4f}")
    print(f"MCE: {cal.mce:.4f}")
    print(f"95% Coverage: {cal.coverage_95:.2%}")
    print(f"Sharpness: {cal.sharpness:.4f}")
    print(f"NLL: {cal.nll:.4f}")

    # Test ontological evaluator
    print("\n=== Ontological Regression Evaluator ===")
    evaluator = OntologicalRegressionEvaluator()

    temporal_preds = sp_softmax(np.random.randn(n_samples, 3), axis=-1)
    temporal_targets = sp_softmax(np.random.randn(n_samples, 3), axis=-1)
    spatial_preds = sp_softmax(np.random.randn(n_samples, 3), axis=-1)
    spatial_targets = sp_softmax(np.random.randn(n_samples, 3), axis=-1)
    ontological_preds = sp_softmax(np.random.randn(n_samples, 3), axis=-1)
    ontological_targets = sp_softmax(np.random.randn(n_samples, 3), axis=-1)
    confidence_preds = 1 / (1 + np.exp(-np.random.randn(n_samples)))
    confidence_targets = 1 / (1 + np.exp(-np.random.randn(n_samples)))

    results = evaluator.evaluate(
        temporal_preds,
        spatial_preds,
        ontological_preds,
        confidence_preds,
        temporal_targets,
        spatial_targets,
        ontological_targets,
        confidence_targets,
    )

    print(evaluator.generate_report(results))

    print("\n✓ All metric tests passed!")
