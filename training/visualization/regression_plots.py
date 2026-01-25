"""
Visualization utilities for regression analysis in Rainbow Pipeline.

Provides plotting functions for:
- Target distribution histograms
- Soft vs hard target comparisons
- Album balance charts
- Calibration curves
- Ontological state visualization
- Prediction intervals
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from training.core.soft_targets import SoftTargets

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Rainbow Table color scheme
ALBUM_COLORS = {
    "Red": "#FF0000",
    "Orange": "#FF8000",
    "Yellow": "#FFD700",
    "Green": "#00FF00",
    "Blue": "#0000FF",
    "Indigo": "#4B0082",
    "Violet": "#8B00FF",
    "White": "#FFFFFF",
    "Black": "#000000",
}

DIMENSION_COLORS = {
    "temporal": ["#FFB3BA", "#BAFFC9", "#BAE1FF"],  # past, present, future
    "spatial": ["#FFE4B5", "#98FB98", "#DDA0DD"],  # thing, place, person
    "ontological": ["#FFC0CB", "#E6E6FA", "#F0E68C"],  # imagined, forgotten, known
}

DIMENSION_LABELS = {
    "temporal": ["Past", "Present", "Future"],
    "spatial": ["Thing", "Place", "Person"],
    "ontological": ["Imagined", "Forgotten", "Known"],
}


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def plot_target_distribution(
    targets: np.ndarray,
    dimension: str,
    title: Optional[str] = None,
    ax: Optional["plt.Axes"] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> "plt.Figure":
    """
    Plot histogram of target distribution for a dimension.

    Args:
        targets: Array of shape (n_samples, 3) with softmax probabilities
        dimension: One of 'temporal', 'spatial', 'ontological'
        title: Optional custom title
        ax: Optional matplotlib axes
        figsize: Figure size if creating new figure

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    labels = DIMENSION_LABELS.get(dimension, ["Class 0", "Class 1", "Class 2"])
    colors = DIMENSION_COLORS.get(dimension, ["#FF6B6B", "#4ECDC4", "#45B7D1"])

    # Plot histograms for each class probability
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.hist(
            targets[:, i],
            bins=50,
            alpha=0.6,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Probability", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title or f"{dimension.title()} Mode Distribution", fontsize=14)
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    return fig


def plot_soft_vs_hard_targets(
    soft_targets: np.ndarray,
    hard_labels: np.ndarray,
    dimension: str,
    sample_indices: Optional[List[int]] = None,
    n_samples: int = 20,
    ax: Optional["plt.Axes"] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> "plt.Figure":
    """
    Compare soft targets with hard (one-hot) labels.

    Args:
        soft_targets: Array of shape (n_samples, 3) with soft probabilities
        hard_labels: Array of shape (n_samples,) with class indices
        dimension: Dimension name for labeling
        sample_indices: Specific sample indices to plot
        n_samples: Number of samples to show if indices not specified
        ax: Optional matplotlib axes
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    if sample_indices is None:
        sample_indices = list(range(min(n_samples, len(soft_targets))))

    n = len(sample_indices)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    labels = DIMENSION_LABELS.get(dimension, ["Class 0", "Class 1", "Class 2"])
    colors = DIMENSION_COLORS.get(dimension, ["#FF6B6B", "#4ECDC4", "#45B7D1"])

    x = np.arange(n)
    bar_width = 0.35

    # Convert hard labels to one-hot
    hard_one_hot = np.zeros((len(hard_labels), 3))
    for i, label in enumerate(hard_labels):
        if 0 <= label < 3:
            hard_one_hot[i, int(label)] = 1.0

    # Stack bars for each class
    bottom_soft = np.zeros(n)
    bottom_hard = np.zeros(n)

    for i, (label, color) in enumerate(zip(labels, colors)):
        soft_vals = soft_targets[sample_indices, i]
        hard_vals = hard_one_hot[sample_indices, i]

        ax.bar(
            x - bar_width / 2,
            soft_vals,
            bar_width,
            bottom=bottom_soft,
            label=f"Soft: {label}",
            color=color,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.bar(
            x + bar_width / 2,
            hard_vals,
            bar_width,
            bottom=bottom_hard,
            label=f"Hard: {label}",
            color=color,
            alpha=0.3,
            edgecolor="black",
            linewidth=0.5,
            hatch="//",
        )

        bottom_soft += soft_vals
        bottom_hard += hard_vals

    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(f"Soft vs Hard Targets: {dimension.title()} Mode", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(sample_indices, rotation=45, ha="right")
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    return fig


def plot_album_balance(
    album_labels: List[str],
    title: Optional[str] = None,
    ax: Optional["plt.Axes"] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> "plt.Figure":
    """
    Plot album distribution showing class balance.

    Args:
        album_labels: List of album names for each sample
        title: Optional custom title
        ax: Optional matplotlib axes
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # Count albums
    album_counts = {}
    for album in album_labels:
        album_counts[album] = album_counts.get(album, 0) + 1

    # Sort by Rainbow order
    rainbow_order = [
        "Red",
        "Orange",
        "Yellow",
        "Green",
        "Blue",
        "Indigo",
        "Violet",
        "White",
        "Black",
    ]
    sorted_albums = sorted(
        album_counts.keys(),
        key=lambda x: rainbow_order.index(x) if x in rainbow_order else 999,
    )

    counts = [album_counts[a] for a in sorted_albums]
    colors = [ALBUM_COLORS.get(a, "#808080") for a in sorted_albums]

    bars = ax.bar(sorted_albums, counts, color=colors, edgecolor="black", linewidth=1)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(
            f"{count}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Album", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title or "Album Distribution", fontsize=14)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    # Add percentage annotations
    total = sum(counts)
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        if pct >= 5:  # Only show if >= 5%
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white" if bar.get_facecolor()[0] < 0.5 else "black",
            )

    return fig


def plot_calibration_curve(
    predicted_probs: np.ndarray,
    true_labels: np.ndarray,
    n_bins: int = 10,
    dimension: Optional[str] = None,
    ax: Optional["plt.Axes"] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> "plt.Figure":
    """
    Plot reliability diagram (calibration curve).

    Args:
        predicted_probs: Predicted probabilities for positive class
        true_labels: Binary true labels (0 or 1)
        n_bins: Number of bins for calibration
        dimension: Optional dimension name for title
        ax: Optional matplotlib axes
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # Compute calibration curve
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mean_predicted = []
    fraction_positive = []
    bin_counts = []

    for i in range(n_bins):
        mask = (predicted_probs >= bin_edges[i]) & (predicted_probs < bin_edges[i + 1])
        if mask.sum() > 0:
            mean_predicted.append(predicted_probs[mask].mean())
            fraction_positive.append(true_labels[mask].mean())
            bin_counts.append(mask.sum())
        else:
            mean_predicted.append(bin_centers[i])
            fraction_positive.append(np.nan)
            bin_counts.append(0)

    mean_predicted = np.array(mean_predicted)
    fraction_positive = np.array(fraction_positive)

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", linewidth=2)

    # Plot actual calibration
    valid_mask = ~np.isnan(fraction_positive)
    ax.plot(
        mean_predicted[valid_mask],
        fraction_positive[valid_mask],
        "o-",
        color="#FF6B6B",
        label="Model",
        linewidth=2,
        markersize=8,
    )

    # Add histogram of predictions at bottom
    ax2 = ax.twinx()
    ax2.hist(
        predicted_probs, bins=n_bins, alpha=0.3, color="#4ECDC4", edgecolor="black"
    )
    ax2.set_ylabel("Count", fontsize=10, color="#4ECDC4")
    ax2.tick_params(axis="y", labelcolor="#4ECDC4")
    ax2.set_ylim(0, ax2.get_ylim()[1] * 3)  # Make histogram shorter

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    title = "Calibration Curve"
    if dimension:
        title += f": {dimension.title()}"
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Compute and display ECE
    valid_bins = bin_counts > 0
    weights = np.array(bin_counts)[valid_bins] / sum(bin_counts)
    ece = np.sum(
        weights
        * np.abs(
            np.array(mean_predicted)[valid_bins]
            - np.array(fraction_positive)[valid_bins]
        )
    )
    ax.text(
        0.05,
        0.95,
        f"ECE = {ece:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    return fig


def plot_ontological_space(
    temporal: np.ndarray,
    spatial: np.ndarray,
    ontological: np.ndarray,
    album_labels: Optional[List[str]] = None,
    method: str = "pca",
    ax: Optional["plt.Axes"] = None,
    figsize: Tuple[int, int] = (10, 10),
) -> "plt.Figure":
    """
    Visualize samples in reduced ontological space.

    Args:
        temporal: Temporal probabilities (n_samples, 3)
        spatial: Spatial probabilities (n_samples, 3)
        ontological: Ontological probabilities (n_samples, 3)
        album_labels: Optional album labels for coloring
        method: Reduction method ('pca', 'tsne', 'umap')
        ax: Optional matplotlib axes
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # Concatenate all dimensions
    combined = np.concatenate([temporal, spatial, ontological], axis=1)

    # Dimensionality reduction
    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(combined)
        ax.set_xlabel(
            f"PC1 ({reducer.explained_variance_ratio_[0]:.1%} var)", fontsize=12
        )
        ax.set_ylabel(
            f"PC2 ({reducer.explained_variance_ratio_[1]:.1%} var)", fontsize=12
        )
    elif method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, perplexity=min(30, len(combined) - 1))
        reduced = reducer.fit_transform(combined)
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
    else:
        # Default to simple mean projection
        reduced = np.column_stack(
            [
                temporal[:, 2] - temporal[:, 0],  # future vs past
                ontological[:, 2] - ontological[:, 0],  # known vs imagined
            ]
        )
        ax.set_xlabel("Temporal: Future ← → Past", fontsize=12)
        ax.set_ylabel("Ontological: Known ← → Imagined", fontsize=12)

    # Plot points
    if album_labels is not None:
        unique_albums = list(set(album_labels))
        for album in unique_albums:
            mask = [a == album for a in album_labels]
            color = ALBUM_COLORS.get(album, "#808080")
            ax.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                c=color,
                label=album,
                alpha=0.6,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )
        ax.legend(loc="best", ncol=2)
    else:
        ax.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c="#4ECDC4",
            alpha=0.6,
            s=50,
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_title("Ontological Space Visualization", fontsize=14)
    ax.grid(True, alpha=0.3)

    return fig


def plot_prediction_intervals(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    true_values: np.ndarray,
    sample_indices: Optional[List[int]] = None,
    n_samples: int = 30,
    confidence: float = 0.95,
    ax: Optional["plt.Axes"] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> "plt.Figure":
    """
    Plot predictions with uncertainty intervals vs true values.

    Args:
        predictions: Predicted values
        uncertainties: Predicted standard deviations
        true_values: Ground truth values
        sample_indices: Specific samples to plot
        n_samples: Number of samples if indices not specified
        confidence: Confidence level for intervals
        ax: Optional matplotlib axes
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    if sample_indices is None:
        sample_indices = list(range(min(n_samples, len(predictions))))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    n = len(sample_indices)
    x = np.arange(n)

    pred = predictions[sample_indices]
    unc = uncertainties[sample_indices]
    true = true_values[sample_indices]

    # Compute confidence interval multiplier
    from scipy.stats import norm

    z = norm.ppf((1 + confidence) / 2)

    # Plot prediction intervals
    ax.fill_between(
        x,
        pred - z * unc,
        pred + z * unc,
        alpha=0.3,
        color="#4ECDC4",
        label=f"{confidence:.0%} CI",
    )

    # Plot predictions
    ax.plot(x, pred, "o-", color="#4ECDC4", label="Prediction", markersize=6)

    # Plot true values
    ax.scatter(x, true, color="#FF6B6B", marker="x", s=100, label="True", zorder=5)

    # Highlight points outside interval
    outside = (true < pred - z * unc) | (true > pred + z * unc)
    if outside.any():
        ax.scatter(
            x[outside],
            true[outside],
            facecolors="none",
            edgecolors="red",
            s=200,
            linewidth=2,
            label="Outside CI",
        )

    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Predictions with Uncertainty Intervals", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(sample_indices, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add coverage statistic
    coverage = 1 - outside.mean()
    ax.text(
        0.02,
        0.98,
        f"Coverage: {coverage:.1%}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    return fig


def plot_per_dimension_metrics(
    metrics: Dict[str, Dict[str, float]],
    metric_names: Optional[List[str]] = None,
    ax: Optional["plt.Axes"] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> "plt.Figure":
    """
    Plot metrics comparison across dimensions.

    Args:
        metrics: Dict mapping dimension -> {metric_name: value}
        metric_names: Specific metrics to plot (defaults to all)
        ax: Optional matplotlib axes
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    dimensions = list(metrics.keys())
    if metric_names is None:
        metric_names = list(metrics[dimensions[0]].keys())

    n_dims = len(dimensions)
    n_metrics = len(metric_names)
    x = np.arange(n_dims)
    bar_width = 0.8 / n_metrics

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_metrics))

    for i, metric in enumerate(metric_names):
        values = [metrics[dim].get(metric, 0) for dim in dimensions]
        offset = (i - n_metrics / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            values,
            bar_width,
            label=metric,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
        )

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )

    ax.set_xlabel("Dimension", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("Per-Dimension Metrics Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in dimensions])
    ax.legend(loc="upper right", ncol=min(3, n_metrics))
    ax.grid(True, alpha=0.3, axis="y")

    return fig


@dataclass
class RegressionVisualizer:
    """
    Comprehensive visualizer for regression analysis.

    Provides a unified interface for generating all regression-related
    visualizations from training results.
    """

    output_dir: str = "./visualizations"
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 150
    style: str = "seaborn-v0_8-whitegrid"

    def __post_init__(self):
        """Initialize visualizer."""
        _check_matplotlib()
        import os

        os.makedirs(self.output_dir, exist_ok=True)

        try:
            plt.style.use(self.style)
        except OSError:
            pass  # Style not available, use default

    def plot_all_distributions(
        self,
        temporal: np.ndarray,
        spatial: np.ndarray,
        ontological: np.ndarray,
        save: bool = True,
    ) -> Dict[str, "plt.Figure"]:
        """
        Generate distribution plots for all dimensions.

        Args:
            temporal: Temporal targets (n_samples, 3)
            spatial: Spatial targets (n_samples, 3)
            ontological: Ontological targets (n_samples, 3)
            save: Whether to save figures

        Returns:
            Dict mapping dimension name to Figure
        """
        figures = {}

        for name, data in [
            ("temporal", temporal),
            ("spatial", spatial),
            ("ontological", ontological),
        ]:
            fig = plot_target_distribution(data, name, figsize=self.figsize)
            figures[name] = fig

            if save:
                fig.savefig(
                    f"{self.output_dir}/{name}_distribution.png",
                    dpi=self.dpi,
                    bbox_inches="tight",
                )

        return figures

    def plot_training_summary(
        self,
        train_losses: List[float],
        val_losses: List[float],
        metrics_history: Dict[str, List[float]],
        save: bool = True,
    ) -> "plt.Figure":
        """
        Generate training summary visualization.

        Args:
            train_losses: Training loss per epoch
            val_losses: Validation loss per epoch
            metrics_history: Dict of metric name -> values per epoch
            save: Whether to save figure

        Returns:
            Figure with training curves
        """
        n_metrics = len(metrics_history) + 1
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        # Loss plot
        epochs = range(1, len(train_losses) + 1)
        axes[0].plot(epochs, train_losses, "b-", label="Train", linewidth=2)
        axes[0].plot(epochs, val_losses, "r-", label="Validation", linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Metric plots
        for ax, (name, values) in zip(axes[1:], metrics_history.items()):
            ax.plot(epochs[: len(values)], values, "g-", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(name)
            ax.set_title(name.replace("_", " ").title())
            ax.grid(True, alpha=0.3)

        fig.tight_layout()

        if save:
            fig.savefig(
                f"{self.output_dir}/training_summary.png",
                dpi=self.dpi,
                bbox_inches="tight",
            )

        return fig

    def generate_full_report(
        self,
        soft_targets: "SoftTargets",
        predictions: Optional[Dict[str, np.ndarray]] = None,
        album_labels: Optional[List[str]] = None,
        hard_labels: Optional[Dict[str, np.ndarray]] = None,
        save: bool = True,
    ) -> Dict[str, "plt.Figure"]:
        """
        Generate comprehensive visualization report.

        Args:
            soft_targets: SoftTargets object with all target arrays
            predictions: Optional dict with predicted distributions
            album_labels: Optional album labels
            hard_labels: Optional dict with discrete labels per dimension
            save: Whether to save figures

        Returns:
            Dict of all generated figures
        """
        figures = {}

        # Distribution plots
        dist_figs = self.plot_all_distributions(
            soft_targets.temporal,
            soft_targets.spatial,
            soft_targets.ontological,
            save=save,
        )
        figures.update({f"dist_{k}": v for k, v in dist_figs.items()})

        # Album balance
        if album_labels is not None:
            fig = plot_album_balance(album_labels, figsize=self.figsize)
            figures["album_balance"] = fig
            if save:
                fig.savefig(
                    f"{self.output_dir}/album_balance.png",
                    dpi=self.dpi,
                    bbox_inches="tight",
                )

        # Soft vs hard comparison
        if hard_labels is not None:
            for dim in ["temporal", "spatial", "ontological"]:
                if dim in hard_labels:
                    soft = getattr(soft_targets, dim)
                    hard = hard_labels[dim]
                    fig = plot_soft_vs_hard_targets(
                        soft, hard, dim, figsize=self.figsize
                    )
                    figures[f"soft_vs_hard_{dim}"] = fig
                    if save:
                        fig.savefig(
                            f"{self.output_dir}/soft_vs_hard_{dim}.png",
                            dpi=self.dpi,
                            bbox_inches="tight",
                        )

        # Ontological space
        fig = plot_ontological_space(
            soft_targets.temporal,
            soft_targets.spatial,
            soft_targets.ontological,
            album_labels=album_labels,
            figsize=self.figsize,
        )
        figures["ontological_space"] = fig
        if save:
            fig.savefig(
                f"{self.output_dir}/ontological_space.png",
                dpi=self.dpi,
                bbox_inches="tight",
            )

        return figures

    def close_all(self):
        """Close all open figures."""
        plt.close("all")


if __name__ == "__main__":
    # Demo visualization
    print("Generating demo visualizations...")

    np.random.seed(42)
    n_samples = 100

    # Generate synthetic soft targets
    temporal = np.random.dirichlet([2, 1, 1], n_samples)
    spatial = np.random.dirichlet([1, 2, 1], n_samples)
    ontological = np.random.dirichlet([1, 1, 2], n_samples)

    # Generate album labels
    albums = [
        "Red",
        "Orange",
        "Yellow",
        "Green",
        "Blue",
        "Indigo",
        "Violet",
        "White",
        "Black",
    ]
    album_labels = np.random.choice(
        albums, n_samples, p=[0.15, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05]
    ).tolist()

    # Create visualizer
    viz = RegressionVisualizer(output_dir="./demo_visualizations")

    # Generate distribution plots
    print("Plotting target distributions...")
    fig = plot_target_distribution(temporal, "temporal")
    plt.close(fig)

    # Plot album balance
    print("Plotting album balance...")
    fig = plot_album_balance(album_labels)
    plt.close(fig)

    # Plot ontological space
    print("Plotting ontological space...")
    fig = plot_ontological_space(temporal, spatial, ontological, album_labels)
    plt.close(fig)

    # Plot calibration curve
    print("Plotting calibration curve...")
    pred_probs = np.random.beta(2, 2, n_samples)
    true_labels = (np.random.rand(n_samples) < pred_probs).astype(int)
    fig = plot_calibration_curve(pred_probs, true_labels)
    plt.close(fig)

    # Plot prediction intervals
    print("Plotting prediction intervals...")
    predictions = np.random.randn(n_samples) * 0.5 + 0.5
    uncertainties = np.abs(np.random.randn(n_samples) * 0.1) + 0.05
    true_values = predictions + np.random.randn(n_samples) * uncertainties * 1.5
    fig = plot_prediction_intervals(predictions, uncertainties, true_values)
    plt.close(fig)

    print("Demo complete!")
