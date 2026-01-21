"""
Evaluation metrics for multi-class rebracketing classification.

Provides per-class and aggregate metrics including:
- Per-class F1, Precision, Recall
- Macro/Micro/Weighted averaging
- Confusion matrices
- Top-k accuracy
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns


class MultiClassMetrics:
    """Compute and track multi-class classification metrics."""

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize metrics tracker.

        Args:
            num_classes: Number of classes
            class_names: Optional list of class names for visualization
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]

        self.reset()

    def reset(self):
        """Reset all tracked values."""
        self.all_predictions = []
        self.all_labels = []

    def update(self, predictions: torch.Tensor, labels: torch.Tensor):
        """
        Update metrics with new batch.

        Args:
            predictions: Predicted class indices [batch] or logits [batch, num_classes]
            labels: True class indices [batch]
        """
        # Convert to numpy
        if predictions.dim() > 1:
            # Logits: take argmax
            predictions = torch.argmax(predictions, dim=-1)

        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        self.all_predictions.extend(predictions.tolist())
        self.all_labels.extend(labels.tolist())

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary of metric names to values
        """
        if not self.all_predictions:
            return {}

        preds = np.array(self.all_predictions)
        labels = np.array(self.all_labels)

        metrics = {}

        # Overall accuracy
        metrics["accuracy"] = accuracy_score(labels, preds)

        # Per-class metrics
        per_class_precision = precision_score(
            labels, preds, average=None, zero_division=0
        )
        per_class_recall = recall_score(labels, preds, average=None, zero_division=0)
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)

        for i, name in enumerate(self.class_names):
            metrics[f"precision_{name}"] = per_class_precision[i]
            metrics[f"recall_{name}"] = per_class_recall[i]
            metrics[f"f1_{name}"] = per_class_f1[i]

        # Aggregate metrics
        metrics["macro_precision"] = precision_score(
            labels, preds, average="macro", zero_division=0
        )
        metrics["macro_recall"] = recall_score(
            labels, preds, average="macro", zero_division=0
        )
        metrics["macro_f1"] = f1_score(labels, preds, average="macro", zero_division=0)

        metrics["micro_precision"] = precision_score(
            labels, preds, average="micro", zero_division=0
        )
        metrics["micro_recall"] = recall_score(
            labels, preds, average="micro", zero_division=0
        )
        metrics["micro_f1"] = f1_score(labels, preds, average="micro", zero_division=0)

        metrics["weighted_precision"] = precision_score(
            labels, preds, average="weighted", zero_division=0
        )
        metrics["weighted_recall"] = recall_score(
            labels, preds, average="weighted", zero_division=0
        )
        metrics["weighted_f1"] = f1_score(
            labels, preds, average="weighted", zero_division=0
        )

        return metrics

    def get_confusion_matrix(self, normalize: Optional[str] = None) -> np.ndarray:
        """
        Get confusion matrix.

        Args:
            normalize: {"true", "pred", "all"} or None

        Returns:
            Confusion matrix [num_classes, num_classes]
        """
        if not self.all_predictions:
            return np.zeros((self.num_classes, self.num_classes))

        preds = np.array(self.all_predictions)
        labels = np.array(self.all_labels)

        cm = confusion_matrix(labels, preds, labels=list(range(self.num_classes)))

        if normalize == "true":
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            cm = cm.astype("float") / cm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            cm = cm.astype("float") / cm.sum()

        return cm

    def plot_confusion_matrix(
        self,
        normalize: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Plot confusion matrix.

        Args:
            normalize: {"true", "pred", "all"} or None
            save_path: Optional path to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        cm = self.get_confusion_matrix(normalize=normalize)

        fig, ax = plt.subplots(figsize=figsize)

        # Use seaborn for nice heatmap
        fmt = ".2f" if normalize else "d"
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
            cbar_kws={"label": "Count" if not normalize else "Proportion"},
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        title = "Confusion Matrix"
        if normalize:
            title += f" (normalized by {normalize})"
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved confusion matrix to {save_path}")

        return fig

    def get_classification_report(self) -> str:
        """
        Get sklearn classification report.

        Returns:
            String classification report
        """
        if not self.all_predictions:
            return "No predictions yet"

        preds = np.array(self.all_predictions)
        labels = np.array(self.all_labels)

        return classification_report(
            labels,
            preds,
            target_names=self.class_names,
            digits=3,
            zero_division=0,
        )

    def get_most_confused_pairs(self, top_k: int = 5) -> List[Tuple[str, str, int]]:
        """
        Get most confused class pairs (excluding diagonal).

        Args:
            top_k: Number of pairs to return

        Returns:
            List of (class_i, class_j, count) tuples
        """
        cm = self.get_confusion_matrix(normalize=None)

        # Mask diagonal
        cm_no_diag = cm.copy()
        np.fill_diagonal(cm_no_diag, 0)

        # Find top-k confused pairs
        pairs = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and cm_no_diag[i, j] > 0:
                    pairs.append(
                        (
                            self.class_names[i],
                            self.class_names[j],
                            int(cm_no_diag[i, j]),
                        )
                    )

        # Sort by count
        pairs.sort(key=lambda x: x[2], reverse=True)

        return pairs[:top_k]


class MultiLabelMetrics:
    """Compute and track multi-label classification metrics."""

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        threshold: float = 0.5,
    ):
        """
        Initialize metrics tracker.

        Args:
            num_classes: Number of classes
            class_names: Optional list of class names
            threshold: Threshold for binary predictions
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.threshold = threshold

        self.reset()

    def reset(self):
        """Reset all tracked values."""
        self.all_predictions = []
        self.all_labels = []

    def update(self, predictions: torch.Tensor, labels: torch.Tensor):
        """
        Update metrics with new batch.

        Args:
            predictions: Predicted probabilities [batch, num_classes] (after sigmoid)
            labels: True binary labels [batch, num_classes]
        """
        # Apply sigmoid if not already
        if predictions.max() > 1.0 or predictions.min() < 0.0:
            predictions = torch.sigmoid(predictions)

        # Threshold to binary
        binary_preds = (predictions > self.threshold).long()

        self.all_predictions.append(binary_preds.cpu().numpy())
        self.all_labels.append(labels.cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary of metric names to values
        """
        if not self.all_predictions:
            return {}

        preds = np.vstack(self.all_predictions)
        labels = np.vstack(self.all_labels)

        metrics = {}

        # Hamming loss (fraction of incorrect labels)
        metrics["hamming_loss"] = np.mean(preds != labels)

        # Subset accuracy (exact match)
        metrics["subset_accuracy"] = np.mean(np.all(preds == labels, axis=1))

        # Per-class metrics
        for i, name in enumerate(self.class_names):
            metrics[f"f1_{name}"] = f1_score(labels[:, i], preds[:, i], zero_division=0)
            metrics[f"precision_{name}"] = precision_score(
                labels[:, i], preds[:, i], zero_division=0
            )
            metrics[f"recall_{name}"] = recall_score(
                labels[:, i], preds[:, i], zero_division=0
            )

        # Aggregate metrics (samples average)
        metrics["macro_f1"] = f1_score(labels, preds, average="macro", zero_division=0)
        metrics["micro_f1"] = f1_score(labels, preds, average="micro", zero_division=0)
        metrics["weighted_f1"] = f1_score(
            labels, preds, average="weighted", zero_division=0
        )

        return metrics


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 3) -> float:
    """
    Compute top-k accuracy.

    Args:
        logits: Model logits [batch, num_classes]
        labels: True labels [batch]
        k: Top-k parameter

    Returns:
        Top-k accuracy
    """
    # Get top-k predictions
    _, topk_preds = torch.topk(logits, k=k, dim=-1)

    # Check if true label is in top-k
    labels = labels.unsqueeze(-1).expand_as(topk_preds)
    correct = (topk_preds == labels).any(dim=-1)

    return correct.float().mean().item()


if __name__ == "__main__":
    # Quick test
    print("Testing MultiClassMetrics...")

    num_classes = 8
    class_names = [
        "spatial",
        "temporal",
        "causal",
        "perceptual",
        "memory",
        "ontological",
        "narrative",
        "identity",
    ]

    metrics = MultiClassMetrics(num_classes=num_classes, class_names=class_names)

    # Simulate predictions
    torch.manual_seed(42)
    for _ in range(10):
        logits = torch.randn(16, num_classes)
        labels = torch.randint(0, num_classes, (16,))
        metrics.update(logits, labels)

    # Compute metrics
    results = metrics.compute()
    print("\nMetrics:")
    for key, val in sorted(results.items()):
        print(f"  {key}: {val:.3f}")

    # Confusion matrix
    print("\nConfusion matrix:")
    cm = metrics.get_confusion_matrix()
    print(cm)

    # Most confused pairs
    print("\nMost confused pairs:")
    for cls1, cls2, count in metrics.get_most_confused_pairs(top_k=3):
        print(f"  {cls1} → {cls2}: {count}")

    # Classification report
    print("\nClassification report:")
    print(metrics.get_classification_report())

    print("\n✓ Tests passed!")
