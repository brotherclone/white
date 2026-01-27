"""
Loss functions for continuous regression targets.

Includes:
- Standard regression losses (MSE, Huber, Smooth L1)
- Distribution losses (KL divergence for softmax outputs)
- Combined multi-task losses
- Per-target weighting support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Literal
from dataclasses import dataclass


@dataclass
class RegressionLossOutput:
    """Container for regression loss components."""

    total_loss: torch.Tensor
    per_target_losses: Dict[str, torch.Tensor]

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary of scalar values."""
        return {
            "total_loss": self.total_loss.item(),
            **{f"{k}_loss": v.item() for k, v in self.per_target_losses.items()},
        }


class MSELoss(nn.Module):
    """
    Mean Squared Error loss for regression.

    loss = (1/n) * Σ(predicted - target)²
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute MSE loss.

        Args:
            predictions: Predicted values [batch, num_targets]
            targets: Target values [batch, num_targets]
            weights: Optional per-sample weights [batch]

        Returns:
            Loss tensor (scalar if reduction="mean")
        """
        squared_error = (predictions - targets) ** 2

        if weights is not None:
            # Expand weights for broadcasting
            weights = weights.unsqueeze(-1).expand_as(squared_error)
            squared_error = squared_error * weights

        if self.reduction == "mean":
            return squared_error.mean()
        elif self.reduction == "sum":
            return squared_error.sum()
        else:  # none
            return squared_error


class HuberLoss(nn.Module):
    """
    Huber loss (smooth combination of L1 and L2).

    Robust to outliers. For small errors, behaves like MSE.
    For large errors, behaves like MAE (less sensitive to outliers).

    loss = 0.5 * (y - ŷ)² if |y - ŷ| < δ
         = δ * |y - ŷ| - 0.5 * δ² otherwise
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Huber loss.

        Args:
            predictions: Predicted values [batch, num_targets]
            targets: Target values [batch, num_targets]
            weights: Optional per-sample weights [batch]

        Returns:
            Loss tensor
        """
        diff = predictions - targets
        abs_diff = torch.abs(diff)

        # Quadratic for small errors, linear for large
        quadratic = 0.5 * diff**2
        linear = self.delta * abs_diff - 0.5 * self.delta**2

        loss = torch.where(abs_diff <= self.delta, quadratic, linear)

        if weights is not None:
            weights = weights.unsqueeze(-1).expand_as(loss)
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 loss (same as Huber with delta=1).

    Commonly used in object detection (e.g., Faster R-CNN).
    """

    def __init__(self, beta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Smooth L1 loss.

        Args:
            predictions: Predicted values [batch, num_targets]
            targets: Target values [batch, num_targets]
            weights: Optional per-sample weights [batch]

        Returns:
            Loss tensor
        """
        loss = F.smooth_l1_loss(predictions, targets, reduction="none", beta=self.beta)

        if weights is not None:
            weights = weights.unsqueeze(-1).expand_as(loss)
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class KLDivergenceLoss(nn.Module):
    """
    KL Divergence loss for probability distributions.

    KL(target || prediction) = Σ target * log(target / prediction)

    Best for soft targets that represent probability distributions
    (e.g., ontological mode distributions).
    """

    def __init__(self, reduction: str = "batchmean", log_target: bool = False):
        """
        Initialize KL divergence loss.

        Args:
            reduction: Reduction method ("batchmean", "mean", "sum", "none")
            log_target: If True, target is already in log space
        """
        super().__init__()
        self.reduction = reduction
        self.log_target = log_target

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute KL divergence loss.

        Args:
            predictions: Log probabilities [batch, num_classes]
            targets: Target probabilities [batch, num_classes]
            weights: Optional per-sample weights [batch]

        Returns:
            Loss tensor
        """
        # Ensure predictions are log probabilities
        if not self.log_target:
            # Apply log_softmax to predictions if they're logits
            log_predictions = F.log_softmax(predictions, dim=-1)
        else:
            log_predictions = predictions

        loss = F.kl_div(
            log_predictions,
            targets,
            reduction="none",
            log_target=self.log_target,
        )

        # KL div returns [batch, num_classes], sum over classes
        loss = loss.sum(dim=-1)  # [batch]

        if weights is not None:
            loss = loss * weights

        if self.reduction == "batchmean":
            return loss.mean()
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class DistributionMSELoss(nn.Module):
    """
    MSE loss for probability distributions.

    Alternative to KL divergence that doesn't require log probabilities.
    Works well when targets may contain zeros.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute MSE between probability distributions.

        Args:
            predictions: Logits or probabilities [batch, num_classes]
            targets: Target probabilities [batch, num_classes]
            weights: Optional per-sample weights [batch]

        Returns:
            Loss tensor
        """
        # Apply softmax to get probabilities
        probs = F.softmax(predictions, dim=-1)

        loss = (probs - targets) ** 2

        # Sum over classes, then reduce over batch
        loss = loss.sum(dim=-1)  # [batch]

        if weights is not None:
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class PerTargetWeightedLoss(nn.Module):
    """
    Weighted combination of losses for multiple targets.

    Allows different loss functions and weights per target.
    """

    def __init__(
        self,
        target_configs: Dict[str, Dict],
        default_loss: str = "mse",
        default_weight: float = 1.0,
    ):
        """
        Initialize per-target weighted loss.

        Args:
            target_configs: Configuration per target:
                {
                    "intensity": {"loss": "mse", "weight": 1.0},
                    "temporal": {"loss": "kl_div", "weight": 0.8},
                }
            default_loss: Default loss type if not specified
            default_weight: Default weight if not specified
        """
        super().__init__()

        self.target_configs = target_configs
        self.default_loss = default_loss
        self.default_weight = default_weight

        # Create loss functions
        self.loss_fns = nn.ModuleDict()
        self.weights = {}

        for name, config in target_configs.items():
            loss_type = config.get("loss", default_loss)
            weight = config.get("weight", default_weight)

            if loss_type == "mse":
                self.loss_fns[name] = MSELoss(reduction="mean")
            elif loss_type == "huber":
                delta = config.get("delta", 1.0)
                self.loss_fns[name] = HuberLoss(delta=delta, reduction="mean")
            elif loss_type == "smooth_l1":
                beta = config.get("beta", 1.0)
                self.loss_fns[name] = SmoothL1Loss(beta=beta, reduction="mean")
            elif loss_type == "kl_div":
                self.loss_fns[name] = KLDivergenceLoss(reduction="batchmean")
            elif loss_type == "dist_mse":
                self.loss_fns[name] = DistributionMSELoss(reduction="mean")
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            self.weights[name] = weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        sample_weights: Optional[torch.Tensor] = None,
    ) -> RegressionLossOutput:
        """
        Compute weighted combination of per-target losses.

        Args:
            predictions: Dictionary of predictions per target
            targets: Dictionary of targets per target
            sample_weights: Optional per-sample weights [batch]

        Returns:
            RegressionLossOutput with total and per-target losses
        """
        per_target_losses = {}
        total_loss = 0.0

        for name, loss_fn in self.loss_fns.items():
            if name not in predictions or name not in targets:
                continue

            pred = predictions[name]
            target = targets[name]

            loss = loss_fn(pred, target, sample_weights)
            per_target_losses[name] = loss

            total_loss = total_loss + self.weights[name] * loss

        return RegressionLossOutput(
            total_loss=total_loss,
            per_target_losses=per_target_losses,
        )


class OntologicalRegressionLoss(nn.Module):
    """
    Combined loss for Rainbow Table ontological regression.

    Handles:
    - Temporal distribution (3 softmax) - KL divergence
    - Spatial distribution (3 softmax) - KL divergence
    - Ontological distribution (3 softmax) - KL divergence
    - Chromatic confidence (1 sigmoid) - BCE
    """

    def __init__(
        self,
        temporal_weight: float = 0.8,
        spatial_weight: float = 0.8,
        ontological_weight: float = 0.8,
        confidence_weight: float = 0.5,
        distribution_loss: Literal["kl_div", "mse", "smooth_l1"] = "kl_div",
    ):
        """
        Initialize ontological regression loss.

        Args:
            temporal_weight: Weight for temporal distribution loss
            spatial_weight: Weight for spatial distribution loss
            ontological_weight: Weight for ontological distribution loss
            confidence_weight: Weight for confidence loss
            distribution_loss: Loss type for distributions
        """
        super().__init__()

        self.weights = {
            "temporal": temporal_weight,
            "spatial": spatial_weight,
            "ontological": ontological_weight,
            "confidence": confidence_weight,
        }

        self.distribution_loss = distribution_loss

        if distribution_loss == "kl_div":
            self.dist_loss_fn = KLDivergenceLoss(reduction="batchmean")
        elif distribution_loss == "mse":
            self.dist_loss_fn = DistributionMSELoss(reduction="mean")
        else:
            self.dist_loss_fn = lambda p, t, w=None: F.smooth_l1_loss(
                F.softmax(p, dim=-1), t, reduction="mean"
            )

    def forward(
        self,
        temporal_logits: torch.Tensor,
        spatial_logits: torch.Tensor,
        ontological_logits: torch.Tensor,
        confidence_pred: torch.Tensor,
        temporal_targets: torch.Tensor,
        spatial_targets: torch.Tensor,
        ontological_targets: torch.Tensor,
        confidence_targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> RegressionLossOutput:
        """
        Compute combined ontological regression loss.

        Args:
            temporal_logits: Temporal logits [batch, 3]
            spatial_logits: Spatial logits [batch, 3]
            ontological_logits: Ontological logits [batch, 3]
            confidence_pred: Confidence predictions [batch, 1]
            temporal_targets: Temporal soft targets [batch, 3]
            spatial_targets: Spatial soft targets [batch, 3]
            ontological_targets: Ontological soft targets [batch, 3]
            confidence_targets: Confidence targets [batch, 1]
            sample_weights: Optional per-sample weights [batch]

        Returns:
            RegressionLossOutput with all loss components
        """
        # Distribution losses
        temporal_loss = self.dist_loss_fn(
            temporal_logits, temporal_targets, sample_weights
        )
        spatial_loss = self.dist_loss_fn(
            spatial_logits, spatial_targets, sample_weights
        )
        ontological_loss = self.dist_loss_fn(
            ontological_logits, ontological_targets, sample_weights
        )

        # Confidence loss (BCE)
        if sample_weights is not None:
            # Weighted BCE
            bce = F.binary_cross_entropy(
                confidence_pred, confidence_targets, reduction="none"
            )
            confidence_loss = (bce.squeeze() * sample_weights).mean()
        else:
            confidence_loss = F.binary_cross_entropy(
                confidence_pred, confidence_targets, reduction="mean"
            )

        # Combined loss
        total_loss = (
            self.weights["temporal"] * temporal_loss
            + self.weights["spatial"] * spatial_loss
            + self.weights["ontological"] * ontological_loss
            + self.weights["confidence"] * confidence_loss
        )

        return RegressionLossOutput(
            total_loss=total_loss,
            per_target_losses={
                "temporal": temporal_loss,
                "spatial": spatial_loss,
                "ontological": ontological_loss,
                "confidence": confidence_loss,
            },
        )


class CombinedClassificationRegressionLoss(nn.Module):
    """
    Combined loss for multitask classification and regression.

    total_loss = α * classification_loss + β * regression_loss
    """

    def __init__(
        self,
        classification_weight: float = 1.0,
        regression_weight: float = 1.0,
        num_classes: int = 8,
        multi_label: bool = False,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize combined loss.

        Args:
            classification_weight: Weight for classification loss
            regression_weight: Weight for total regression loss
            num_classes: Number of classification classes
            multi_label: If True, use BCE for classification
            label_smoothing: Label smoothing for cross-entropy
        """
        super().__init__()

        self.classification_weight = classification_weight
        self.regression_weight = regression_weight

        if multi_label:
            self.cls_loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.reg_loss_fn = OntologicalRegressionLoss()

    def forward(
        self,
        classification_logits: torch.Tensor,
        classification_targets: torch.Tensor,
        temporal_logits: torch.Tensor,
        spatial_logits: torch.Tensor,
        ontological_logits: torch.Tensor,
        confidence_pred: torch.Tensor,
        temporal_targets: torch.Tensor,
        spatial_targets: torch.Tensor,
        ontological_targets: torch.Tensor,
        confidence_targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined classification + regression loss.

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Classification loss
        cls_loss = self.cls_loss_fn(classification_logits, classification_targets)

        # Regression loss
        reg_output = self.reg_loss_fn(
            temporal_logits,
            spatial_logits,
            ontological_logits,
            confidence_pred,
            temporal_targets,
            spatial_targets,
            ontological_targets,
            confidence_targets,
            sample_weights,
        )

        # Combined loss
        total_loss = (
            self.classification_weight * cls_loss
            + self.regression_weight * reg_output.total_loss
        )

        loss_components = {
            "total": total_loss,
            "classification": cls_loss,
            "regression_total": reg_output.total_loss,
            **{f"regression_{k}": v for k, v in reg_output.per_target_losses.items()},
        }

        return total_loss, loss_components


if __name__ == "__main__":
    # Quick tests
    print("Testing regression losses...")

    batch_size = 4
    num_targets = 3

    predictions = torch.randn(batch_size, num_targets)
    targets = torch.randn(batch_size, num_targets)
    weights = torch.rand(batch_size)

    # Test MSE
    print("\n=== MSE Loss ===")
    mse = MSELoss()
    loss = mse(predictions, targets)
    print(f"MSE loss: {loss.item():.4f}")

    loss_weighted = mse(predictions, targets, weights)
    print(f"Weighted MSE loss: {loss_weighted.item():.4f}")

    # Test Huber
    print("\n=== Huber Loss ===")
    huber = HuberLoss(delta=1.0)
    loss = huber(predictions, targets)
    print(f"Huber loss: {loss.item():.4f}")

    # Test KL Divergence
    print("\n=== KL Divergence ===")
    probs_pred = F.softmax(torch.randn(batch_size, 3), dim=-1)
    probs_target = F.softmax(torch.randn(batch_size, 3), dim=-1)

    kl = KLDivergenceLoss()
    loss = kl(torch.log(probs_pred), probs_target)
    print(f"KL divergence: {loss.item():.4f}")

    # Test Ontological Regression Loss
    print("\n=== Ontological Regression Loss ===")
    ont_loss = OntologicalRegressionLoss()

    temporal_logits = torch.randn(batch_size, 3)
    spatial_logits = torch.randn(batch_size, 3)
    ontological_logits = torch.randn(batch_size, 3)
    confidence_pred = torch.sigmoid(torch.randn(batch_size, 1))

    temporal_targets = F.softmax(torch.randn(batch_size, 3), dim=-1)
    spatial_targets = F.softmax(torch.randn(batch_size, 3), dim=-1)
    ontological_targets = F.softmax(torch.randn(batch_size, 3), dim=-1)
    confidence_targets = torch.rand(batch_size, 1)

    output = ont_loss(
        temporal_logits,
        spatial_logits,
        ontological_logits,
        confidence_pred,
        temporal_targets,
        spatial_targets,
        ontological_targets,
        confidence_targets,
    )

    print(f"Total loss: {output.total_loss.item():.4f}")
    for name, loss in output.per_target_losses.items():
        print(f"  {name}: {loss.item():.4f}")

    # Test Per-Target Weighted Loss
    print("\n=== Per-Target Weighted Loss ===")
    pt_loss = PerTargetWeightedLoss(
        {
            "intensity": {"loss": "mse", "weight": 1.0},
            "fluidity": {"loss": "huber", "weight": 0.8, "delta": 0.5},
        }
    )

    preds = {
        "intensity": torch.randn(batch_size, 1),
        "fluidity": torch.randn(batch_size, 1),
    }
    targs = {
        "intensity": torch.randn(batch_size, 1),
        "fluidity": torch.randn(batch_size, 1),
    }

    output = pt_loss(preds, targs)
    print(f"Total loss: {output.total_loss.item():.4f}")
    for name, loss in output.per_target_losses.items():
        print(f"  {name}: {loss.item():.4f}")

    print("\n✓ All loss tests passed!")
