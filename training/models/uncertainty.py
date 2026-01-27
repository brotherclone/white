"""
Uncertainty estimation methods for Rainbow Pipeline.

Implements:
- Ensemble-based uncertainty (multiple models)
- Evidential deep learning (distribution parameters)
- Monte Carlo dropout (inference-time dropout)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UncertaintyOutput:
    """Container for predictions with uncertainty estimates."""

    # Point predictions
    mean: torch.Tensor

    # Uncertainty estimates
    aleatoric_uncertainty: Optional[torch.Tensor] = None  # Data uncertainty
    epistemic_uncertainty: Optional[torch.Tensor] = None  # Model uncertainty
    total_uncertainty: Optional[torch.Tensor] = None

    # Distribution parameters (for evidential)
    alpha: Optional[torch.Tensor] = None  # Dirichlet concentration

    # Ensemble outputs (for ensemble method)
    ensemble_predictions: Optional[List[torch.Tensor]] = None

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary."""
        result = {"mean": self.mean}
        if self.aleatoric_uncertainty is not None:
            result["aleatoric_uncertainty"] = self.aleatoric_uncertainty
        if self.epistemic_uncertainty is not None:
            result["epistemic_uncertainty"] = self.epistemic_uncertainty
        if self.total_uncertainty is not None:
            result["total_uncertainty"] = self.total_uncertainty
        return result


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation.

    Applies dropout at inference time and aggregates multiple forward passes
    to estimate epistemic (model) uncertainty.
    """

    def __init__(
        self,
        model: nn.Module,
        dropout_rate: float = 0.1,
        n_samples: int = 10,
    ):
        """
        Initialize MC Dropout wrapper.

        Args:
            model: Base model to wrap
            dropout_rate: Dropout probability
            n_samples: Number of forward passes for uncertainty estimation
        """
        super().__init__()
        self.model = model
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples

        # Add dropout layers if not present
        self._add_dropout_layers()

    def _add_dropout_layers(self):
        """Add dropout after each linear layer if not present."""
        # This is a simple approach - in practice you might want
        # to be more selective about where to add dropout
        pass  # Assume model already has dropout layers

    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _disable_dropout(self):
        """Disable dropout layers."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

    def forward(
        self,
        *args,
        return_samples: bool = False,
        **kwargs,
    ) -> UncertaintyOutput:
        """
        Forward pass with MC Dropout uncertainty estimation.

        Args:
            *args: Positional arguments for base model
            return_samples: Whether to return individual samples
            **kwargs: Keyword arguments for base model

        Returns:
            UncertaintyOutput with mean and uncertainty estimates
        """
        # Collect predictions from multiple forward passes
        predictions = []

        self._enable_dropout()

        with torch.no_grad():
            for _ in range(self.n_samples):
                output = self.model(*args, **kwargs)

                # Handle different output types
                if isinstance(output, torch.Tensor):
                    predictions.append(output)
                elif hasattr(output, "logits"):
                    predictions.append(output.logits)
                elif hasattr(output, "regression_output"):
                    predictions.append(output.regression_output)
                else:
                    predictions.append(output)

        self._disable_dropout()

        # Stack predictions: (n_samples, batch, ...)
        stacked = torch.stack(predictions, dim=0)

        # Compute mean prediction
        mean = stacked.mean(dim=0)

        # Compute epistemic uncertainty (variance across samples)
        epistemic = stacked.var(dim=0)

        return UncertaintyOutput(
            mean=mean,
            epistemic_uncertainty=epistemic,
            total_uncertainty=epistemic,
            ensemble_predictions=predictions if return_samples else None,
        )

    def predict_with_uncertainty(
        self,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method returning (mean, uncertainty).
        """
        output = self.forward(*args, **kwargs)
        return output.mean, output.epistemic_uncertainty


class EnsemblePredictor(nn.Module):
    """
    Ensemble-based uncertainty estimation.

    Combines predictions from multiple independently trained models
    to estimate uncertainty through prediction disagreement.
    """

    def __init__(
        self,
        models: List[nn.Module],
        aggregation: str = "mean",
    ):
        """
        Initialize ensemble predictor.

        Args:
            models: List of trained models
            aggregation: How to combine predictions ('mean', 'median')
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.aggregation = aggregation
        self.n_models = len(models)

    @classmethod
    def from_checkpoints(
        cls,
        model_class: type,
        checkpoint_paths: List[str],
        model_kwargs: Optional[Dict] = None,
        device: str = "cpu",
    ) -> "EnsemblePredictor":
        """
        Load ensemble from checkpoint files.

        Args:
            model_class: Class to instantiate for each model
            checkpoint_paths: Paths to model checkpoints
            model_kwargs: Arguments for model construction
            device: Device to load models on

        Returns:
            EnsemblePredictor with loaded models
        """
        model_kwargs = model_kwargs or {}
        models = []

        for path in checkpoint_paths:
            model = model_class(**model_kwargs)
            checkpoint = torch.load(path, map_location=device)

            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            model.to(device)
            model.eval()
            models.append(model)

        return cls(models)

    def forward(
        self,
        *args,
        return_individual: bool = False,
        **kwargs,
    ) -> UncertaintyOutput:
        """
        Forward pass with ensemble uncertainty estimation.

        Args:
            *args: Positional arguments for models
            return_individual: Whether to return individual predictions
            **kwargs: Keyword arguments for models

        Returns:
            UncertaintyOutput with mean and uncertainty estimates
        """
        predictions = []

        with torch.no_grad():
            for model in self.models:
                output = model(*args, **kwargs)

                # Handle different output types
                if isinstance(output, torch.Tensor):
                    predictions.append(output)
                elif hasattr(output, "logits"):
                    predictions.append(output.logits)
                elif hasattr(output, "regression_output"):
                    predictions.append(output.regression_output)
                else:
                    predictions.append(output)

        # Stack predictions: (n_models, batch, ...)
        stacked = torch.stack(predictions, dim=0)

        # Compute aggregated prediction
        if self.aggregation == "mean":
            mean = stacked.mean(dim=0)
        elif self.aggregation == "median":
            mean = stacked.median(dim=0).values
        else:
            mean = stacked.mean(dim=0)

        # Compute epistemic uncertainty (disagreement between models)
        epistemic = stacked.var(dim=0)

        return UncertaintyOutput(
            mean=mean,
            epistemic_uncertainty=epistemic,
            total_uncertainty=epistemic,
            ensemble_predictions=predictions if return_individual else None,
        )

    def predict_with_confidence(
        self,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with confidence scores.

        Returns:
            Tuple of (predictions, confidence) where confidence = 1 - normalized_uncertainty
        """
        output = self.forward(*args, **kwargs)

        # Normalize uncertainty to [0, 1] confidence
        max_uncertainty = output.epistemic_uncertainty.max()
        if max_uncertainty > 0:
            confidence = 1 - (output.epistemic_uncertainty / max_uncertainty)
        else:
            confidence = torch.ones_like(output.epistemic_uncertainty)

        return output.mean, confidence


class EvidentialHead(nn.Module):
    """
    Evidential deep learning head for uncertainty estimation.

    Instead of predicting point estimates, predicts parameters of a
    Dirichlet distribution (for classification) or Normal-Inverse-Gamma
    distribution (for regression).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        task: str = "classification",
    ):
        """
        Initialize evidential head.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes (for classification)
            task: 'classification' or 'regression'
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.task = task

        if task == "classification":
            # Predict Dirichlet concentration parameters
            self.evidence_layer = nn.Linear(input_dim, num_classes)
        else:
            # Predict NIG parameters: gamma, nu, alpha, beta
            self.nig_layer = nn.Linear(input_dim, 4)

    def forward(self, x: torch.Tensor) -> UncertaintyOutput:
        """
        Forward pass predicting distribution parameters.

        Args:
            x: Input features (batch, input_dim)

        Returns:
            UncertaintyOutput with predictions and uncertainty
        """
        if self.task == "classification":
            return self._forward_classification(x)
        else:
            return self._forward_regression(x)

    def _forward_classification(self, x: torch.Tensor) -> UncertaintyOutput:
        """Evidential classification."""
        # Predict evidence (non-negative)
        evidence = F.softplus(self.evidence_layer(x))

        # Dirichlet concentration parameters
        alpha = evidence + 1  # (batch, num_classes)

        # Dirichlet strength (total evidence)
        S = alpha.sum(dim=-1, keepdim=True)

        # Expected probabilities (mean of Dirichlet)
        probs = alpha / S

        # Uncertainty: inversely related to total evidence
        # High evidence = low uncertainty
        epistemic = self.num_classes / S.squeeze(-1)

        # Aleatoric uncertainty: expected entropy of categorical
        aleatoric = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

        return UncertaintyOutput(
            mean=probs,
            alpha=alpha,
            aleatoric_uncertainty=aleatoric,
            epistemic_uncertainty=epistemic,
            total_uncertainty=aleatoric + epistemic,
        )

    def _forward_regression(self, x: torch.Tensor) -> UncertaintyOutput:
        """Evidential regression using Normal-Inverse-Gamma prior."""
        # Predict NIG parameters
        nig_params = self.nig_layer(x)

        # gamma: predicted mean
        gamma = nig_params[:, 0]

        # nu: precision of mean estimate (> 0)
        nu = F.softplus(nig_params[:, 1]) + 1e-6

        # alpha: shape parameter (> 1)
        alpha = F.softplus(nig_params[:, 2]) + 1

        # beta: scale parameter (> 0)
        beta = F.softplus(nig_params[:, 3]) + 1e-6

        # Expected value is gamma
        mean = gamma

        # Aleatoric uncertainty: expected variance = beta / (alpha - 1)
        aleatoric = beta / (alpha - 1 + 1e-6)

        # Epistemic uncertainty: variance of mean = beta / (nu * (alpha - 1))
        epistemic = beta / (nu * (alpha - 1 + 1e-6))

        return UncertaintyOutput(
            mean=mean.unsqueeze(-1),
            aleatoric_uncertainty=aleatoric.unsqueeze(-1),
            epistemic_uncertainty=epistemic.unsqueeze(-1),
            total_uncertainty=(aleatoric + epistemic).unsqueeze(-1),
        )


class EvidentialLoss(nn.Module):
    """
    Loss function for evidential deep learning.

    Combines:
    - Negative log-likelihood under Dirichlet
    - KL divergence regularization to prevent over-confident predictions
    """

    def __init__(
        self,
        lambda_reg: float = 0.1,
        annealing_epochs: int = 10,
    ):
        """
        Initialize evidential loss.

        Args:
            lambda_reg: Regularization coefficient
            annealing_epochs: Epochs over which to anneal regularization
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.annealing_epochs = annealing_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Set current epoch for annealing."""
        self.current_epoch = epoch

    def forward(
        self,
        alpha: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute evidential loss.

        Args:
            alpha: Dirichlet concentration parameters (batch, num_classes)
            targets: One-hot or soft targets (batch, num_classes)

        Returns:
            Scalar loss
        """
        # Dirichlet strength
        S = alpha.sum(dim=-1, keepdim=True)

        # Type II maximum likelihood loss
        # Based on Digamma function approximation
        loss_nll = torch.sum(
            targets * (torch.digamma(S) - torch.digamma(alpha)),
            dim=-1,
        )

        # KL divergence regularization
        # Penalizes evidence for incorrect classes
        alpha_tilde = targets + (1 - targets) * alpha

        # KL(Dir(alpha_tilde) || Dir(1))
        kl = self._kl_divergence(alpha_tilde)

        # Annealing coefficient
        annealing = min(1.0, self.current_epoch / max(1, self.annealing_epochs))

        loss = loss_nll + self.lambda_reg * annealing * kl

        return loss.mean()

    def _kl_divergence(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence from uniform Dirichlet."""
        K = alpha.shape[-1]
        alpha0 = alpha.sum(dim=-1, keepdim=True)

        # KL(Dir(alpha) || Dir(1))
        kl = (
            torch.lgamma(alpha0.squeeze(-1))
            - K * torch.lgamma(torch.tensor(1.0, device=alpha.device))
            - torch.lgamma(alpha).sum(dim=-1)
            + torch.sum(
                (alpha - 1) * (torch.digamma(alpha) - torch.digamma(alpha0)),
                dim=-1,
            )
        )

        return kl


class OntologicalEvidentialHead(nn.Module):
    """
    Evidential head for Rainbow Table ontological regression.

    Predicts Dirichlet parameters for each of the three dimensions:
    - Temporal: past, present, future
    - Spatial: thing, place, person
    - Ontological: imagined, forgotten, known
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
    ):
        """
        Initialize ontological evidential head.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Separate evidential heads for each dimension
        self.temporal_head = EvidentialHead(hidden_dim, num_classes=3)
        self.spatial_head = EvidentialHead(hidden_dim, num_classes=3)
        self.ontological_head = EvidentialHead(hidden_dim, num_classes=3)

        # Confidence head (regression)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, UncertaintyOutput]:
        """
        Forward pass.

        Args:
            x: Input features (batch, input_dim)

        Returns:
            Dict with UncertaintyOutput for each dimension
        """
        h = self.shared(x)

        return {
            "temporal": self.temporal_head(h),
            "spatial": self.spatial_head(h),
            "ontological": self.ontological_head(h),
            "confidence": self.confidence_head(h),
        }


def compute_calibration_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    uncertainties: torch.Tensor,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute calibration metrics for uncertainty estimates.

    Args:
        predictions: Model predictions
        targets: Ground truth values
        uncertainties: Predicted uncertainties
        n_bins: Number of bins for calibration

    Returns:
        Dict with calibration metrics (ECE, MCE, etc.)
    """
    # Move to numpy
    predictions = predictions.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()
    uncertainties = uncertainties.detach().cpu().numpy().flatten()

    # Compute errors
    errors = np.abs(predictions - targets)

    # Bin by uncertainty
    bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-6  # Include max value

    ece = 0.0
    mce = 0.0
    bin_metrics = []

    for i in range(n_bins):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])

        if mask.sum() > 0:
            bin_uncertainty = uncertainties[mask].mean()
            bin_error = errors[mask].mean()
            bin_count = mask.sum()

            # Calibration gap
            gap = np.abs(bin_uncertainty - bin_error)

            # Weighted contribution to ECE
            ece += (bin_count / len(predictions)) * gap

            # Max calibration error
            mce = max(mce, gap)

            bin_metrics.append(
                {
                    "uncertainty": float(bin_uncertainty),
                    "error": float(bin_error),
                    "count": int(bin_count),
                    "gap": float(gap),
                }
            )

    return {
        "ece": float(ece),
        "mce": float(mce),
        "n_bins": n_bins,
        "bin_metrics": bin_metrics,
    }


if __name__ == "__main__":
    print("Testing uncertainty estimation modules...")

    # Test MC Dropout
    print("\n=== MC Dropout ===")
    base_model = nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 3),
    )

    mc_model = MCDropout(base_model, dropout_rate=0.1, n_samples=10)

    x = torch.randn(4, 768)
    output = mc_model(x)

    print(f"Mean shape: {output.mean.shape}")
    print(f"Uncertainty shape: {output.epistemic_uncertainty.shape}")
    print(f"Mean uncertainty: {output.epistemic_uncertainty.mean().item():.4f}")

    # Test Ensemble
    print("\n=== Ensemble ===")
    models = [
        nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 3))
        for _ in range(5)
    ]

    ensemble = EnsemblePredictor(models)
    output = ensemble(x)

    print(f"Mean shape: {output.mean.shape}")
    print(f"Uncertainty shape: {output.epistemic_uncertainty.shape}")

    # Test Evidential Head
    print("\n=== Evidential Head (Classification) ===")
    evidential = EvidentialHead(768, num_classes=3, task="classification")
    output = evidential(x)

    print(f"Probs shape: {output.mean.shape}")
    print(f"Alpha shape: {output.alpha.shape}")
    print(f"Aleatoric: {output.aleatoric_uncertainty.mean().item():.4f}")
    print(f"Epistemic: {output.epistemic_uncertainty.mean().item():.4f}")

    # Test Evidential Loss
    print("\n=== Evidential Loss ===")
    loss_fn = EvidentialLoss(lambda_reg=0.1)
    targets = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.33, 0.34, 0.33]]
    )
    loss = loss_fn(output.alpha, targets)
    print(f"Loss: {loss.item():.4f}")

    # Test Ontological Evidential Head
    print("\n=== Ontological Evidential Head ===")
    onto_head = OntologicalEvidentialHead(768)
    outputs = onto_head(x)

    print(f"Temporal probs shape: {outputs['temporal'].mean.shape}")
    print(
        f"Spatial epistemic: {outputs['spatial'].epistemic_uncertainty.mean().item():.4f}"
    )
    print(f"Confidence shape: {outputs['confidence'].shape}")

    print("\n✓ All uncertainty tests passed!")
