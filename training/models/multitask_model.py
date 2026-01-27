"""
Multi-task model combining classification and regression heads.

Supports simultaneous training of:
- Classification (discrete album/rebracketing type prediction)
- Rainbow Table ontological regression (continuous mode distributions)

Loss weighting strategies:
- Fixed weights
- Uncertainty-based weighting (Kendall et al.)
- Gradient normalization (GradNorm)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Literal, List
from dataclasses import dataclass

from .rainbow_table_regression_head import (
    RainbowTableRegressionHead,
    OntologicalScores,
)
from .multiclass_classifier import MultiClassRebracketingClassifier


@dataclass
class MultiTaskOutput:
    """Container for multi-task model outputs."""

    # Classification outputs
    classification_logits: torch.Tensor  # [batch, num_classes]

    # Regression outputs
    ontological_scores: OntologicalScores

    # Embeddings (for inspection/debugging)
    embeddings: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to flat dictionary."""
        result = {
            "classification_logits": self.classification_logits,
            **self.ontological_scores.to_dict(),
        }
        if self.embeddings is not None:
            result["embeddings"] = self.embeddings
        return result


@dataclass
class MultiTaskLoss:
    """Container for multi-task loss components."""

    total_loss: torch.Tensor
    classification_loss: torch.Tensor
    temporal_loss: torch.Tensor
    spatial_loss: torch.Tensor
    ontological_loss: torch.Tensor
    confidence_loss: torch.Tensor

    # Optional per-task weights (for logging)
    weights: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary of scalar values."""
        result = {
            "total_loss": self.total_loss.item(),
            "classification_loss": self.classification_loss.item(),
            "temporal_loss": self.temporal_loss.item(),
            "spatial_loss": self.spatial_loss.item(),
            "ontological_loss": self.ontological_loss.item(),
            "confidence_loss": self.confidence_loss.item(),
        }
        if self.weights:
            result["weights"] = self.weights
        return result


class MultiTaskRainbowModel(nn.Module):
    """
    Multi-task model combining classification and ontological regression.

    Architecture:
        TextEncoder → [batch, hidden_size]
                    ↓
           ┌────────┴────────┐
           ↓                 ↓
    ClassificationHead  RainbowTableRegressionHead
           ↓                 ↓
    [batch, num_classes]   OntologicalScores (10 values)
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        num_classes: int = 8,
        classifier_hidden_dims: List[int] = [256, 128],
        regression_hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3,
        activation: str = "relu",
        multi_label: bool = False,
        predict_uncertainty: bool = False,
    ):
        """
        Initialize multi-task model.

        Args:
            text_encoder: Pre-trained text encoder module
            num_classes: Number of classification classes (rebracketing types)
            classifier_hidden_dims: Hidden dims for classification head
            regression_hidden_dims: Hidden dims for regression head
            dropout: Dropout probability
            activation: Activation function
            multi_label: If True, use multi-label classification
            predict_uncertainty: If True, regression head predicts variance
        """
        super().__init__()

        self.text_encoder = text_encoder
        input_dim = text_encoder.hidden_size

        # Classification head
        self.classifier = MultiClassRebracketingClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=classifier_hidden_dims,
            dropout=dropout,
            activation=activation,
            multi_label=multi_label,
        )

        # Regression head
        self.regression_head = RainbowTableRegressionHead(
            input_dim=input_dim,
            hidden_dims=regression_hidden_dims,
            dropout=dropout,
            activation=activation,
            predict_uncertainty=predict_uncertainty,
        )

        self.num_classes = num_classes
        self.multi_label = multi_label
        self.predict_uncertainty = predict_uncertainty

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embeddings: bool = False,
    ) -> MultiTaskOutput:
        """
        Forward pass through both heads.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            return_embeddings: If True, include embeddings in output

        Returns:
            MultiTaskOutput with classification logits and ontological scores
        """
        # Shared encoding
        embeddings = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Classification
        classification_logits = self.classifier(embeddings)

        # Regression
        ontological_scores = self.regression_head(embeddings)

        return MultiTaskOutput(
            classification_logits=classification_logits,
            ontological_scores=ontological_scores,
            embeddings=embeddings if return_embeddings else None,
        )

    def forward_classification_only(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through classification head only."""
        embeddings = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return self.classifier(embeddings)

    def forward_regression_only(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> OntologicalScores:
        """Forward pass through regression head only."""
        embeddings = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return self.regression_head(embeddings)

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        classification_threshold: float = 0.5,
    ) -> Dict:
        """
        Get predictions from both heads.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            classification_threshold: Threshold for multi-label classification

        Returns:
            Dictionary with predictions from both tasks
        """
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)

            # Classification predictions
            if self.multi_label:
                probs = torch.sigmoid(output.classification_logits)
                class_predictions = (probs > classification_threshold).long()
            else:
                class_predictions = torch.argmax(output.classification_logits, dim=-1)

            # Regression predictions
            albums = self.regression_head.predict_album(output.ontological_scores)
            combined_modes = self.regression_head.predict_combined_mode(
                output.ontological_scores
            )

            return {
                "classification": class_predictions,
                "classification_logits": output.classification_logits,
                "albums": albums,
                "combined_modes": combined_modes,
                "ontological_scores": output.ontological_scores,
            }


class MultiTaskLossComputer(nn.Module):
    """
    Computes combined loss for multi-task learning.

    Supports multiple loss weighting strategies:
    - fixed: Manual weights for each task
    - uncertainty: Learned task-specific uncertainties (Kendall et al.)
    - gradnorm: Dynamic gradient normalization (GradNorm paper)
    """

    def __init__(
        self,
        weighting_strategy: Literal["fixed", "uncertainty", "gradnorm"] = "fixed",
        loss_weights: Optional[Dict[str, float]] = None,
        classification_loss: Literal["cross_entropy", "bce"] = "cross_entropy",
        regression_loss: Literal["kl_div", "mse", "smooth_l1"] = "kl_div",
        label_smoothing: float = 0.0,
    ):
        """
        Initialize loss computer.

        Args:
            weighting_strategy: How to weight task losses
            loss_weights: Manual weights (only for "fixed" strategy)
            classification_loss: Loss function for classification
            regression_loss: Loss function for regression targets
            label_smoothing: Label smoothing for cross-entropy
        """
        super().__init__()

        self.weighting_strategy = weighting_strategy
        self.classification_loss_type = classification_loss
        self.regression_loss_type = regression_loss
        self.label_smoothing = label_smoothing

        # Default weights
        default_weights = {
            "classification": 1.0,
            "temporal": 0.8,
            "spatial": 0.8,
            "ontological": 0.8,
            "confidence": 0.5,
        }
        self.loss_weights = loss_weights or default_weights

        # For uncertainty weighting: learn log(σ²) for each task
        if weighting_strategy == "uncertainty":
            self.log_vars = nn.ParameterDict(
                {
                    name: nn.Parameter(torch.zeros(1))
                    for name in self.loss_weights.keys()
                }
            )

        # Classification loss function
        if classification_loss == "cross_entropy":
            self.cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:  # bce
            self.cls_loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        output: MultiTaskOutput,
        classification_targets: torch.Tensor,
        temporal_targets: torch.Tensor,
        spatial_targets: torch.Tensor,
        ontological_targets: torch.Tensor,
        confidence_targets: torch.Tensor,
    ) -> MultiTaskLoss:
        """
        Compute combined multi-task loss.

        Args:
            output: MultiTaskOutput from model
            classification_targets: Class labels [batch] or [batch, num_classes]
            temporal_targets: Soft targets [batch, 3]
            spatial_targets: Soft targets [batch, 3]
            ontological_targets: Soft targets [batch, 3]
            confidence_targets: Confidence targets [batch, 1]

        Returns:
            MultiTaskLoss with all components
        """
        # Classification loss
        cls_loss = self.cls_loss_fn(
            output.classification_logits, classification_targets
        )

        # Get regression logits for loss computation
        t_logits, s_logits, o_logits, conf_logit = self.regression_head_logits(output)

        # Regression losses (KL divergence for distributions)
        if self.regression_loss_type == "kl_div":
            # KL(target || prediction) - use log_softmax for numerical stability
            temporal_loss = F.kl_div(
                F.log_softmax(t_logits, dim=-1),
                temporal_targets,
                reduction="batchmean",
            )
            spatial_loss = F.kl_div(
                F.log_softmax(s_logits, dim=-1),
                spatial_targets,
                reduction="batchmean",
            )
            ontological_loss = F.kl_div(
                F.log_softmax(o_logits, dim=-1),
                ontological_targets,
                reduction="batchmean",
            )
        elif self.regression_loss_type == "mse":
            temporal_loss = F.mse_loss(F.softmax(t_logits, dim=-1), temporal_targets)
            spatial_loss = F.mse_loss(F.softmax(s_logits, dim=-1), spatial_targets)
            ontological_loss = F.mse_loss(
                F.softmax(o_logits, dim=-1), ontological_targets
            )
        else:  # smooth_l1
            temporal_loss = F.smooth_l1_loss(
                F.softmax(t_logits, dim=-1), temporal_targets
            )
            spatial_loss = F.smooth_l1_loss(
                F.softmax(s_logits, dim=-1), spatial_targets
            )
            ontological_loss = F.smooth_l1_loss(
                F.softmax(o_logits, dim=-1), ontological_targets
            )

        # Confidence loss (BCE for sigmoid output)
        confidence_loss = F.binary_cross_entropy(
            output.ontological_scores.chromatic_confidence,
            confidence_targets,
        )

        # Combine losses with weighting strategy
        if self.weighting_strategy == "fixed":
            weights = self.loss_weights
            total_loss = (
                weights["classification"] * cls_loss
                + weights["temporal"] * temporal_loss
                + weights["spatial"] * spatial_loss
                + weights["ontological"] * ontological_loss
                + weights["confidence"] * confidence_loss
            )

        elif self.weighting_strategy == "uncertainty":
            # Uncertainty weighting: loss_i / (2 * σ_i²) + log(σ_i)
            # Using log_var = log(σ²), so σ² = exp(log_var)
            precision_cls = torch.exp(-self.log_vars["classification"])
            precision_t = torch.exp(-self.log_vars["temporal"])
            precision_s = torch.exp(-self.log_vars["spatial"])
            precision_o = torch.exp(-self.log_vars["ontological"])
            precision_c = torch.exp(-self.log_vars["confidence"])

            total_loss = (
                precision_cls * cls_loss
                + self.log_vars["classification"]
                + precision_t * temporal_loss
                + self.log_vars["temporal"]
                + precision_s * spatial_loss
                + self.log_vars["spatial"]
                + precision_o * ontological_loss
                + self.log_vars["ontological"]
                + precision_c * confidence_loss
                + self.log_vars["confidence"]
            )

            weights = {k: torch.exp(-v).item() for k, v in self.log_vars.items()}

        else:  # gradnorm (placeholder - requires custom training loop)
            weights = self.loss_weights
            total_loss = (
                weights["classification"] * cls_loss
                + weights["temporal"] * temporal_loss
                + weights["spatial"] * spatial_loss
                + weights["ontological"] * ontological_loss
                + weights["confidence"] * confidence_loss
            )

        return MultiTaskLoss(
            total_loss=total_loss,
            classification_loss=cls_loss,
            temporal_loss=temporal_loss,
            spatial_loss=spatial_loss,
            ontological_loss=ontological_loss,
            confidence_loss=confidence_loss,
            weights=weights if self.weighting_strategy != "fixed" else None,
        )

    def regression_head_logits(
        self,
        output: MultiTaskOutput,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract regression logits from output for loss computation.

        Note: This requires the model to expose logits, or we approximate
        by inverting softmax (which loses numerical precision).

        For best results, use model.regression_head.forward_logits() directly.
        """
        # Approximate logits from probabilities (not ideal but works)
        # log(softmax(x)) ≈ x - log(sum(exp(x)))
        # For small temperature, we can approximate
        scores = output.ontological_scores

        # Use log probabilities as proxy for logits
        t_logits = torch.log(scores.temporal_scores + 1e-8)
        s_logits = torch.log(scores.spatial_scores + 1e-8)
        o_logits = torch.log(scores.ontological_scores + 1e-8)
        conf_logit = torch.logit(scores.chromatic_confidence + 1e-8)

        return t_logits, s_logits, o_logits, conf_logit


class SequentialTrainer:
    """
    Sequential training strategy for multi-task learning.

    Trains classification first, then regression, then optional joint fine-tuning.
    Useful when simultaneous multi-task learning is unstable.
    """

    def __init__(
        self,
        model: MultiTaskRainbowModel,
        phase1_epochs: int = 10,
        phase2_epochs: int = 10,
        phase3_epochs: int = 5,
        freeze_encoder_phase2: bool = False,
    ):
        """
        Initialize sequential trainer.

        Args:
            model: Multi-task model to train
            phase1_epochs: Epochs for classification-only training
            phase2_epochs: Epochs for regression-only training
            phase3_epochs: Epochs for joint fine-tuning
            freeze_encoder_phase2: If True, freeze encoder during phase 2
        """
        self.model = model
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs
        self.phase3_epochs = phase3_epochs
        self.freeze_encoder_phase2 = freeze_encoder_phase2

    def get_phase_parameters(self, phase: int) -> List[nn.Parameter]:
        """Get trainable parameters for a specific phase."""
        if phase == 1:
            # Phase 1: encoder + classifier
            return list(self.model.text_encoder.parameters()) + list(
                self.model.classifier.parameters()
            )

        elif phase == 2:
            # Phase 2: regression head (optionally freeze encoder)
            if self.freeze_encoder_phase2:
                return list(self.model.regression_head.parameters())
            else:
                return list(self.model.text_encoder.parameters()) + list(
                    self.model.regression_head.parameters()
                )

        else:  # phase == 3
            # Phase 3: all parameters
            return list(self.model.parameters())

    def freeze_for_phase(self, phase: int):
        """Freeze/unfreeze parameters for a specific phase."""
        # Unfreeze all first
        for param in self.model.parameters():
            param.requires_grad = True

        if phase == 1:
            # Freeze regression head
            for param in self.model.regression_head.parameters():
                param.requires_grad = False

        elif phase == 2:
            # Freeze classifier
            for param in self.model.classifier.parameters():
                param.requires_grad = False

            if self.freeze_encoder_phase2:
                for param in self.model.text_encoder.parameters():
                    param.requires_grad = False

        # Phase 3: everything unfrozen


if __name__ == "__main__":
    # Quick tests (without actual encoder)
    print("Testing MultiTaskRainbowModel components...")

    # Mock text encoder
    class MockEncoder(nn.Module):
        def __init__(self, hidden_size=768):
            super().__init__()
            self.hidden_size = hidden_size
            self.linear = nn.Linear(512, hidden_size)

        def forward(self, input_ids, attention_mask):
            batch_size = input_ids.shape[0]
            return torch.randn(batch_size, self.hidden_size)

    batch_size = 4
    seq_len = 128
    num_classes = 8

    encoder = MockEncoder()
    model = MultiTaskRainbowModel(
        text_encoder=encoder,
        num_classes=num_classes,
        classifier_hidden_dims=[256, 128],
        regression_hidden_dims=[256, 128],
    )

    # Test forward pass
    print("\n=== Forward pass ===")
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    output = model(input_ids, attention_mask)
    print(f"Classification logits shape: {output.classification_logits.shape}")
    print(f"Temporal scores shape: {output.ontological_scores.temporal_scores.shape}")
    print(f"Spatial scores shape: {output.ontological_scores.spatial_scores.shape}")
    print(
        f"Ontological scores shape: {output.ontological_scores.ontological_scores.shape}"
    )
    print(f"Confidence shape: {output.ontological_scores.chromatic_confidence.shape}")
    print("✓ Forward pass works!")

    # Test predictions
    print("\n=== Predictions ===")
    predictions = model.predict(input_ids, attention_mask)
    print(f"Classification predictions: {predictions['classification']}")
    print(f"Albums: {predictions['albums']}")
    print(f"Combined modes: {predictions['combined_modes']}")
    print("✓ Predictions work!")

    # Test loss computation
    print("\n=== Loss computation ===")
    loss_computer = MultiTaskLossComputer(
        weighting_strategy="fixed",
    )

    # Create mock targets
    cls_targets = torch.randint(0, num_classes, (batch_size,))
    temporal_targets = F.softmax(torch.randn(batch_size, 3), dim=-1)
    spatial_targets = F.softmax(torch.randn(batch_size, 3), dim=-1)
    ontological_targets = F.softmax(torch.randn(batch_size, 3), dim=-1)
    confidence_targets = torch.rand(batch_size, 1)

    loss = loss_computer(
        output,
        cls_targets,
        temporal_targets,
        spatial_targets,
        ontological_targets,
        confidence_targets,
    )

    print(f"Total loss: {loss.total_loss.item():.4f}")
    print(f"Classification loss: {loss.classification_loss.item():.4f}")
    print(f"Temporal loss: {loss.temporal_loss.item():.4f}")
    print(f"Spatial loss: {loss.spatial_loss.item():.4f}")
    print(f"Ontological loss: {loss.ontological_loss.item():.4f}")
    print(f"Confidence loss: {loss.confidence_loss.item():.4f}")
    print("✓ Loss computation works!")

    # Test uncertainty weighting
    print("\n=== Uncertainty weighting ===")
    loss_computer_unc = MultiTaskLossComputer(
        weighting_strategy="uncertainty",
    )
    loss_unc = loss_computer_unc(
        output,
        cls_targets,
        temporal_targets,
        spatial_targets,
        ontological_targets,
        confidence_targets,
    )
    print(f"Total loss (uncertainty weighted): {loss_unc.total_loss.item():.4f}")
    print(f"Learned weights: {loss_unc.weights}")
    print("✓ Uncertainty weighting works!")

    # Test sequential trainer phases
    print("\n=== Sequential trainer ===")
    seq_trainer = SequentialTrainer(model)

    for phase in [1, 2, 3]:
        seq_trainer.freeze_for_phase(phase)
        trainable = sum(p.requires_grad for p in model.parameters())
        total = sum(1 for _ in model.parameters())
        print(f"Phase {phase}: {trainable}/{total} parameters trainable")
    print("✓ Sequential trainer works!")
