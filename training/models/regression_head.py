"""
Regression head for continuous metric prediction.

Predicts continuous values like intensity, fluidity, temporal complexity,
and supports bounded/unbounded outputs with uncertainty estimation.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Literal, Tuple, Dict


class RegressionHead(nn.Module):
    """
    MLP regression head for predicting continuous values.

    Supports:
    - Single or multiple continuous targets
    - Bounded outputs (sigmoid) or unbounded outputs (none)
    - Optional uncertainty estimation (variance prediction)
    """

    def __init__(
        self,
        input_dim: int,
        num_targets: int = 1,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3,
        activation: str = "relu",
        output_activation: Optional[
            Literal["sigmoid", "tanh", "softplus", "none"]
        ] = None,
        predict_uncertainty: bool = False,
    ):
        """
        Initialize regression head.

        Args:
            input_dim: Size of input embeddings
            num_targets: Number of continuous targets to predict
            hidden_dims: List of hidden layer sizes
            dropout: Dropout probability
            activation: Hidden layer activation function ("relu", "gelu", "tanh")
            output_activation: Output activation for bounded targets:
                - "sigmoid": bounds to [0, 1]
                - "tanh": bounds to [-1, 1]
                - "softplus": ensures non-negative outputs
                - None: unbounded output
            predict_uncertainty: If True, also predicts variance for each target
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_targets = num_targets
        self.hidden_dims = hidden_dims
        self.output_activation = output_activation
        self.predict_uncertainty = predict_uncertainty

        # Build shared MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Hidden activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        # Output layer for mean predictions
        self.mean_layer = nn.Linear(prev_dim, num_targets)

        # Optional variance layer for uncertainty estimation
        if predict_uncertainty:
            self.variance_layer = nn.Linear(prev_dim, num_targets)
            # Softplus to ensure positive variance
            self.variance_activation = nn.Softplus()

        # Output activation
        if output_activation == "sigmoid":
            self.output_act = nn.Sigmoid()
        elif output_activation == "tanh":
            self.output_act = nn.Tanh()
        elif output_activation == "softplus":
            self.output_act = nn.Softplus()
        else:
            self.output_act = None

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict continuous values from embeddings.

        Args:
            embeddings: Input embeddings [batch, input_dim]

        Returns:
            Tuple of:
                - predictions: Continuous predictions [batch, num_targets]
                - variance: Uncertainty estimates [batch, num_targets] or None
        """
        # Shared representation
        hidden = self.shared_layers(embeddings)

        # Mean predictions
        predictions = self.mean_layer(hidden)

        # Apply output activation if specified
        if self.output_act is not None:
            predictions = self.output_act(predictions)

        # Uncertainty estimation
        variance = None
        if self.predict_uncertainty:
            variance = self.variance_activation(self.variance_layer(hidden))

        return predictions, variance

    def predict(
        self,
        embeddings: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions from embeddings.

        Args:
            embeddings: Input embeddings [batch, input_dim]
            return_uncertainty: If True, also return variance estimates

        Returns:
            - If return_uncertainty=False: predictions [batch, num_targets]
            - If return_uncertainty=True: (predictions, variance) tuple
        """
        with torch.no_grad():
            predictions, variance = self.forward(embeddings)

            if return_uncertainty and variance is not None:
                return predictions, variance
            return predictions

    def get_prediction_intervals(
        self,
        embeddings: torch.Tensor,
        confidence: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute prediction intervals with confidence bounds.

        Args:
            embeddings: Input embeddings [batch, input_dim]
            confidence: Confidence level (default 95%)

        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if not self.predict_uncertainty:
            raise ValueError("Model not configured for uncertainty estimation")

        with torch.no_grad():
            predictions, variance = self.forward(embeddings)
            std = torch.sqrt(variance)

            # Z-score for confidence interval (approximately 1.96 for 95%)
            from scipy import stats

            z = stats.norm.ppf((1 + confidence) / 2)

            lower = predictions - z * std
            upper = predictions + z * std

            return predictions, lower, upper


class MultiTargetRegressionHead(nn.Module):
    """
    Regression head with separate sub-heads for different target groups.

    Useful when targets have different characteristics (e.g., different
    output ranges or different optimal loss functions).
    """

    def __init__(
        self,
        input_dim: int,
        target_configs: Dict[str, Dict],
        shared_hidden_dims: List[int] = [256],
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        """
        Initialize multi-target regression head.

        Args:
            input_dim: Size of input embeddings
            target_configs: Dictionary mapping target names to configs:
                {
                    "intensity": {"num_targets": 1, "activation": "sigmoid"},
                    "complexity": {"num_targets": 1, "activation": None},
                }
            shared_hidden_dims: Hidden dims for shared representation
            dropout: Dropout probability
            activation: Hidden layer activation
        """
        super().__init__()

        self.input_dim = input_dim
        self.target_configs = target_configs
        self.target_names = list(target_configs.keys())

        # Shared representation
        layers = []
        prev_dim = input_dim

        for hidden_dim in shared_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)
        self.shared_dim = prev_dim

        # Per-target output heads
        self.target_heads = nn.ModuleDict()
        self.target_activations = {}

        for name, config in target_configs.items():
            num_targets = config.get("num_targets", 1)
            self.target_heads[name] = nn.Linear(prev_dim, num_targets)

            # Store activation type
            act = config.get("activation", None)
            if act == "sigmoid":
                self.target_activations[name] = nn.Sigmoid()
            elif act == "softmax":
                self.target_activations[name] = nn.Softmax(dim=-1)
            elif act == "tanh":
                self.target_activations[name] = nn.Tanh()
            elif act == "softplus":
                self.target_activations[name] = nn.Softplus()
            else:
                self.target_activations[name] = None

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict all targets from embeddings.

        Args:
            embeddings: Input embeddings [batch, input_dim]

        Returns:
            Dictionary mapping target names to predictions
        """
        # Shared representation
        hidden = self.shared_layers(embeddings)

        # Per-target predictions
        outputs = {}
        for name in self.target_names:
            pred = self.target_heads[name](hidden)

            # Apply activation if specified
            if self.target_activations[name] is not None:
                pred = self.target_activations[name](pred)

            outputs[name] = pred

        return outputs


if __name__ == "__main__":
    # Quick tests
    print("Testing RegressionHead...")

    batch_size = 4
    input_dim = 768
    embeddings = torch.randn(batch_size, input_dim)

    # Test 1: Single target, bounded output
    print("\n=== Single target, bounded [0,1] ===")
    head = RegressionHead(
        input_dim=input_dim,
        num_targets=1,
        hidden_dims=[256, 128],
        output_activation="sigmoid",
    )

    predictions, variance = head(embeddings)
    print(f"Input shape: {embeddings.shape}")
    print(f"Output shape: {predictions.shape}")
    print(
        f"Output range: [{predictions.min().item():.3f}, {predictions.max().item():.3f}]"
    )
    print(
        "✓ Test passed!" if predictions.shape == (batch_size, 1) else "✗ Test failed!"
    )

    # Test 2: Multiple targets, unbounded
    print("\n=== Multiple targets, unbounded ===")
    head_multi = RegressionHead(
        input_dim=input_dim,
        num_targets=3,
        hidden_dims=[256],
        output_activation=None,
    )

    predictions_multi, _ = head_multi(embeddings)
    print(f"Output shape: {predictions_multi.shape}")
    print(
        "✓ Test passed!"
        if predictions_multi.shape == (batch_size, 3)
        else "✗ Test failed!"
    )

    # Test 3: With uncertainty estimation
    print("\n=== With uncertainty estimation ===")
    head_unc = RegressionHead(
        input_dim=input_dim,
        num_targets=2,
        hidden_dims=[256, 128],
        output_activation="sigmoid",
        predict_uncertainty=True,
    )

    predictions_unc, variance_unc = head_unc(embeddings)
    print(f"Predictions shape: {predictions_unc.shape}")
    print(f"Variance shape: {variance_unc.shape}")
    print(
        f"Variance range: [{variance_unc.min().item():.3f}, {variance_unc.max().item():.3f}]"
    )
    print(
        "✓ Test passed!"
        if predictions_unc.shape == (batch_size, 2)
        and variance_unc.shape == (batch_size, 2)
        else "✗ Test failed!"
    )

    # Test 4: MultiTargetRegressionHead
    print("\n=== MultiTargetRegressionHead ===")
    head_mt = MultiTargetRegressionHead(
        input_dim=input_dim,
        target_configs={
            "intensity": {"num_targets": 1, "activation": "sigmoid"},
            "fluidity": {"num_targets": 1, "activation": "sigmoid"},
            "complexity": {"num_targets": 1, "activation": None},
        },
        shared_hidden_dims=[256],
    )

    outputs = head_mt(embeddings)
    print(f"Target names: {list(outputs.keys())}")
    for name, pred in outputs.items():
        print(
            f"  {name}: shape={pred.shape}, range=[{pred.min().item():.3f}, {pred.max().item():.3f}]"
        )
    print("✓ Test passed!" if len(outputs) == 3 else "✗ Test failed!")
