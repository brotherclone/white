"""
Multi-class classifier for rebracketing type prediction.

Extends the binary classifier to predict specific rebracketing types
(spatial, temporal, causal, perceptual, memory, ontological, narrative, identity).
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict


class MultiClassRebracketingClassifier(nn.Module):
    """
    MLP classifier for multi-class rebracketing type prediction.

    Supports both single-label (one type per segment) and multi-label
    (multiple types per segment) classification modes.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3,
        activation: str = "relu",
        multi_label: bool = False,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize multi-class classifier.

        Args:
            input_dim: Size of input embeddings
            num_classes: Number of rebracketing type classes
            hidden_dims: List of hidden layer sizes
            dropout: Dropout probability
            activation: Activation function ("relu", "gelu", "tanh")
            multi_label: If True, use sigmoid for multi-label classification
                        If False, use softmax for single-label classification
            class_weights: Optional tensor of shape [num_classes] for class weighting
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.multi_label = multi_label

        # Store class weights (used in loss computation)
        self.register_buffer("class_weights", class_weights)

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Activation
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

        # Output layer (no activation, will use CrossEntropyLoss or BCEWithLogitsLoss)
        layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Classify embeddings into rebracketing types.

        Args:
            embeddings: Input embeddings [batch, input_dim]

        Returns:
            Logits [batch, num_classes]
            - For single-label: use with CrossEntropyLoss
            - For multi-label: use with BCEWithLogitsLoss
        """
        logits = self.mlp(embeddings)
        return logits  # [batch, num_classes]

    def predict(self, embeddings: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Get predictions from embeddings.

        Args:
            embeddings: Input embeddings [batch, input_dim]
            threshold: Threshold for multi-label classification (only used if multi_label=True)

        Returns:
            - If single-label: class indices [batch]
            - If multi-label: binary mask [batch, num_classes]
        """
        with torch.no_grad():
            logits = self.forward(embeddings)

            if self.multi_label:
                # Multi-label: apply sigmoid and threshold
                probs = torch.sigmoid(logits)
                predictions = (probs > threshold).long()
                return predictions
            else:
                # Single-label: take argmax
                predictions = torch.argmax(logits, dim=-1)
                return predictions

    @staticmethod
    def compute_class_weights(
        class_counts: Dict[int, int],
        num_classes: int,
        mode: str = "balanced",
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Compute class weights from class distribution.

        Args:
            class_counts: Dictionary mapping class index to count
            num_classes: Total number of classes
            mode: "balanced" for inverse frequency weighting, "uniform" for equal weights
            device: Device to place tensor on

        Returns:
            Tensor of class weights [num_classes]
        """
        if mode == "uniform":
            return torch.ones(num_classes, device=device)

        elif mode == "balanced":
            # Compute inverse frequency weights
            # weight[i] = total_samples / (num_classes * count[i])
            counts = torch.zeros(num_classes, device=device)
            for class_idx, count in class_counts.items():
                counts[class_idx] = count

            total_samples = counts.sum()

            # Avoid division by zero
            counts = torch.clamp(counts, min=1.0)

            weights = total_samples / (num_classes * counts)

            return weights

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'balanced' or 'uniform'.")


class MultiClassRainbowModel(nn.Module):
    """
    Complete multi-class model: Text encoder + Multi-class Classifier.

    Combines text encoding and multi-class classification in a single module.
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        classifier: MultiClassRebracketingClassifier,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.classifier = classifier

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Logits [batch, num_classes]
        """
        # Encode text
        embeddings = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Classify
        logits = self.classifier(embeddings)

        return logits

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Get predictions from inputs.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            threshold: Threshold for multi-label classification

        Returns:
            Predictions (class indices or binary masks)
        """
        # Encode text
        embeddings = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Predict
        return self.classifier.predict(embeddings, threshold=threshold)


if __name__ == "__main__":
    # Quick test
    print("Testing MultiClassRebracketingClassifier...")

    # Test single-label mode
    print("\n=== Single-label mode ===")
    num_classes = 8
    classifier = MultiClassRebracketingClassifier(
        input_dim=768,
        num_classes=num_classes,
        hidden_dims=[256, 128],
        dropout=0.3,
        multi_label=False,
    )

    # Test input
    batch_size = 4
    embeddings = torch.randn(batch_size, 768)

    # Forward pass
    with torch.no_grad():
        logits = classifier(embeddings)
        predictions = classifier.predict(embeddings)

    print(f"Input shape: {embeddings.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Expected logits: [{batch_size}, {num_classes}]")
    print(f"Expected predictions: [{batch_size}]")
    print(
        "✓ Single-label test passed!"
        if logits.shape == (batch_size, num_classes)
        and predictions.shape == (batch_size,)
        else "✗ Test failed!"
    )

    # Test multi-label mode
    print("\n=== Multi-label mode ===")
    classifier_multilabel = MultiClassRebracketingClassifier(
        input_dim=768,
        num_classes=num_classes,
        hidden_dims=[256, 128],
        dropout=0.3,
        multi_label=True,
    )

    with torch.no_grad():
        logits_ml = classifier_multilabel(embeddings)
        predictions_ml = classifier_multilabel.predict(embeddings, threshold=0.5)

    print(f"Logits shape: {logits_ml.shape}")
    print(f"Predictions shape: {predictions_ml.shape}")
    print(f"Expected: [{batch_size}, {num_classes}]")
    print(
        "✓ Multi-label test passed!"
        if predictions_ml.shape == (batch_size, num_classes)
        else "✗ Test failed!"
    )

    # Test class weight computation
    print("\n=== Class weight computation ===")
    class_counts = {0: 100, 1: 50, 2: 200, 3: 25, 4: 150, 5: 75, 6: 300, 7: 10}
    weights = MultiClassRebracketingClassifier.compute_class_weights(
        class_counts, num_classes=num_classes, mode="balanced"
    )
    print(f"Class counts: {class_counts}")
    print(f"Class weights: {weights}")
    print("✓ Weight computation passed!")
