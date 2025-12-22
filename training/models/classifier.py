"""
Binary classifier for rebracketing detection.

Simple MLP head on top of text embeddings.
"""

import torch
import torch.nn as nn
from typing import List


class BinaryClassifier(nn.Module):
    """MLP classifier for binary classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        """
        Args:
            input_dim: Size of input embeddings
            hidden_dims: List of hidden layer sizes
            dropout: Dropout probability
            activation: Activation function ("relu", "gelu", "tanh")
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

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

        # Output layer (no activation, will use BCEWithLogitsLoss)
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Classify embeddings.

        Args:
            embeddings: Input embeddings [batch, input_dim]

        Returns:
            Logits [batch, 1]
        """
        logits = self.mlp(embeddings)
        return logits.squeeze(-1)  # [batch]


class RainbowModel(nn.Module):
    """
    Complete model: Text encoder + Classifier.

    Combines text encoding and classification in a single module.
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        classifier: nn.Module,
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
            Logits [batch]
        """
        # Encode text
        embeddings = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Classify
        logits = self.classifier(embeddings)

        return logits


if __name__ == "__main__":
    # Quick test
    print("Testing BinaryClassifier...")

    classifier = BinaryClassifier(
        input_dim=768,
        hidden_dims=[256, 128],
        dropout=0.3,
    )

    # Test input
    batch_size = 4
    embeddings = torch.randn(batch_size, 768)

    # Forward pass
    with torch.no_grad():
        logits = classifier(embeddings)

    print(f"\nInput shape: {embeddings.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected: [{batch_size}]")
    print("✓ Test passed!" if logits.shape == (batch_size,) else "✗ Test failed!")
