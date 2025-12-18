"""
Text encoder using DeBERTa.

Wraps HuggingFace transformers with configurable pooling and layer freezing.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """Text encoder using pre-trained transformer."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        hidden_size: int = 768,
        freeze_layers: int = 0,
        pooling: str = "mean",
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            hidden_size: Expected hidden size (for validation)
            freeze_layers: Number of encoder layers to freeze (0 = fine-tune all)
            pooling: How to pool token embeddings ("cls", "mean", "max")
        """
        super().__init__()

        self.model_name = model_name
        self.hidden_size = hidden_size
        self.pooling = pooling

        # Load pre-trained model
        print(f"Loading text encoder: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)

        # Validate hidden size
        actual_hidden_size = self.encoder.config.hidden_size
        if actual_hidden_size != hidden_size:
            print(
                f"Warning: Actual hidden size {actual_hidden_size} != expected {hidden_size}"
            )
            self.hidden_size = actual_hidden_size

        # Freeze layers if requested
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

    def _freeze_layers(self, num_layers: int):
        """Freeze the first N encoder layers."""
        print(f"Freezing first {num_layers} encoder layers")

        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

        # Freeze encoder layers
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            layers = self.encoder.encoder.layer
            for i in range(min(num_layers, len(layers))):
                for param in layers[i].parameters():
                    param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode text to fixed-size embeddings.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Pooled embeddings [batch, hidden_size]
        """
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Get sequence output (all token embeddings)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # Pool to single vector
        if self.pooling == "cls":
            # Use [CLS] token (first token)
            pooled = sequence_output[:, 0, :]

        elif self.pooling == "mean":
            # Mean over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size())
            sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask

        elif self.pooling == "max":
            # Max over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size())
            sequence_output[mask_expanded == 0] = -1e9  # Mask padding
            pooled = torch.max(sequence_output, dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        return pooled


if __name__ == "__main__":
    # Quick test
    from transformers import AutoTokenizer

    print("Testing TextEncoder...")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    encoder = TextEncoder(
        model_name="microsoft/deberta-v3-base",
        pooling="mean",
    )

    # Test input
    text = "The song takes place in a vaudeville theater where a spiritualist has tried to levitate."
    encoding = tokenizer(text, return_tensors="pt", padding=True)

    # Forward pass
    with torch.no_grad():
        embeddings = encoder(
            input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"]
        )

    print(f"\nInput shape: {encoding['input_ids'].shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Expected: [1, {encoder.hidden_size}]")
    print(
        "✓ Test passed!"
        if embeddings.shape == (1, encoder.hidden_size)
        else "✗ Test failed!"
    )
