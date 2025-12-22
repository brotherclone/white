# python
"""
Text encoder using DeBERTa.

Wraps HuggingFace transformers with configurable pooling and layer freezing.
"""
import re
import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """Text encoder using pre-trained transformer."""

    def __init__(self, model_name: str, **kwargs):
        """
        Load model while preventing unexpected kwargs from being forwarded to the model __init__.
        Supported custom kwargs (will be removed before calling the HF loader):
          - pooling: 'cls' | 'mean' | 'max' (default: 'cls')
          - freeze_layers: int (optional)
          - use_safetensors: bool (default: True)
        """
        super().__init__()

        # Handle and remove custom kwargs so they are not forwarded to model __init__
        self.pooling = kwargs.pop("pooling", "cls")
        use_safetensors = kwargs.pop("use_safetensors", True)
        freeze_layers = kwargs.pop("freeze_layers", None)

        if self.pooling not in ("cls", "mean", "max"):
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Load model (safetensors preferred)
        self.encoder = AutoModel.from_pretrained(
            model_name, use_safetensors=use_safetensors, **kwargs
        )

        # Expose hidden size for consumers
        self.hidden_size = getattr(self.encoder.config, "hidden_size", None)

        # Optionally freeze layers using the helper
        if freeze_layers is not None:
            try:
                n = int(freeze_layers)
            except Exception:
                n = None
            if n and n > 0:
                self._freeze_layers(n)

    def _freeze_layers(self, num_layers: int):
        """Freeze the first N encoder layers."""
        print(f"Freezing first {num_layers} encoder layers")

        # Freeze embeddings if available
        if hasattr(self.encoder, "embeddings"):
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False

        # Freeze encoder layers for common HF naming (e.g. encoder.layer)
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            layers = self.encoder.encoder.layer
            for i in range(min(num_layers, len(layers))):
                for param in layers[i].parameters():
                    param.requires_grad = False
        else:
            # Generic fallback: try to freeze by name pattern
            for name, param in self.encoder.named_parameters():
                m = re.search(r"\.layer\.(\d+)\.", name) or re.search(
                    r"\.layers\.(\d+)\.", name
                )
                if not m and "encoder" in name:
                    m = re.search(r"\.(\d+)\.", name)
                if m:
                    try:
                        idx = int(m.group(1))
                        if idx < num_layers:
                            param.requires_grad = False
                    except Exception:
                        pass

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
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]

        if self.pooling == "cls":
            pooled = sequence_output[:, 0, :]

        elif self.pooling == "mean":
            mask_expanded = (
                attention_mask.unsqueeze(-1).expand_as(sequence_output).float()
            )
            sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask

        elif self.pooling == "max":
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(sequence_output)
            sequence_output = sequence_output.masked_fill(mask_expanded == 0, -1e9)
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
