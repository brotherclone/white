"""
Refractor CDM (Chromatic Distribution Model) — full-mix calibration head.

A small MLP trained on CLAP embeddings from staged_raw_material/_main.wav files.
Bridges the acoustic gap between the base Refractor (trained on 10–30s catalog
segments) and full produced mixes.

Architecture:
    Input: CLAP 512-dim + optional concept 768-dim = 512 or 1280-dim
    Hidden: Linear(input_dim → 256) → ReLU → Dropout(0.3)
            Linear(256 → 128)       → ReLU → Dropout(0.3)
    Heads:  three independent Linear(128 → 3) + softmax
            (temporal, spatial, ontological)

Output: three 3-dim softmax distributions matching CHROMATIC_TARGETS axes.
ONNX outputs named: "temporal", "spatial", "ontological"

Params: ~50K (without concept), ~250K (with concept)
"""

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:

    class RefractorCDMModel(nn.Module):
        """Calibration MLP: CLAP [+ concept] → temporal/spatial/ontological."""

        def __init__(
            self,
            clap_dim: int = 512,
            concept_dim: int = 768,
            use_concept: bool = True,
            hidden_dims: list[int] | None = None,
            dropout: float = 0.3,
        ):
            super().__init__()
            if hidden_dims is None:
                hidden_dims = [256, 128]

            self.use_concept = use_concept
            input_dim = clap_dim + (concept_dim if use_concept else 0)

            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            self.shared = nn.Sequential(*layers)

            # Three independent heads — avoids cross-task gradient interference
            self.temporal_head = nn.Linear(prev, 3)
            self.spatial_head = nn.Linear(prev, 3)
            self.ontological_head = nn.Linear(prev, 3)

        def forward(
            self,
            clap_emb: "torch.Tensor",
            concept_emb: "torch.Tensor | None" = None,
        ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """
            Args:
                clap_emb: (batch, 512) float32
                concept_emb: (batch, 768) float32, or None if use_concept=False

            Returns:
                (temporal, spatial, ontological) — each (batch, 3) softmax probabilities
            """
            if self.use_concept and concept_emb is not None:
                x = torch.cat([clap_emb, concept_emb], dim=-1)
            else:
                x = clap_emb

            h = self.shared(x)
            temporal = torch.softmax(self.temporal_head(h), dim=-1)
            spatial = torch.softmax(self.spatial_head(h), dim=-1)
            ontological = torch.softmax(self.ontological_head(h), dim=-1)
            return temporal, spatial, ontological


def export_onnx(
    model: "RefractorCDMModel",
    path: str,
    use_concept: bool = True,
) -> None:
    """Export a trained RefractorCDMModel to ONNX.

    Args:
        model: Trained model in eval mode.
        path: Output .onnx file path.
        use_concept: Whether model expects a concept embedding input.
    """
    import torch

    model.eval()
    clap_dummy = torch.zeros(1, 512, dtype=torch.float32)

    if use_concept:
        concept_dummy = torch.zeros(1, 768, dtype=torch.float32)
        input_names = ["clap_emb", "concept_emb"]
        dynamic_axes = {
            "clap_emb": {0: "batch"},
            "concept_emb": {0: "batch"},
            "temporal": {0: "batch"},
            "spatial": {0: "batch"},
            "ontological": {0: "batch"},
        }
        dummy_inputs = (clap_dummy, concept_dummy)
    else:
        input_names = ["clap_emb"]
        dynamic_axes = {
            "clap_emb": {0: "batch"},
            "temporal": {0: "batch"},
            "spatial": {0: "batch"},
            "ontological": {0: "batch"},
        }
        dummy_inputs = (clap_dummy,)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_inputs,
            path,
            input_names=input_names,
            output_names=["temporal", "spatial", "ontological"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
        )
    print(f"Exported RefractorCDM ONNX → {path}")
