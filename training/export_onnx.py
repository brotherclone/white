#!/usr/bin/env python3
"""
Export the multimodal fusion model to ONNX format.

Runs on Modal (needs torch). Returns the .onnx file locally.

Usage:
    modal run training/export_onnx.py
    modal run training/export_onnx.py --verify  # also run onnxruntime verification
"""

import modal

app = modal.App("white-onnx-export")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "numpy", "onnx", "onnxruntime", "onnxscript"
)


@app.function(image=image, timeout=300)
def export_fusion_model(model_weights: bytes, verify: bool = False):
    """Load fusion model, export to ONNX, optionally verify."""
    import io

    import numpy as np
    import onnx
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ---- Reconstruct model class (must match modal_midi_fusion.py) ----

    class PianoRollEncoder(nn.Module):
        def __init__(self, output_dim=512):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.fc = nn.Sequential(
                nn.Linear(128 * 4 * 4, output_dim),
                nn.ReLU(),
            )

        def forward(self, x):
            features = self.conv(x)
            features = features.view(features.size(0), -1)
            return self.fc(features)

    class MultimodalFusionModel(nn.Module):
        def __init__(self, modality_dropout=0.15):
            super().__init__()
            self.modality_dropout = modality_dropout
            self.midi_encoder = PianoRollEncoder(output_dim=512)
            self.null_audio = nn.Parameter(torch.randn(512) * 0.02)
            self.null_midi = nn.Parameter(torch.randn(512) * 0.02)
            self.null_concept = nn.Parameter(torch.randn(768) * 0.02)
            self.null_lyric = nn.Parameter(torch.randn(768) * 0.02)
            self.fusion = nn.Sequential(
                nn.Linear(2560, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            self.temporal_head = nn.Linear(512, 3)
            self.spatial_head = nn.Linear(512, 3)
            self.ontological_head = nn.Linear(512, 3)
            self.confidence_head = nn.Linear(512, 1)

        def forward(
            self,
            piano_roll,
            audio_emb,
            concept_emb,
            lyric_emb,
            has_audio,
            has_midi,
            has_lyric,
        ):
            batch_size = piano_roll.size(0)
            midi_emb = self.midi_encoder(piano_roll)

            audio_mask = has_audio.unsqueeze(1)
            midi_mask = has_midi.unsqueeze(1)
            lyric_mask = has_lyric.unsqueeze(1)

            # No modality dropout at inference (model.eval())

            audio_emb = torch.where(
                audio_mask, audio_emb, self.null_audio.expand(batch_size, -1)
            )
            midi_emb = torch.where(
                midi_mask, midi_emb, self.null_midi.expand(batch_size, -1)
            )
            lyric_emb = torch.where(
                lyric_mask, lyric_emb, self.null_lyric.expand(batch_size, -1)
            )

            fused = torch.cat([audio_emb, midi_emb, concept_emb, lyric_emb], dim=-1)
            fused = self.fusion(fused)

            return (
                F.softmax(self.temporal_head(fused), dim=-1),
                F.softmax(self.spatial_head(fused), dim=-1),
                F.softmax(self.ontological_head(fused), dim=-1),
                torch.sigmoid(self.confidence_head(fused)),
            )

    # ---- Load weights ----
    print("Loading model weights...")
    checkpoint = torch.load(io.BytesIO(model_weights), map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    model = MultimodalFusionModel()
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} params")

    # ---- Export to ONNX ----
    print("Exporting to ONNX...")
    batch = 1
    dummy_inputs = (
        torch.randn(batch, 1, 128, 256),  # piano_roll
        torch.randn(batch, 512),  # audio_emb
        torch.randn(batch, 768),  # concept_emb
        torch.randn(batch, 768),  # lyric_emb
        torch.ones(batch, dtype=torch.bool),  # has_audio
        torch.ones(batch, dtype=torch.bool),  # has_midi
        torch.ones(batch, dtype=torch.bool),  # has_lyric
    )

    onnx_buf = io.BytesIO()
    torch.onnx.export(
        model,
        dummy_inputs,
        onnx_buf,
        input_names=[
            "piano_roll",
            "audio_emb",
            "concept_emb",
            "lyric_emb",
            "has_audio",
            "has_midi",
            "has_lyric",
        ],
        output_names=["temporal", "spatial", "ontological", "confidence"],
        dynamic_axes={
            "piano_roll": {0: "batch"},
            "audio_emb": {0: "batch"},
            "concept_emb": {0: "batch"},
            "lyric_emb": {0: "batch"},
            "has_audio": {0: "batch"},
            "has_midi": {0: "batch"},
            "has_lyric": {0: "batch"},
            "temporal": {0: "batch"},
            "spatial": {0: "batch"},
            "ontological": {0: "batch"},
            "confidence": {0: "batch"},
        },
        opset_version=17,
    )

    onnx_bytes = onnx_buf.getvalue()
    size_mb = len(onnx_bytes) / (1024 * 1024)
    print(f"ONNX model: {size_mb:.1f} MB")

    # ---- Validate ONNX structure ----
    print("Validating ONNX model...")
    onnx_model = onnx.load_from_string(onnx_bytes)
    onnx.checker.check_model(onnx_model)
    print("  ONNX checker: PASS")

    # Print input/output info
    for inp in onnx_model.graph.input:
        dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  Input:  {inp.name} {dims}")
    for out in onnx_model.graph.output:
        dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name} {dims}")

    # ---- Verify with onnxruntime ----
    if verify:
        import onnxruntime as ort

        print("\nVerifying with onnxruntime...")
        sess = ort.InferenceSession(onnx_bytes)

        # Run inference with dummy data
        inputs = {
            "piano_roll": np.random.randn(1, 1, 128, 256).astype(np.float32),
            "audio_emb": np.random.randn(1, 512).astype(np.float32),
            "concept_emb": np.random.randn(1, 768).astype(np.float32),
            "lyric_emb": np.random.randn(1, 768).astype(np.float32),
            "has_audio": np.array([True]),
            "has_midi": np.array([True]),
            "has_lyric": np.array([True]),
        }
        outputs = sess.run(None, inputs)
        temporal, spatial, ontological, confidence = outputs

        print(f"  temporal:      {temporal.shape} sum={temporal.sum():.4f}")
        print(f"  spatial:       {spatial.shape} sum={spatial.sum():.4f}")
        print(f"  ontological:   {ontological.shape} sum={ontological.sum():.4f}")
        print(f"  confidence:    {confidence.shape} val={confidence[0][0]:.4f}")

        # Verify softmax sums to 1
        assert abs(temporal.sum() - 1.0) < 0.001, f"temporal sum={temporal.sum()}"
        assert abs(spatial.sum() - 1.0) < 0.001, f"spatial sum={spatial.sum()}"
        assert (
            abs(ontological.sum() - 1.0) < 0.001
        ), f"ontological sum={ontological.sum()}"
        assert 0.0 <= confidence[0][0] <= 1.0, f"confidence={confidence[0][0]}"

        # Batch test
        batch_inputs = {
            k: np.tile(v, (4, *([1] * (v.ndim - 1)))) for k, v in inputs.items()
        }
        batch_outputs = sess.run(None, batch_inputs)
        print(f"  Batch test (4): {[o.shape for o in batch_outputs]}")

        print("  onnxruntime: PASS")

    return onnx_bytes


@app.local_entrypoint()
def main(verify: bool = True):
    """Export fusion model to ONNX."""
    from pathlib import Path

    data_dir = Path(__file__).parent / "data"
    model_path = data_dir / "fusion_model.pt"

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return

    print(f"Loading {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")
    model_bytes = model_path.read_bytes()

    onnx_bytes = export_fusion_model.remote(model_weights=model_bytes, verify=verify)

    output_path = data_dir / "fusion_model.onnx"
    output_path.write_bytes(onnx_bytes)

    size_mb = len(onnx_bytes) / (1024 * 1024)
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")
