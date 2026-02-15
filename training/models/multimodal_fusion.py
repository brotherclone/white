"""
Canonical MultimodalFusionModel definition.

This is the authoritative model architecture used by:
- training/modal_midi_fusion.py (training on Modal GPU)
- training/export_onnx.py (ONNX export on Modal)
- training/chromatic_scorer.py (CPU inference via ONNX)

Architecture:
    PianoRollEncoder CNN (1.1M params, unfrozen during training)
    + multimodal fusion MLP (3.2M params)
    + 4 regression heads (temporal, spatial, ontological, confidence)

Input: [audio 512 + MIDI 512 + concept 768 + lyric 768] = 2560-dim
Output: temporal [3], spatial [3], ontological [3], confidence [1]
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:

    class PianoRollEncoder(nn.Module):
        """CNN: [batch, 1, 128, 256] -> [batch, 512]"""

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
        """Full fusion model: piano roll CNN + null embeddings + fusion MLP + heads.

        During training: modality dropout randomly masks present modalities.
        During inference (eval mode): no dropout, null embeddings used for absent modalities.
        """

        def __init__(self, modality_dropout=0.15):
            super().__init__()
            self.modality_dropout = modality_dropout

            self.midi_encoder = PianoRollEncoder(output_dim=512)

            # Learned null embeddings for missing modalities
            self.null_audio = nn.Parameter(torch.randn(512) * 0.02)
            self.null_midi = nn.Parameter(torch.randn(512) * 0.02)
            self.null_concept = nn.Parameter(torch.randn(768) * 0.02)
            self.null_lyric = nn.Parameter(torch.randn(768) * 0.02)

            # Fusion MLP: [audio 512 + midi 512 + concept 768 + lyric 768] = 2560
            self.fusion = nn.Sequential(
                nn.Linear(2560, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
            )

            # Regression heads (softmax over 3 modes each)
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

            if self.training and self.modality_dropout > 0:
                drop_audio = (
                    torch.rand(batch_size, 1, device=piano_roll.device)
                    < self.modality_dropout
                )
                drop_midi = (
                    torch.rand(batch_size, 1, device=piano_roll.device)
                    < self.modality_dropout
                )
                drop_lyric = (
                    torch.rand(batch_size, 1, device=piano_roll.device)
                    < self.modality_dropout
                )
                audio_mask = audio_mask & ~drop_audio
                midi_mask = midi_mask & ~drop_midi
                lyric_mask = lyric_mask & ~drop_lyric

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

            return {
                "temporal": F.softmax(self.temporal_head(fused), dim=-1),
                "spatial": F.softmax(self.spatial_head(fused), dim=-1),
                "ontological": F.softmax(self.ontological_head(fused), dim=-1),
                "confidence": torch.sigmoid(self.confidence_head(fused)),
            }

else:

    class PianoRollEncoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch is required for PianoRollEncoder")

    class MultimodalFusionModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch is required for MultimodalFusionModel")
