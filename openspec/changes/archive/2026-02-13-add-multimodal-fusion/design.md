## Context

The chromatic fitness function currently uses text-only DeBERTa embeddings. Spatial mode accuracy is bottlenecked at 62% because instrumental tracks (Yellow/Green) have no lyrics. Adding audio and MIDI modalities should fix this.

Key constraints:
- 10,544 training segments; 85% have audio, 43% have MIDI, ~60% have lyrics
- MIDI sparsity is the norm (57% absent), not an edge case
- Training on RunPod GPU instances — precomputation saves cost
- The model will score *generated* MIDI/audio at inference time, so MIDI will always be present during scoring

## Goals / Non-Goals

**Goals**:
- Audio + MIDI + text fusion model that improves spatial mode accuracy from 62% to >85%
- Handle missing modalities gracefully (majority of training data is incomplete)
- Fast iteration cycle — swap encoders without retraining everything

**Non-Goals**:
- Prosodic or structural lyric encoding (deferred to `add-prosodic-lyric-encoding`)
- Real-time inference optimization (that's Phase 10)
- Training the audio/MIDI encoders from scratch

## Decisions

### D1: Audio Encoder — CLAP (`laion/larger_clap_music`)

**Decision**: Use CLAP (Contrastive Language-Audio Pretraining), specifically the music-only variant.

**Why**:
- **Music-specific**: Trained on music audio, unlike Wav2Vec2 which was trained on LibriSpeech speech data
- **Text-aligned embedding space**: CLAP was contrastively trained so audio embeddings live in the same space as text embeddings — this gives us cross-modal alignment for free before fusion even happens
- **License**: Apache-2.0 (permissive)
- **Output**: `[batch, 512]` audio embedding (single vector per segment, no pooling needed)
- **Integration**: Supported in HuggingFace `transformers` library
- **Downloads**: ~490K on HuggingFace, well-maintained

**Alternatives considered**:
- `m-a-p/MERT-v1-95M`: Purpose-built music encoder, 768-dim output, 4.3M downloads. Richer representations but CC-BY-NC-4.0 license and requires mean-pooling over sequence. **Fallback option** if CLAP underperforms.
- `facebook/wav2vec2-base-960h`: Very popular (105M downloads) but trained on English speech, not music. Wrong domain.
- Custom CNN on mel spectrograms: No pretraining benefit. Only consider if pretrained models overfit to our small dataset.

**Usage**:
```python
from transformers import ClapModel, ClapProcessor

model = ClapModel.from_pretrained("laion/larger_clap_music")
processor = ClapProcessor.from_pretrained("laion/larger_clap_music")

# Precompute: run once per segment, store the 512-dim embedding
inputs = processor(audios=waveform, sampling_rate=48000, return_tensors="pt")
audio_emb = model.get_audio_features(**inputs)  # [1, 512]
```

### D2: MIDI Encoder — Piano Roll CNN

**Decision**: Convert MIDI binary to piano roll matrix (128 pitch bins x time steps), process with a lightweight CNN.

**Why**:
- **Simple and debuggable**: Piano roll is a visual representation — easy to inspect and verify
- **Proven**: Standard approach in MIR (Music Information Retrieval) literature
- **Existing infrastructure**: `app/util/midi_segment_utils.py` already parses MIDI binary from the parquet
- **Output**: `[batch, 512]` MIDI embedding (matching CLAP audio dim for symmetry)

**Alternatives considered**:
- Event-based tokenization + transformer: More expressive but significantly more complex (vocab design, variable length sequences, positional encoding). **Upgrade path** if piano roll doesn't capture enough harmonic structure.
- `m-a-p/MERT` on rendered MIDI audio: Render MIDI to audio, then use audio encoder. Loses MIDI-specific information (velocity, note boundaries) and adds latency.

**Architecture**:
```python
class PianoRollEncoder(nn.Module):
    def __init__(self, output_dim=512, time_steps=256):
        # Piano roll: [batch, 1, 128, time_steps] (like grayscale image)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),              # [batch, 32, 64, T/2]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),              # [batch, 64, 32, T/4]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), # [batch, 128, 4, 4]
        )
        self.fc = nn.Linear(128 * 4 * 4, output_dim)  # → [batch, 512]
```

**Piano roll parameters**:
- Pitch bins: 128 (full MIDI range)
- Time resolution: 100ms per step (segment duration / time_steps)
- Values: Velocity (0-127) normalized to [0, 1], or binary (note on/off)
- Max time steps: 256 (covers ~25 seconds at 100ms resolution)

### D3: Missing Modality — Learned Null Embeddings

**Decision**: Use trainable null embedding vectors per modality instead of zero-fill.

**Why**:
- With 57% MIDI-absent and 15% audio-absent, the model sees "missing" more than "present" for MIDI
- A learned null embedding lets the model distinguish "no MIDI data available" from "the MIDI segment was silent/empty"
- Zero-fill conflates absence with silence, which could confuse the fusion layers

**Implementation**:
```python
class ModalityAwareFusion(nn.Module):
    def __init__(self):
        # Learned null embeddings (one per modality)
        self.null_audio = nn.Parameter(torch.randn(512))
        self.null_midi = nn.Parameter(torch.randn(512))
        self.null_text = nn.Parameter(torch.randn(768))

    def forward(self, audio_emb, midi_emb, text_emb, modality_mask):
        # modality_mask: [batch, 3] boolean — [text, audio, midi]
        # Substitute null embeddings where modality is absent
        audio_emb = torch.where(
            modality_mask[:, 1:2], audio_emb, self.null_audio.expand_as(audio_emb)
        )
        midi_emb = torch.where(
            modality_mask[:, 2:3], midi_emb, self.null_midi.expand_as(midi_emb)
        )
        text_emb = torch.where(
            modality_mask[:, 0:1], text_emb, self.null_text.expand_as(text_emb)
        )
        return torch.cat([audio_emb, midi_emb, text_emb], dim=-1)
```

**Training with modality dropout**: During training, randomly zero out present modalities (substitute null embedding) with probability p=0.15. This forces the model to not over-rely on any single modality and prepares it for real missing-data scenarios.

### D4: Training Strategy — Precompute Then Fuse

**Decision**: Two-stage approach. Precompute encoder outputs once, then train a lightweight fusion model.

**Why**:
- Running CLAP on every batch every epoch is expensive (~2 seconds per forward pass per batch on GPU)
- Precomputing once means fusion training is fast (minutes not hours)
- Enables rapid iteration on fusion architecture without re-encoding
- Can swap CLAP for MERT later without touching fusion code

**Stage 1: Precompute embeddings** (run once):
```
For each segment:
  audio_emb = CLAP.get_audio_features(audio_waveform)  → [512] float32
  midi_emb = PianoRollCNN(piano_roll_matrix)            → [512] float32
  text_emb = load from re-embedded parquet              → [768] float16

Store as: training_segments_embeddings.parquet
  - segment_id
  - audio_embedding: [512] (nullable)
  - midi_embedding: [512] (nullable)
  - text_embedding: [768] (nullable)
  - modality_mask: [3] boolean
```

**Stage 2: Train fusion** (iterative):
```
Load precomputed embeddings → fast DataLoader
Train: fusion MLP + regression heads
Iterate: try different fusion strategies, dropout rates, learning rates
```

**When to re-encode**: Only when switching the underlying encoder (e.g., CLAP → MERT). The fusion model is decoupled from the encoder choice.

### D5: Fusion Architecture — Late Concatenation with Projection

**Decision**: Concatenate modality embeddings, project through shared MLP, feed to existing regression heads.

**Dimensions**:
```
audio_emb:  [batch, 512]  (CLAP)
midi_emb:   [batch, 512]  (Piano Roll CNN)
text_emb:   [batch, 768]  (DeBERTa)
                           ─────────
concat:     [batch, 1792]
  → Linear(1792, 1024) → ReLU → Dropout(0.3)
  → Linear(1024, 512)
  → RegressionHead(512, 3) × 3 (temporal, spatial, ontological)
```

**Why late fusion**: Modality-specific encoders are pretrained and frozen (or lightly fine-tuned). Concatenation preserves modality-specific information without forcing alignment through shared layers. This is the simplest approach that works.

**Upgrade path**: If late fusion plateaus, try gated fusion (learnable importance weights per modality per sample) before investing in cross-attention.

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| CLAP embeddings don't capture chromatic distinctions | Model doesn't improve over text-only | Fall back to MERT-95M; or fine-tune CLAP audio encoder on our data |
| 57% MIDI-absent hurts MIDI encoder training | MIDI encoder doesn't learn useful features | Modality dropout + null embeddings; at inference MIDI is always present |
| Piano roll loses harmonic detail | Chord voicing/inversion not captured | Upgrade to event-based tokenization if ablation shows MIDI contribution is low |
| Precomputed embeddings prevent end-to-end fine-tuning | Ceiling on accuracy | After finding best fusion architecture, do one final end-to-end training run with unfrozen encoders and lower learning rates |
| Blue album has only 12% MIDI | Blue systematically worse | Acceptable — Blue has 96% audio coverage which should compensate |

## Resolved Questions

1. **Audio sample rate**: All 10,544 segments are 44,100 Hz. CLAP expects 48kHz, so we need a resample step during precomputation (librosa or torchaudio). One-time cost.

## Open Questions

1. **Piano roll time resolution**: 100ms is a starting point. Should we try finer (50ms) or coarser (200ms) resolution?
2. **Regression head reuse**: Should we reuse the Phase 4 trained regression head weights as initialization, or train from scratch on the new fused representation?
