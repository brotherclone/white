# Add Multimodal Fusion

## Problem

The chromatic fitness function (used by the Evolutionary Music Generator to score candidates) currently relies on **text embeddings only**. This causes:

1. **Spatial mode bottleneck**: 62% accuracy because GREEN/YELLOW albums ("Place") are instrumental - no lyrics = no text embeddings
2. **Missing audio semantics**: "What does GREEN *sound* like?" is unanswered
3. **Missing rhythmic/melodic patterns**: MIDI structure ignored
4. **Lyric-melody disconnect**: Prosody, syllabic density, and vocal rhythm not captured

## Solution

Add three modality encoders with multi-pronged lyric representation:

### 1. Audio Encoder

**Options** (choose one):
- **Wav2Vec2**: Pretrained speech representation, fine-tune for music
- **CLAP**: Contrastive Language-Audio Pretraining, connects audio to text
- **MusicFM**: Music foundation model (if available)
- **Custom CNN**: Mel-spectrogram → conv layers

**Output**: `[batch, 768]` audio embedding

**Preprocessing**:
```python
# Convert audio to 16kHz mono, extract 30-second segments
# Compute mel-spectrogram or feed directly to pretrained model
```

### 2. MIDI Encoder

**Options** (choose one):
- **Piano roll**: 128 pitch bins × time steps → CNN or LSTM
- **Event-based**: Tokenize MIDI events, feed to transformer
- **Music Transformer**: Pretrained on MIDI corpus

**Output**: `[batch, 512]` MIDI embedding

**Features to capture**:
- Chord voicings (open vs closed, inversions)
- Rhythmic density (notes per measure)
- Melodic contour (stepwise vs leaps)
- Harmonic progression patterns

### 3. Text Encoder (Existing DeBERTa)

Reuses the existing DeBERTa-v3 text encoder from Phases 1-4:
```python
text_emb = deberta_model(lyrics_text)  # [batch, 768]
```
- Captures vocabulary, themes, conceptual content
- Example: "spatial" "place" "geography" → GREEN
- For instrumental segments (no lyrics): zero tensor with modality mask=False

**Future extension**: Prosodic and structural lyric encoding (see `add-prosodic-lyric-encoding` change) can replace this with a richer `[batch, 1152]` representation once Phase 3.1/3.2 results validate the approach.

### 4. Multimodal Fusion

**Fusion Strategy Options**:

#### Late Fusion / Concatenation (Recommended for Phase 3)
```python
class ConcatFusionModel(nn.Module):
    def __init__(self):
        self.audio_encoder = Wav2Vec2Encoder()     # → [batch, 768]
        self.midi_encoder = PianoRollEncoder()      # → [batch, 512]
        self.text_encoder = DeBERTaEncoder()        # → [batch, 768]

        # Concatenate all modalities
        self.fusion = nn.Sequential(
            nn.Linear(768 + 512 + 768, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512)
        )

        # Reuse existing Phase 4 heads
        self.temporal_head = RegressionHead(512, 3)
        self.spatial_head = RegressionHead(512, 3)
        self.ontological_head = RegressionHead(512, 3)

    def forward(self, audio, midi, text, modality_mask):
        audio_emb = self.audio_encoder(audio)   # zeroed if mask[audio]=False
        midi_emb = self.midi_encoder(midi)       # zeroed if mask[midi]=False
        text_emb = self.text_encoder(text)       # zeroed if mask[text]=False

        # Apply modality mask (zero out absent modalities)
        audio_emb = audio_emb * modality_mask[:, 1:2]  # audio
        midi_emb = midi_emb * modality_mask[:, 2:3]     # midi
        text_emb = text_emb * modality_mask[:, 0:1]     # text

        # Fuse all modalities
        fused = self.fusion(torch.cat([audio_emb, midi_emb, text_emb], dim=-1))

        # Predict ontological distributions
        temporal = self.temporal_head(fused)
        spatial = self.spatial_head(fused)
        ontological = self.ontological_head(fused)

        return temporal, spatial, ontological
```

#### Alternative: Cross-Modal Attention (Phase 3.5, if needed)
```python
class CrossAttentionFusion(nn.Module):
    def forward(self, audio_emb, midi_emb, lyric_emb):
        # Let audio attend to MIDI
        audio_to_midi = self.cross_attn(audio_emb, midi_emb, midi_emb)
        
        # Let MIDI attend to lyrics
        midi_to_lyrics = self.cross_attn(midi_emb, lyric_emb, lyric_emb)
        
        # Combine with residual connections
        fused = audio_emb + audio_to_midi + midi_to_lyrics
        return fused
```

## Implementation Phases

### Prerequisites
- `prepare-multimodal-data` change must be complete (text embeddings, coverage verification)

### Phase 3.1: Audio + MIDI (No Lyrics)
- Add audio encoder (Wav2Vec2)
- Add MIDI encoder (piano roll CNN)
- Late fusion (concatenate embeddings) with modality mask
- Retrain Phase 4 regression heads
- **Goal**: Fix spatial mode accuracy for instrumental tracks

### Phase 3.2: Add Semantic Text
- Wire existing DeBERTa encoder into fusion model
- Three-modality late fusion: audio + MIDI + text
- **Goal**: Improve overall accuracy with text context

### Future: Prosodic + Structural Lyrics
See `add-prosodic-lyric-encoding` change (deferred until Phase 3.1/3.2 results validate the approach).

## Dataset Updates

### Current State (as of 2026-02-07)

The extraction pipeline was re-run on 2026-02-06 after MIDI bug fixes. Data now lives in two split files:

| File | Rows | Purpose |
|------|------|---------|
| `training/data/training_segments_metadata.parquet` | 10,544 | All columns except binary blobs (fast queries) |
| `training/data/training_segments_media.parquet` | 10,544 | Full columns including `audio_waveform` and `midi_binary` |

**Coverage**:
| Modality | Segments | % of total | Notes |
|----------|----------|------------|-------|
| Audio (`audio_waveform`) | 8,972 | 85.1% | 1,572 segments missing audio |
| MIDI (`midi_binary`) | 4,563 | **43.3%** | Majority-missing — design for sparsity |
| Text (`lyric_text`) | varies | ~60% | Instrumental tracks have no lyrics |
| Text embeddings | **0** | **0%** | Old file had them; new extraction needs re-embed |

**MIDI coverage by album color**:
| Color | MIDI | Total | % |
|-------|------|-------|---|
| Black | 1,064 | 1,777 | 60% |
| Orange | 810 | 1,493 | 54% |
| Red | 640 | 1,326 | 48% |
| UNLABELED | 1,648 | 3,506 | 47% |
| Yellow | 147 | 345 | 43% |
| Blue | 254 | 2,097 | **12%** |

**Missing colors**: Green, Indigo, Violet are absent from the dataset entirely.

**Key columns** (metadata parquet):
- `segment_id`, `song_id`, `track_id`
- `rainbow_color`, `rainbow_color_temporal_mode`, `rainbow_color_ontological_mode`, `rainbow_color_objectional_mode`
- `has_audio`, `has_midi`, `midi_file`, `source_audio_file`, `segment_audio_file`
- `lyric_text`, `content_type`, `vocals`
- `bpm`, `key_signature_note`, `key_signature_mode`

**Key columns** (media parquet):
- `audio_waveform` (binary), `audio_sample_rate` (int32)
- `midi_binary` (binary, nullable — 57% null)

**Legacy file**: `training_data_with_embeddings.parquet` (8,972 rows) has DeBERTa embeddings in an `embedding` column but `midi_binary` is entirely null (pre-bugfix). Kept for reference but should not be used for multimodal training.

### Preprocessing Pipeline

```python
# Phase 3.0: Re-embed new extraction
# Uses existing core/embedding_loader.py DeBERTaEmbeddingEncoder

# Phase 3.1: Extract audio + MIDI features from media parquet
# Audio: decode audio_waveform binary → resample → encode
# MIDI: decode midi_binary → parse via app/util/midi_segment_utils.py → encode
# Missing modalities: zero-tensor with modality mask flag
```

### Design Constraint: MIDI Sparsity

With only 43% MIDI coverage, the model MUST treat MIDI as optional:
- Use a **modality presence mask** `[has_text, has_audio, has_midi]` as model input
- **Modality dropout** during training simulates missing MIDI (already planned)
- Blue album (12% MIDI) will rely almost entirely on audio + text
- At inference time (scoring music candidates), MIDI will always be present since we're scoring generated MIDI

## Success Metrics

Current Phase 4 (text-only):
- Temporal: 95% accuracy
- Ontological: 93% accuracy
- Spatial: **62% accuracy** ← BOTTLENECK

Target Phase 3 (multimodal):
- Temporal: **>97%** (audio temporal features help)
- Ontological: **>95%** (richer representation)
- Spatial: **>90%** (audio fixes instrumental tracks)

## Integration with Music Production Agent

Once trained, this model becomes the fitness function:
```python
# In app/generator/evolutionary_composer.py

class ChromaticScorer:
    def __init__(self, model_path):
        self.model = load_multimodal_model(model_path)
    
    def score(self, midi_candidate, audio_render, lyrics=None):
        """
        Score a music candidate for chromatic consistency.
        
        Args:
            midi_candidate: MIDI file or piano roll
            audio_render: Rendered audio (or None, render MIDI)
            lyrics: Optional lyrics text
        
        Returns:
            {
                'temporal_dist': [p_past, p_present, p_future],
                'spatial_dist': [p_place, p_motion, p_transcend],
                'ontological_dist': [p_info, p_time, p_space],
                'confidence': float,
                'predicted_album': str  # "GREEN", "RED", etc.
            }
        """
        # Extract features
        audio_emb = self.model.audio_encoder(audio_render)
        midi_emb = self.model.midi_encoder(midi_candidate)
        
        if lyrics:
            text_emb = self.model.text_encoder(lyrics)
        else:
            text_emb = torch.zeros(768)  # Null encoding for instrumental
        
        # Predict
        temporal, spatial, ontological = self.model(audio_emb, midi_emb, text_emb)
        
        return {
            'temporal_dist': temporal.softmax(-1).tolist(),
            'spatial_dist': spatial.softmax(-1).tolist(),
            'ontological_dist': ontological.softmax(-1).tolist(),
            'confidence': min(temporal.max(), spatial.max(), ontological.max()),
            'predicted_album': self.mode_to_album(temporal, spatial, ontological)
        }
```

## Dependencies

**Python packages**:
```
transformers  # Wav2Vec2, DeBERTa
librosa  # Audio processing
pretty_midi  # MIDI parsing
torch-audiomentations  # Data augmentation (optional)
```

## Breaking Changes

- Dataset format changes (add multimodal columns)
- Model architecture changes (multimodal input)
- Training scripts need updates (load audio/MIDI, not just text)
- Inference API changes (accept MIDI + audio, not just text)

## Related Changes

- **Prerequisite**: `prepare-multimodal-data` (data readiness — embeddings, coverage verification)
- **Follow-up**: `add-prosodic-lyric-encoding` (Phases 3.3/3.4 — deferred until results validate approach)