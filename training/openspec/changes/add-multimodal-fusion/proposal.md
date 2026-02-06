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

### 3. Lyric Encoder (Three-Pronged)

This is the key innovation - capture lyrics at three levels:

#### 3a. Semantic Encoding (What It Means)
```python
semantic_emb = deberta_model(lyrics_text)  # [batch, 768]
```
- Existing DeBERTa-v3 text encoder
- Captures vocabulary, themes, conceptual content
- Example: "spatial" "place" "geography" → GREEN

#### 3b. Prosodic Encoding (How It's Delivered)
```python
# Requires forced alignment (Montreal Forced Aligner, Gentle)
alignment = forced_aligner(audio, lyrics_text, midi)

prosodic_features = extract_prosody(alignment):
    - pitch_contour_per_syllable: [N_syllables, 3]  # mean, std, range
    - note_duration_per_phoneme: [N_phonemes, 1]
    - stress_pattern_matching: [N_words, 1]  # does pitch align with stress?
    - melisma_detection: [N_syllables, 1]  # single note vs multiple
    - legato_vs_staccato: [N_phrases, 1]  # note overlap with next syllable
    
prosodic_emb = mlp(prosodic_features)  # [batch, 256]
```

**Chromatic implications**:
- GREEN: Sustained notes on place names, legato phrasing, sparse syllables
- RED: Rapid syllabic delivery, staccato, syncopation
- VIOLET: Rubato, unexpected pitch leaps, metric modulation

#### 3c. Structural Encoding (Form/Rhythm)
```python
# No alignment needed - just counting
structural_features = {
    'notes_per_syllable': len(midi_notes) / syllable_count,
    'melisma_ratio': melismatic_notes / total_notes,
    'syllabic_density': syllables_per_measure,
    'rhythmic_alignment': onset_sync_score,  # syllable onsets match beat grid?
    'phrase_length_variance': std(phrase_lengths),
    'repetition_ratio': repeated_phrases / total_phrases,
}

structural_emb = mlp(structural_features)  # [batch, 128]
```

**Combined Lyric Embedding**:
```python
lyric_emb = concat([semantic_emb, prosodic_emb, structural_emb])  # [batch, 768+256+128=1152]
```

### 4. Multimodal Fusion

**Fusion Strategy Options**:

#### Early Fusion (Recommended for Phase 3)
```python
class EarlyFusionModel(nn.Module):
    def __init__(self):
        self.audio_encoder = Wav2Vec2Encoder()  # → [batch, 768]
        self.midi_encoder = PianoRollEncoder()  # → [batch, 512]
        self.lyric_encoder = ThreeProngLyricEncoder()  # → [batch, 1152]
        
        # Concatenate all modalities
        self.fusion = nn.Sequential(
            nn.Linear(768 + 512 + 1152, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512)
        )
        
        # Reuse existing Phase 4 heads
        self.temporal_head = RegressionHead(512, 3)
        self.spatial_head = RegressionHead(512, 3)
        self.ontological_head = RegressionHead(512, 3)
    
    def forward(self, audio, midi, lyrics_text, lyrics_prosody, lyrics_structure):
        audio_emb = self.audio_encoder(audio)
        midi_emb = self.midi_encoder(midi)
        
        # Three-pronged lyric encoding
        semantic = self.deberta(lyrics_text)
        prosodic = self.prosody_mlp(lyrics_prosody)
        structural = self.structure_mlp(lyrics_structure)
        lyric_emb = torch.cat([semantic, prosodic, structural], dim=-1)
        
        # Fuse all modalities
        fused = self.fusion(torch.cat([audio_emb, midi_emb, lyric_emb], dim=-1))
        
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

### Phase 3.1: Audio + MIDI (No Lyrics)
- Add audio encoder (Wav2Vec2)
- Add MIDI encoder (piano roll CNN)
- Early fusion
- Retrain Phase 4 regression heads
- **Goal**: Fix spatial mode accuracy for instrumental tracks

### Phase 3.2: Add Semantic Lyrics
- Add DeBERTa encoder (already exists)
- Concat with audio/MIDI
- **Goal**: Improve overall accuracy with text context

### Phase 3.3: Add Prosodic Lyrics
- Set up forced alignment pipeline (Montreal Forced Aligner)
- Extract prosodic features from alignment
- Add prosody MLP
- **Goal**: Capture vocal delivery style (GREEN legato vs RED staccato)

### Phase 3.4: Add Structural Lyrics
- Extract syllable counts, rhythmic stats
- Add structure MLP
- **Goal**: Capture phrasing patterns, rhythmic density

## Dataset Updates

Current: `rainbow_table_segments.parquet` has:
- `segment_id`, `concept_text`, `album`, `temporal_mode`, `spatial_mode`, `ontological_mode`

**Need to add**:
```python
# Preprocessing script: extract_multimodal_features.py

for segment in segments:
    # Audio
    audio_path = f"audio/{segment.album}/{segment.track_id}.wav"
    audio_features = extract_audio(audio_path, segment.start_time, segment.end_time)
    
    # MIDI
    midi_path = f"midi/{segment.album}/{segment.track_id}.mid"
    midi_features = extract_midi(midi_path, segment.start_time, segment.end_time)
    
    # Lyrics - Prosodic (if vocal track)
    if segment.has_vocals:
        alignment = forced_aligner.align(audio_path, segment.lyrics_text)
        prosody = extract_prosody(alignment, midi_features)
    else:
        prosody = None
    
    # Lyrics - Structural
    structure = extract_structure(segment.lyrics_text, midi_features)
    
    # Save
    segment.audio_embedding = audio_features
    segment.midi_embedding = midi_features
    segment.prosody_features = prosody
    segment.structure_features = structure
```

**New dataset schema**:
```
rainbow_table_multimodal.parquet:
  - segment_id
  - concept_text
  - album
  - temporal_mode, spatial_mode, ontological_mode
  - audio_embedding: [768] float array
  - midi_embedding: [512] float array
  - prosody_features: [256] float array (nullable for instrumental)
  - structure_features: [128] float array
```

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
            lyric_emb = self.model.lyric_encoder(lyrics, midi_candidate, audio_render)
        else:
            lyric_emb = torch.zeros(1152)  # Null encoding for instrumental
        
        # Predict
        temporal, spatial, ontological = self.model(audio_emb, midi_emb, lyric_emb)
        
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
montreal-forced-aligner  # Prosodic alignment
torch-audiomentations  # Data augmentation
```

**External tools**:
- Montreal Forced Aligner: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner
- Or Gentle: https://github.com/lowerquality/gentle (Python-based alternative)

## Breaking Changes

- Dataset format changes (add multimodal columns)
- Model architecture changes (multimodal input)
- Training scripts need updates (load audio/MIDI, not just text)
- Inference API changes (accept MIDI + audio, not just text)

## Timeline Estimate

- **Phase 3.1** (Audio + MIDI): 1-2 weeks
  - Audio encoder integration: 3 days
  - MIDI encoder implementation: 4 days
  - Data preprocessing pipeline: 3 days
  - Training + evaluation: 3 days

- **Phase 3.2** (Add semantic lyrics): 2 days
  - Already have DeBERTa encoder
  - Just concatenation + retrain

- **Phase 3.3** (Add prosodic lyrics): 1 week
  - Forced alignment setup: 3 days
  - Feature extraction pipeline: 2 days
  - Training + evaluation: 2 days

- **Phase 3.4** (Add structural lyrics): 2 days
  - Feature extraction (counting): 1 day
  - Training + evaluation: 1 day

**Total**: 3-4 weeks for full multimodal implementation