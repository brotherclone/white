# Multimodal Fusion

## ADDED Requirements

### Requirement: Audio Encoding
The system SHALL encode audio waveforms into dense embeddings using pretrained or custom audio models.

#### Scenario: Wav2Vec2 encoding
- **WHEN** audio encoder type is set to "wav2vec2"
- **THEN** pretrained Wav2Vec2 model extracts contextualized audio representations

#### Scenario: CLAP encoding
- **WHEN** audio encoder type is set to "clap"
- **THEN** Contrastive Language-Audio Pretraining model produces embeddings aligned with text space

#### Scenario: Custom CNN encoding
- **WHEN** audio encoder type is set to "cnn"
- **THEN** mel spectrograms are processed by CNN to extract audio features

#### Scenario: Waveform to embedding
- **WHEN** a raw waveform tensor [batch, samples] is input
- **THEN** an embedding tensor [batch, hidden_dim] is output

### Requirement: MIDI Encoding
The system SHALL encode MIDI event sequences into dense embeddings capturing harmonic and rhythmic information.

#### Scenario: Piano roll encoding
- **WHEN** MIDI encoder type is set to "piano_roll"
- **THEN** MIDI is converted to pitch x time matrix and processed by CNN

#### Scenario: Event-based encoding
- **WHEN** MIDI encoder type is set to "event_based"
- **THEN** NOTE_ON/OFF events are tokenized and processed by transformer

#### Scenario: Music Transformer encoding
- **WHEN** MIDI encoder type is set to "music_transformer"
- **THEN** bar-aware positional encoding captures musical structure

#### Scenario: MIDI events to embedding
- **WHEN** a MIDI event tensor [batch, events, features] is input
- **THEN** an embedding tensor [batch, hidden_dim] is output

### Requirement: Multimodal Fusion Strategy
The system SHALL combine text, audio, and MIDI representations using configurable fusion strategies.

#### Scenario: Early fusion
- **WHEN** fusion strategy is "early"
- **THEN** raw inputs are concatenated before any encoding

#### Scenario: Late fusion
- **WHEN** fusion strategy is "late"
- **THEN** modality-specific embeddings are concatenated after independent encoding

#### Scenario: Cross-modal attention fusion
- **WHEN** fusion strategy is "cross_attention"
- **THEN** each modality attends to other modalities to capture interactions

#### Scenario: Gated fusion
- **WHEN** fusion strategy is "gated"
- **THEN** learnable gates determine importance weights for each modality

### Requirement: Audio Preprocessing Pipeline
The system SHALL preprocess audio waveforms to ensure consistent input format and enable data augmentation.

#### Scenario: Padding to fixed duration
- **WHEN** an audio segment is shorter than target duration
- **THEN** it is zero-padded to match the configured length

#### Scenario: Truncation to fixed duration
- **WHEN** an audio segment is longer than target duration
- **THEN** it is truncated to match the configured length

#### Scenario: Sample rate normalization
- **WHEN** audio is loaded with a non-standard sample rate
- **THEN** it is resampled to the target rate (e.g., 44.1kHz or 16kHz)

#### Scenario: Mel spectrogram computation
- **WHEN** CNN encoder requires spectrograms
- **THEN** waveforms are converted to mel spectrograms with configured parameters

#### Scenario: Audio augmentation
- **WHEN** augmentation is enabled in training mode
- **THEN** random time stretch, pitch shift, or noise injection is applied

### Requirement: MIDI Preprocessing Pipeline
The system SHALL preprocess MIDI event sequences to ensure consistent format and temporal alignment with audio.

#### Scenario: MIDI parsing from binary
- **WHEN** MIDI data is stored as binary in parquet or separate files
- **THEN** it is parsed into structured event sequences

#### Scenario: Event tokenization
- **WHEN** MIDI events are tokenized
- **THEN** each NOTE_ON, NOTE_OFF, and timing event is mapped to an integer token

#### Scenario: Temporal alignment with audio
- **WHEN** MIDI and audio correspond to the same segment
- **THEN** MIDI timestamps are aligned with audio using SMPTE or LRC timestamps

#### Scenario: Piano roll conversion
- **WHEN** piano roll encoder is used
- **THEN** MIDI events are converted to a pitch x time matrix with velocity values

#### Scenario: Polyphony handling
- **WHEN** multiple notes are played simultaneously
- **THEN** the encoder handles overlapping events without information loss

#### Scenario: MIDI augmentation
- **WHEN** augmentation is enabled in training mode
- **THEN** random transposition, velocity scaling, or time quantization is applied

### Requirement: Multimodal Dataset Interface
The system SHALL extend the dataset to load and return text, audio, and MIDI for each segment.

#### Scenario: Multimodal batch structure
- **WHEN** a batch is loaded from the dataset
- **THEN** it returns a dict with keys: text, audio, midi, label

#### Scenario: Lazy loading for efficiency
- **WHEN** audio and MIDI files are large
- **THEN** they are loaded on-demand rather than preloaded into memory

#### Scenario: Missing modality handling
- **WHEN** a segment lacks audio or MIDI data
- **THEN** zero tensors or None values are returned and handled gracefully

#### Scenario: Batch collation
- **WHEN** batching variable-length sequences
- **THEN** padding is applied to create uniform tensor shapes

### Requirement: Cross-Modal Attention Mechanism
The system SHALL implement attention layers that enable modalities to query and attend to each other.

#### Scenario: Text queries audio context
- **WHEN** text embedding queries audio embeddings
- **THEN** attention weights identify relevant audio regions for each text token

#### Scenario: Audio queries MIDI context
- **WHEN** audio embedding queries MIDI embeddings
- **THEN** attention weights capture harmonic-rhythmic correlations

#### Scenario: Multi-head attention
- **WHEN** cross-modal attention is configured with multiple heads
- **THEN** different attention heads learn different types of cross-modal relationships

#### Scenario: Attention dropout
- **WHEN** dropout is applied to attention weights
- **THEN** it prevents overfitting to specific cross-modal patterns

### Requirement: Modality Dropout for Robustness
The system SHALL randomly disable modalities during training to ensure the model doesn't over-rely on any single input.

#### Scenario: Random modality dropout
- **WHEN** modality dropout is enabled with probability p
- **THEN** each modality is randomly zeroed with probability p during training

#### Scenario: Inference with all modalities
- **WHEN** running inference
- **THEN** all available modalities are used (no dropout)

#### Scenario: Graceful degradation
- **WHEN** a modality is unavailable at inference
- **THEN** the model performs reasonably well with remaining modalities

### Requirement: Temporal Alignment Verification
The system SHALL verify that audio and MIDI are temporally aligned to prevent training on mismatched data.

#### Scenario: SMPTE timestamp alignment
- **WHEN** audio and MIDI segments are extracted using SMPTE timestamps
- **THEN** the start and end times match within a tolerance threshold

#### Scenario: Alignment quality check
- **WHEN** loading training data
- **THEN** misaligned segments are flagged and optionally excluded

#### Scenario: Manual alignment correction
- **WHEN** alignment is detected as incorrect
- **THEN** offset correction is applied based on metadata

### Requirement: Fusion Architecture Configuration
The system SHALL provide comprehensive configuration options for multimodal fusion.

#### Scenario: Audio encoder selection
- **WHEN** config.model.audio_encoder.type is set to "wav2vec2", "clap", or "cnn"
- **THEN** the corresponding encoder is instantiated

#### Scenario: MIDI encoder selection
- **WHEN** config.model.midi_encoder.type is set to "piano_roll", "event_based", or "music_transformer"
- **THEN** the corresponding encoder is instantiated

#### Scenario: Fusion strategy selection
- **WHEN** config.model.fusion.strategy is set to "early", "late", "cross_attention", or "gated"
- **THEN** the corresponding fusion module is instantiated

#### Scenario: Hidden dimension compatibility
- **WHEN** encoders have different output dimensions
- **THEN** projection layers align dimensions before fusion

### Requirement: Gradient Monitoring Across Modalities
The system SHALL monitor gradient norms for each modality to detect training imbalances.

#### Scenario: Per-modality gradient norms
- **WHEN** backward pass completes
- **THEN** gradient norms for text, audio, and MIDI encoders are logged separately

#### Scenario: Gradient imbalance detection
- **WHEN** one modality's gradients are significantly larger than others
- **THEN** a warning is logged suggesting learning rate adjustment

#### Scenario: Gradient clipping per-modality
- **WHEN** gradient clipping is enabled
- **THEN** it can be applied independently to each encoder

### Requirement: Storage Strategy for Multimodal Data
The system SHALL support flexible storage strategies for audio and MIDI to balance convenience and performance.

#### Scenario: Audio in parquet binary
- **WHEN** audio is stored as binary blobs in parquet
- **THEN** it is decoded on-the-fly during data loading

#### Scenario: Audio on disk referenced by path
- **WHEN** audio is stored as separate WAV files
- **THEN** file paths in parquet are used to load audio on-demand

#### Scenario: Precomputed audio features
- **WHEN** audio features (spectrograms, embeddings) are precomputed
- **THEN** they are loaded directly to speed up training

#### Scenario: MIDI in parquet binary
- **WHEN** MIDI is stored as binary blobs in parquet
- **THEN** it is parsed on-the-fly during data loading

#### Scenario: MIDI on disk referenced by path
- **WHEN** MIDI is stored as separate .mid files
- **THEN** file paths in parquet are used to load MIDI on-demand
