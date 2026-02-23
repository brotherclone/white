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

### Requirement: Text Encoding
The system SHALL encode lyric text into semantic embeddings using the existing DeBERTa-v3 encoder from Phases 1-4. Future prosodic and structural encoding is deferred to `add-prosodic-lyric-encoding`.

#### Scenario: Semantic text embedding
- **WHEN** lyric text is available for a segment
- **THEN** the DeBERTa-v3 encoder produces a `[batch, 768]` text embedding

#### Scenario: Precomputed embedding loading
- **WHEN** precomputed text embeddings exist in the training parquet
- **THEN** they are loaded directly instead of running the encoder at training time

#### Scenario: Instrumental segment (no lyrics)
- **WHEN** a segment has no lyric text (`vocals=False` or `lyric_text` is empty)
- **THEN** a zero tensor is returned and `modality_mask[text]` is set to False

### Requirement: Multimodal Fusion Strategy
The system SHALL combine text, audio, and MIDI representations using configurable fusion strategies.

#### Scenario: Late fusion (concatenation)
- **WHEN** fusion strategy is "late"
- **THEN** modality-specific embeddings are independently encoded and concatenated before shared projection layers

#### Scenario: Early fusion
- **WHEN** fusion strategy is "early"
- **THEN** raw or minimally-processed inputs are concatenated before a shared encoder

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
The system SHALL extend the dataset to load and return text, audio, and MIDI for each segment, reading from the split parquet files (`training_segments_metadata.parquet` for metadata, `training_segments_media.parquet` for binary blobs).

#### Scenario: Multimodal batch structure
- **WHEN** a batch is loaded from the dataset
- **THEN** it returns a dict with keys: text, audio, midi, modality_mask, label

#### Scenario: Lazy loading for efficiency
- **WHEN** audio and MIDI files are large
- **THEN** they are loaded on-demand from the media parquet rather than preloaded into memory

#### Scenario: Missing MIDI handling (majority case)
- **WHEN** a segment has `has_midi=False` (57% of segments, up to 88% for Blue album)
- **THEN** a zero tensor is returned for the MIDI modality and `modality_mask[midi]` is set to False

#### Scenario: Missing audio handling
- **WHEN** a segment has `has_audio=False` (15% of segments)
- **THEN** a zero tensor is returned for the audio modality and `modality_mask[audio]` is set to False

#### Scenario: Modality presence mask
- **WHEN** a batch is constructed
- **THEN** a boolean mask tensor `[batch, 3]` indicates which of `[text, audio, midi]` are present per sample

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
- **WHEN** config.model.fusion.strategy is set to "late", "early", "cross_attention", or "gated"
- **THEN** the corresponding fusion module is instantiated (late fusion is the recommended default)

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
The system SHALL support the split-parquet storage layout where `training_segments_metadata.parquet` holds all non-binary columns (fast queries, 75 columns) and `training_segments_media.parquet` holds the same columns plus `audio_waveform` (binary), `audio_sample_rate` (int32), and `midi_binary` (binary, nullable).

#### Scenario: Metadata-only queries
- **WHEN** the dataset needs segment metadata (labels, flags, paths) without binary data
- **THEN** it reads from `training_segments_metadata.parquet` to avoid loading multi-GB binary columns

#### Scenario: Audio from parquet binary
- **WHEN** audio is needed for a segment and `audio_waveform` is non-null in the media parquet
- **THEN** the binary blob is decoded using the corresponding `audio_sample_rate`

#### Scenario: Audio from disk path fallback
- **WHEN** `audio_waveform` is null but `segment_audio_file` path exists on disk
- **THEN** the audio is loaded from the file path as a fallback

#### Scenario: Precomputed audio features
- **WHEN** audio features (spectrograms, embeddings) are precomputed and stored
- **THEN** they are loaded directly to speed up training

#### Scenario: MIDI from parquet binary
- **WHEN** MIDI is needed and `midi_binary` is non-null in the media parquet
- **THEN** it is parsed on-the-fly via `app/util/midi_segment_utils.py`

#### Scenario: MIDI from disk path fallback
- **WHEN** `midi_binary` is null but `midi_file` path exists on disk
- **THEN** the MIDI file is loaded from the path and parsed

#### Scenario: MIDI absent
- **WHEN** both `midi_binary` is null and `midi_file` path does not exist (57% of segments)
- **THEN** the segment is flagged as MIDI-absent and a zero tensor with mask is used
