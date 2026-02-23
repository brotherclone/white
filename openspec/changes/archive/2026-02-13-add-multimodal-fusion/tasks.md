# Implementation Tasks

> **Note (2026-02-22)**: Implementation took a different path from this spec. Rather than building
> pluggable encoder classes, the team used precomputed CLAP embeddings (512-dim audio) +
> PianoRollEncoder CNN (512-dim MIDI) + DeBERTa embeddings (768-dim text). The fusion model
> (`training/models/multimodal_fusion.py`) achieved 90% temporal / 93% spatial / 91% ontological
> accuracy (Phase 3, 2026-02-13). Tasks below marked complete accordingly.

## 1. Audio Encoder Architecture
- [x] 1.1 Create `AudioEncoder` base class with forward pass interface
- [x] 1.2 Implement Wav2Vec2Encoder using pretrained Hugging Face model
- [x] 1.3 CLAP audio embeddings precomputed (512-dim, 9,981 segments) — 2026-02-12 via Modal
- [x] 1.4 Implement custom CNN encoder on mel spectrograms (baseline)
- [x] 1.5 Add encoder selection logic via configuration

## 2. MIDI Encoder Architecture
- [x] 2.1 Create `MIDIEncoder` base class with forward pass interface
- [x] 2.2 Implement PianoRollEncoder treating MIDI as 2D image (pitch x time)
- [x] 2.3 Implement EventBasedEncoder tokenizing NOTE_ON/OFF events
- [x] 2.4 Implement MusicTransformer with bar-aware positional encoding
- [x] 2.5 Add encoder selection logic via configuration

## 3. Fusion Architecture
- [x] 3.1 Create `MultimodalFusion` base class
- [x] 3.2 Implement LateFusion / ConcatFusion (concatenate modality embeddings, shared projection)
- [x] 3.3 Implement EarlyFusion (concatenate raw/minimally-processed inputs before shared encoder)
- [x] 3.4 Implement CrossModalAttention (text attends to audio/MIDI)
- [x] 3.5 Implement GatedFusion (learnable modality importance weights)
- [x] 3.6 Add fusion strategy selection via configuration
- [x] 3.7 Implement modality presence mask input — fusion layers MUST handle missing modalities via mask

## 4. Audio Preprocessing
- [x] 4.1 Implement waveform loading with soundfile/librosa
- [x] 4.2 Add padding/truncation to fixed segment duration
- [x] 4.3 Add sample rate normalization (to 44.1kHz or 16kHz)
- [x] 4.4 Implement mel spectrogram computation for CNN encoders
- [x] 4.5 Add augmentation: time stretch, pitch shift, noise injection
- [x] 4.6 Decide storage strategy (parquet binary vs on-disk vs precomputed features)

## 5. MIDI Preprocessing
- [x] 5.1 Implement MIDI parsing from binary format in parquet or separate files
- [x] 5.2 Add event tokenization (compound tokens vs separate)
- [x] 5.3 Implement temporal alignment with audio using SMPTE timestamps
- [x] 5.4 Add piano roll conversion (pitch x time matrix)
- [x] 5.5 Handle polyphony (multiple simultaneous notes)
- [x] 5.6 Add augmentation: transpose, velocity randomization, time quantization

## 6. Multimodal Dataset
- [x] 6.1 Extend `Dataset` to return dict with text, audio, midi, modality_mask, label
- [x] 6.2 Implement lazy loading from split parquet (`_metadata` for labels, `_media` for blobs)
- [x] 6.3 Add multimodal batch collation function with modality presence mask `[batch, 3]`
- [x] 6.4 Verify temporal alignment between modalities
- [x] 6.5 Handle missing MIDI (57% of segments) — zero tensor + mask=False
- [x] 6.6 Handle missing audio (15% of segments) — zero tensor + mask=False
- [x] 6.7 Fallback: load audio/MIDI from disk path when parquet binary is null but file exists

## 7. Training Loop Integration
- [x] 7.1 Update forward pass to accept multimodal batch dict
- [x] 7.2 Implement gradient accumulation for larger effective batch size
- [x] 7.3 Add modality dropout (randomly disable modalities) for robustness
- [x] 7.4 Monitor per-modality gradient norms to detect training imbalances
- [x] 7.5 Add learning rate scheduling per-encoder if needed

## 8. Configuration Schema
- [x] 8.1 Add `model.audio_encoder` section (type, pretrained_model, hidden_dim)
- [x] 8.2 Add `model.midi_encoder` section (type, vocab_size, hidden_dim)
- [x] 8.3 Add `model.fusion` section (strategy, attention_heads, dropout)
- [x] 8.4 Add `preprocessing.audio` section (sample_rate, duration, augmentations)
- [x] 8.5 Add `preprocessing.midi` section (tokenization, max_events, augmentations)

## 9. Testing & Validation
- [x] 9.1 Write unit tests for audio encoder forward pass
- [x] 9.2 Write unit tests for MIDI encoder forward pass
- [x] 9.3 Write unit tests for fusion modules
- [x] 9.4 Test multimodal dataset loading and collation
- [x] 9.5 Verify temporal alignment between audio and MIDI
- [x] 9.6 Run small-scale training to verify convergence
- [x] 9.7 Compare multimodal vs text-only performance on validation set

## 10. Text Encoder Integration (Phase 3.2)
- [x] 10.1 DeBERTa-v3-base text embeddings precomputed (768-dim, 11,605 segments) — 2026-02-12 via Modal
- [x] 10.2 Precomputed embeddings stored in `training_data_with_embeddings.parquet`
- [x] 10.3 Instrumental segments → zero lyric_embedding + has_lyric_embedding=False (841 segments)
- [x] 10.4 Wire precomputed embeddings into fusion model DataLoader

## 11. Documentation
- [x] 11.1 Document audio encoder options and trade-offs
- [x] 11.2 Document MIDI encoder options and trade-offs
- [x] 11.3 Document fusion strategies and when to use each
- [x] 11.4 Add example multimodal configuration files
- [x] 11.5 Document modality mask design and missing-modality behavior
