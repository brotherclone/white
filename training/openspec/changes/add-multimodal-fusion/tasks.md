# Implementation Tasks

## 1. Audio Encoder Architecture
- [ ] 1.1 Create `AudioEncoder` base class with forward pass interface
- [ ] 1.2 Implement Wav2Vec2Encoder using pretrained Hugging Face model
- [ ] 1.3 Implement CLAPEncoder for contrastive language-audio pretraining
- [ ] 1.4 Implement custom CNN encoder on mel spectrograms (baseline)
- [ ] 1.5 Add encoder selection logic via configuration

## 2. MIDI Encoder Architecture
- [ ] 2.1 Create `MIDIEncoder` base class with forward pass interface
- [ ] 2.2 Implement PianoRollEncoder treating MIDI as 2D image (pitch x time)
- [ ] 2.3 Implement EventBasedEncoder tokenizing NOTE_ON/OFF events
- [ ] 2.4 Implement MusicTransformer with bar-aware positional encoding
- [ ] 2.5 Add encoder selection logic via configuration

## 3. Fusion Architecture
- [ ] 3.1 Create `MultimodalFusion` base class
- [ ] 3.2 Implement EarlyFusion (concatenate before encoding)
- [ ] 3.3 Implement LateFusion (concatenate after independent encoding)
- [ ] 3.4 Implement CrossModalAttention (text attends to audio/MIDI)
- [ ] 3.5 Implement GatedFusion (learnable modality importance weights)
- [ ] 3.6 Add fusion strategy selection via configuration

## 4. Audio Preprocessing
- [ ] 4.1 Implement waveform loading with soundfile/librosa
- [ ] 4.2 Add padding/truncation to fixed segment duration
- [ ] 4.3 Add sample rate normalization (to 44.1kHz or 16kHz)
- [ ] 4.4 Implement mel spectrogram computation for CNN encoders
- [ ] 4.5 Add augmentation: time stretch, pitch shift, noise injection
- [ ] 4.6 Decide storage strategy (parquet binary vs on-disk vs precomputed features)

## 5. MIDI Preprocessing
- [ ] 5.1 Implement MIDI parsing from binary format in parquet or separate files
- [ ] 5.2 Add event tokenization (compound tokens vs separate)
- [ ] 5.3 Implement temporal alignment with audio using SMPTE timestamps
- [ ] 5.4 Add piano roll conversion (pitch x time matrix)
- [ ] 5.5 Handle polyphony (multiple simultaneous notes)
- [ ] 5.6 Add augmentation: transpose, velocity randomization, time quantization

## 6. Multimodal Dataset
- [ ] 6.1 Extend `Dataset` to return dict with text, audio, MIDI, label
- [ ] 6.2 Implement lazy loading for audio/MIDI (memory efficiency)
- [ ] 6.3 Add multimodal batch collation function
- [ ] 6.4 Verify temporal alignment between modalities
- [ ] 6.5 Handle missing modalities (text-only, audio-only fallback)

## 7. Training Loop Integration
- [ ] 7.1 Update forward pass to accept multimodal batch dict
- [ ] 7.2 Implement gradient accumulation for larger effective batch size
- [ ] 7.3 Add modality dropout (randomly disable modalities) for robustness
- [ ] 7.4 Monitor per-modality gradient norms to detect training imbalances
- [ ] 7.5 Add learning rate scheduling per-encoder if needed

## 8. Configuration Schema
- [ ] 8.1 Add `model.audio_encoder` section (type, pretrained_model, hidden_dim)
- [ ] 8.2 Add `model.midi_encoder` section (type, vocab_size, hidden_dim)
- [ ] 8.3 Add `model.fusion` section (strategy, attention_heads, dropout)
- [ ] 8.4 Add `preprocessing.audio` section (sample_rate, duration, augmentations)
- [ ] 8.5 Add `preprocessing.midi` section (tokenization, max_events, augmentations)

## 9. Testing & Validation
- [ ] 9.1 Write unit tests for audio encoder forward pass
- [ ] 9.2 Write unit tests for MIDI encoder forward pass
- [ ] 9.3 Write unit tests for fusion modules
- [ ] 9.4 Test multimodal dataset loading and collation
- [ ] 9.5 Verify temporal alignment between audio and MIDI
- [ ] 9.6 Run small-scale training to verify convergence
- [ ] 9.7 Compare multimodal vs text-only performance on validation set

## 10. Documentation
- [ ] 10.1 Document audio encoder options and trade-offs
- [ ] 10.2 Document MIDI encoder options and trade-offs
- [ ] 10.3 Document fusion strategies and when to use each
- [ ] 10.4 Add example multimodal configuration files
- [ ] 10.5 Document preprocessing pipeline and alignment verification
