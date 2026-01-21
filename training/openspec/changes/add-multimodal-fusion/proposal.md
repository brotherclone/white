# Change: Add Multimodal Fusion for Text, Audio, and MIDI

## Why
Text-only models capture conceptual rebracketing but miss perceptual and musical dimensions. Rebracketing manifests across modalities - in vocal delivery, harmonic choices, and rhythmic patterns. Multimodal fusion enables the model to learn richer representations by combining text embeddings with audio waveforms and MIDI event sequences.

## What Changes
- Add `AudioEncoder` module supporting Wav2Vec2, CLAP, or custom CNN/Transformer architectures
- Add `MIDIEncoder` module supporting piano roll CNN, event-based transformer, or Music Transformer patterns
- Implement `MultimodalFusion` architecture with late fusion and cross-modal attention
- Extend dataset to load and preprocess audio waveforms and MIDI event sequences
- Add audio preprocessing: padding/truncation, sample rate normalization, augmentation support
- Add MIDI preprocessing: event tokenization, temporal alignment with audio via SMPTE
- Implement cross-modal attention mechanism for text-audio-MIDI interaction
- Add configuration for fusion strategy (early, late, gated, cross-attention)

## Impact
- **BREAKING**: Dataset format expands to include audio and MIDI alongside text
- Affected specs: multimodal-fusion (new capability)
- Affected code:
  - `training/models/audio_encoder.py` (new)
  - `training/models/midi_encoder.py` (new)
  - `training/models/fusion.py` (new)
  - `training/core/pipeline.py` - multimodal dataset implementation
  - `training/core/trainer.py` - batch handling for multiple modalities
  - `training/preprocessing/` - audio and MIDI preprocessing utilities
- Dependencies: soundfile, librosa, mido, torch-audiomentations
- Training time: Significantly increased due to audio/MIDI processing
- GPU memory: Increased, may require gradient accumulation or smaller batches
