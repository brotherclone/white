# Change: Add Data Augmentation for Multimodal Training

## Why
Training data quality and quantity directly impact the multimodal fitness function's accuracy. Audio and MIDI augmentation increases effective dataset size while preserving chromatic labels — critical for underrepresented albums (Yellow/Green have fewer segments) and for improving the model's robustness when scoring evolutionary music candidates that may sound different from the training data.

## What Changes
- Add `AudioAugmenter` with time stretch, pitch shift, noise injection, reverb/EQ
- Add `MIDIAugmenter` with transposition, velocity randomization, quantization variation
- Add `TextAugmenter` with back-translation, synonym replacement, paraphrase (for lyric-bearing segments)
- Implement augmentation pipelines in training loop
- Add configuration for augmentation probability and parameters
- Ensure augmentations preserve chromatic mode characteristics

## Impact
- Affected specs: data-augmentation (new capability)
- Affected code:
  - `training/augmentation/` (new directory)
  - `training/core/pipeline.py` - augmentation integration
- Dependencies: torch-audiomentations, mido, nlpaug
- Priority: Low — revisit if multimodal training data becomes a bottleneck
