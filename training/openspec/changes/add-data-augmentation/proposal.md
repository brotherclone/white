# Change: Add Data Augmentation and Synthetic Generation

## Why
Training data quality and quantity directly impact model performance. Augmentation increases effective dataset size while preserving labels, and synthetic generation via White Agent creates additional labeled training data for underrepresented classes.

## What Changes
- Add `AudioAugmenter` with time stretch, pitch shift, noise injection, reverb/EQ
- Add `TextAugmenter` with back-translation, synonym replacement, paraphrase
- Add `MIDIAugmenter` with transposition, velocity randomization, quantization
- Add `SyntheticGenerator` using White Agent to generate labeled segments
- Implement augmentation pipelines in training loop
- Add configuration for augmentation probability and parameters
- Ensure augmentations preserve rebracketing characteristics

## Impact
- Affected specs: data-augmentation (new capability)
- Affected code:
  - `training/augmentation/` (new directory)
  - `training/core/pipeline.py` - augmentation integration
  - `training/synthetic/` - White Agent integration for synthesis
- Dependencies: torch-audiomentations, nlpaug, mido
