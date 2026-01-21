# Change: Add Chromatic Style Transfer for Cross-Mode Generation

## Why
The Rainbow Table spans multiple chromatic modes (BLACK, RED, ORANGE, etc.), each with distinct ontological characteristics. Style transfer enables generating content in different chromatic modes while preserving conceptual content - a key capability for understanding and synthesizing White Album material across ontological dimensions.

## What Changes
- Add `ChromaticStyleEncoder` to extract chromatic essence from segments
- Add `DisentangledEncoder` for content-style separation
- Add `ChromaticDecoder` for generating segments with target chromatic styles
- Implement style reconstruction, transfer, and preservation losses
- Add adversarial training to distinguish real from transferred segments
- Extend dataset to include chromatic mode labels (BLACK, RED, ORANGE, etc.)
- Add style transfer evaluation metrics: style consistency, content preservation

## Impact
- Affected specs: chromatic-style-transfer (new capability)
- Affected code:
  - `training/models/style_encoder.py` (new)
  - `training/models/disentangled_encoder.py` (new)
  - `training/models/chromatic_decoder.py` (new)
  - `training/core/trainer.py` - multi-loss training for style transfer
  - `training/evaluation/` - style transfer metrics
- Dependencies: May use pretrained discriminator for adversarial loss
- Training complexity: High - requires careful balancing of multiple loss terms
