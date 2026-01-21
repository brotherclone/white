# Change: Add Generative Models for Segment Synthesis

## Why
Beyond understanding rebracketing, the White Album requires generating new segments that authentically manifest chromatic and rebracketing characteristics. Generative models (VAE, Diffusion, GPT-style) enable sampling new segments, completing partial segments, and exploring the latent space of ontological transformations.

## What Changes
- Add `RebracketingVAE` (Variational Autoencoder) conditioned on chromatic mode
- Add `RebracketingDiffusion` model for high-quality segment generation
- Add `SegmentGenerator` (GPT-style transformer decoder) for autoregressive generation
- Implement latent space manipulation and sampling strategies
- Add tokenization strategies for audio (neural codecs) and MIDI (music tokens)
- Implement generation evaluation: quality, diversity, chromatic consistency
- Add configuration for generative model selection and sampling parameters

## Impact
- Affected specs: generative-models (new capability)
- Affected code:
  - `training/models/vae.py` (new)
  - `training/models/diffusion.py` (new)
  - `training/models/autoregressive.py` (new)
  - `training/generation/` - sampling and generation utilities
  - `training/evaluation/` - generation quality metrics
- Dependencies: diffusers library, encodec or similar for audio codecs
- Training complexity: Very high - diffusion models are compute-intensive
- Storage: Generated samples require significant storage
