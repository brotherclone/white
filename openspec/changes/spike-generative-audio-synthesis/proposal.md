# Change: Spike — Generative Audio Synthesis from Chromatic Embedding Space

## Why
The training corpus covers 8 colors in CLAP embedding space. If those clusters are
separable and navigable, it may be possible to *synthesize* audio that targets a specific
chromatic position — either by interpolating between real segments and decoding back to
audio, or by conditioning a generative model on a target CLAP embedding. This spike
investigates feasibility before committing to implementation.

This is explicitly not a production feature — it is a time-boxed research effort to answer
three questions:
1. Are the 8 color clusters separable in CLAP space?
2. Does embedding interpolation between real segments produce perceptually coherent audio?
3. Which decoder approach (retrieval, AudioCraft, Stable Audio, CLAP inversion) is most
   viable for this corpus given cost/quality/latency constraints?

## What Changes
- Spike notebook + report under `training/spikes/generative-audio/`
- No production code changes; no new specs that would constrain implementation
- Findings will inform a follow-on `add-generative-audio-synthesis` proposal (or a
  decision to not pursue this direction)

## Impact
- Affected specs: `generative-audio-synthesis` (new, spike-only — requirements describe
  deliverables, not shipped behavior)
- Affected code: none (spike outputs live in `training/spikes/`)
- External: may require API access to AudioCraft / Stable Audio for evaluation
