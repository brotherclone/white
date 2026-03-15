# Change: Spike — MusicGen Chromatic Text Prompt Evaluation

## Why
The generative audio spike established that CLAP-conditioned synthesis is not viable
and granular retrieval is the current best path. However, MusicGen (Meta AudioCraft)
remains unevaluated for chromatic targeting via text prompts. The question is whether
chromatic mode descriptions (e.g. "melancholic, past-tense, sparse piano, ambient")
yield MusicGen outputs that score above a useful chromatic_match threshold when
evaluated by Refractor.

If ≥50% of MusicGen outputs achieve chromatic_match > 0.4, text-conditioned synthesis
becomes a viable complement to granular retrieval — capable of generating novel audio
rather than recombining corpus recordings.

This is a time-boxed Modal spike: 3 clips × 8 colors = 24 clips, scored by Refractor.

## What Changes
- New Modal script `training/modal_musicgen_spike.py`
- New `training/spikes/musicgen-prompts/spike_report.md` with results + go/no-go
- No production code changes

## Impact
- Affected specs: `generative-audio-synthesis` (MODIFIED)
- Affected code: Modal spike script only
- Cost: ~$2–5 in Modal GPU credits (A10G, 24 clips × ~30s each)
- License note: MusicGen-Medium is CC-BY-NC — outputs cannot be used commercially
  without checking Earthly Frames' specific use case
