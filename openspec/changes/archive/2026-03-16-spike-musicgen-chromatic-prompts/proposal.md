# Change: Spike — Stable Audio Open Chromatic Text Prompt Evaluation

## Why
The generative audio spike established that CLAP-conditioned synthesis is not viable
and granular retrieval is the current best path. However, text-conditioned synthesis
remains unevaluated for chromatic targeting. The question is whether chromatic mode
descriptions (e.g. "melancholic, past-tense, sparse piano, ambient") yield outputs
that score above a useful chromatic_match threshold when evaluated by Refractor.

MusicGen was the original candidate but was replaced by **Stable Audio Open**
(`stabilityai/stable-audio-open-1.0`) for two reasons:
1. MusicGen is CC-BY-NC (non-commercial only). Stable Audio Open's community license
   allows commercial use for organisations with annual revenue < $1M; paid license
   available above that threshold — a practical commercial path vs. a hard block.
2. Stable Audio Open produces stereo 44.1kHz audio up to 47s — better quality floor
   for the chromatic texture use case.

If ≥50% of outputs achieve chromatic_match > 0.4, text-conditioned synthesis becomes
a viable complement to granular retrieval — capable of generating novel audio that
has never existed in the corpus, vs. granular which recombines existing recordings.

This is a time-boxed Modal spike: 3 clips × 8 colors = 24 clips, scored by Refractor.

## What Changes
- New Modal script `training/modal_stable_audio_spike.py`
- New analysis script `training/spikes/stable-audio-prompts/analyze_results.py`
- New `training/spikes/stable-audio-prompts/spike_report.md` (written after run)
- No production code changes

## Impact
- Affected specs: `generative-audio-synthesis` (MODIFIED)
- Affected code: Modal spike script + analysis script only
- Cost: ~$2–5 in Modal GPU credits (A10G, 24 clips × ~30s each)
- License: Stability AI Community License — commercial use OK under $1M revenue threshold
