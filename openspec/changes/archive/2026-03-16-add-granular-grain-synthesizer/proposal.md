# Change: Add Granular Grain Synthesizer

## Why
The generative audio spike (`2026-03-15-spike-generative-audio-synthesis`) found that
color clusters are NOT separable in CLAP space (silhouette -0.30) and no existing model
accepts pre-computed CLAP embeddings as conditioning input. CLAP-space synthesis is
not viable.

However, all corpus source audio is available locally (100% of top-20 segments per color
have accessible `source_audio_file` paths with `start_seconds`/`end_seconds` timestamps).
Granular synthesis from Refractor-scored segments is the highest-value, lowest-cost path
to generating chromatic audio textures from the corpus.

The grain synthesizer:
1. Retrieves the top-N segments for a target color using `retrieve_by_color()` (Refractor scoring)
2. Loads 1-second grains from source audio via `soundfile` at random offsets within each segment
3. Crossfades grains using a Hann window into a continuous texture of configurable length
4. Writes the output WAV and a `grain_map.yml` logging which grains were used

The output is a chromatic collage from the corpus — not new synthesis, but a targeted
texture that carries the chromatic signature of the selected segments.

## What Changes
- New `training/tools/grain_synthesizer.py` — standalone CLI tool
- New `tests/tools/test_grain_synthesizer.py`
- No production pipeline changes; this is a standalone tool for sound design use

## Impact
- Affected specs: `generative-audio-synthesis` (MODIFIED)
- Affected code: new file only
- Dependencies: `soundfile`, `numpy` (already in requirements); optionally `scipy` for
  Hann window (can use numpy fallback)
