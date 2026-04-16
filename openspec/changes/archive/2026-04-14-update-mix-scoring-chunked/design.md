## Context

The Refractor model and CLAP encoder were trained on 10–30s corpus segments extracted
from the White catalog. `score_mix` was designed to score full song bounces against a
chromatic target, but passing a full-length waveform as a single CLAP input is
out-of-distribution — CLAP's internal windowing was not trained to aggregate that way.
Result: low confidence (~0.10) and predictions that land in the wrong color quadrant.

The 84 songs in `staged_raw_material/` each have a `<id>_main.wav` and a known color
from their `.yml` metadata — a free labeled validation set.

## Goals / Non-Goals

- **Goals**: accurate chromatic prediction on full-length mixes; no new model required
  if chunked aggregation reaches ≥70% top-1 accuracy on the 84-song validation set
- **Non-Goals**: real-time scoring; scoring stems individually; training a new end-to-end
  model in this change (deferred to Phase 2 if validation fails)

## Decisions

- **Decision**: 30s windows, 5s stride (configurable via CLI flags)
  - Matches the approximate segment length in the training corpus
  - 5s stride gives ~6× overlap on a typical song — smooths edge effects
  - Stride shorter than 5s would mean many near-duplicate chunks with little new info

- **Decision**: confidence-weighted mean for aggregation (not simple mean, not max)
  - Chunks where Refractor is uncertain (low confidence) contribute less
  - A simple mean would allow a noisy intro/outro to dilute a strong mid-section signal
  - Max-pooling would overweight a single anomalous chunk

- **Decision**: validate before training — use 84-song validation set first
  - If accuracy ≥ 70%: ship chunked scoring, no new model needed
  - If accuracy < 70%: propose Phase 2 calibration MLP (separate change) trained on
    aggregated full-song CLAP embeddings → temporal/spatial/ontological labels

- **Decision**: `chunk_audio` resamples to 48kHz before chunking
  - CLAP expects 48kHz; resampling once before chunking is more efficient than
    per-chunk resampling

## Risks / Trade-offs

- Long songs (>10 min) will be slow — 5s stride on a 10-min song ≈ 118 chunks
  → consider a `--max-chunks` cap (e.g. 60) that samples evenly across the song
- Some colors have few songs (e.g. White = 0 labeled, Black = sparse) — validation
  accuracy will be unreliable for those; report per-color N alongside accuracy

## Open Questions

- Should we also expose per-section scoring (verse/chorus/bridge separately) using
  the arrangement timecodes? Deferred — useful for drift analysis but out of scope here.
