# Change: Full-mix chromatic calibration head (Refractor CDM — Maxisingle Compact Disc)

## Why

The `update-mix-scoring-chunked` change confirmed that chunking full-song mixes into 30s
windows and aggregating with confidence-weighted mean pooling does not fix the low-confidence
problem (~0.10). Produced, mixed songs occupy a fundamentally different acoustic space from
the isolated 10–30s catalog segments that Refractor was trained on. Even when each 30s chunk
is individually scored, the CLAP embeddings of a produced mix land far from the training
distribution and Refractor cannot discriminate between colors.

The fix is a small calibration MLP (the "Refractor CDM") trained directly on aggregated
CLAP embeddings from the 78 labeled `_main.wav` files we already have. Because we train
on the same mix audio we want to score, the model learns what each color actually sounds
like in a finished production context.

## What Changes

- **New `training/models/refractor_cdm_model.py`**: small MLP architecture —
  `512 (+ optional 768 concept) → 256 → 128 → 3 × softmax(3)` — three independent heads
  for temporal/spatial/ontological regression against `CHROMATIC_TARGETS` soft targets
- **New `training/train_refractor_cdm.py`**: training script that loads all
  `staged_raw_material/<id>/<id>_main.wav` files, extracts per-chunk CLAP embeddings,
  and trains the calibration head with random-chunk augmentation; exports ONNX to
  `training/data/refractor_cdm.onnx`
- **Updated `training/refractor.py`**: `Refractor` gains an optional
  `cdm_onnx_path` parameter; when provided, `score()` routes audio-only calls through
  the Refractor CDM head instead of the base ONNX model
- **Updated `app/generators/midi/production/score_mix.py`**: CLI gains
  `--cdm-onnx-path` flag; defaults to `training/data/refractor_cdm.onnx` if the file
  exists, otherwise falls back to the base model
- **Updated `training/validate_mix_scoring.py`**: auto-uses `refractor_cdm.onnx` when
  present, with `--no-cdm` flag to force base model

## Impact

- Affected specs: `audio-mix-scoring`
- Affected code: `training/refractor.py`, `app/generators/midi/production/score_mix.py`,
  `training/validate_mix_scoring.py`, new `training/train_refractor_cdm.py`,
  new `training/models/refractor_cdm_model.py`
- Non-breaking: base Refractor behavior unchanged when `cdm_onnx_path` is not provided;
  `score_mix` falls back gracefully if `refractor_cdm.onnx` does not exist
- Training requires ~5 min on CPU (78 songs × ~43 chunks, tiny MLP)
