# Change: Chunked full-song mix scoring with validation suite

## Why

`score_mix` currently passes the entire waveform to CLAP as a single input. CLAP was
trained on 10–30s corpus segments; a full-length song is out-of-distribution, producing
low confidence scores (~0.10) and predictions that don't reflect the song's actual
chromatic character. The fix is to chunk the mix into overlapping ~30s windows, score
each chunk individually with Refractor (the model already knows what 30s of each color
sounds like), and aggregate results confidence-weighted — exactly mirroring how the
training data was built.

We have 84 labeled songs with `staged_raw_material/<id>/<id>_main.wav` files and known
colors. This gives us a free validation set to confirm the chunked approach works before
deciding whether a new calibration model is needed.

## What Changes

- **`encode_audio_file`** in `score_mix.py`: replace single-pass CLAP encoding with
  overlapping-window chunking (30s windows, 5s stride), returning one CLAP embedding
  per chunk
- **`score_mix`**: score each chunk via Refractor individually, then aggregate
  temporal/spatial/ontological distributions using confidence-weighted mean pooling
- **New: `validate_mix_scoring.py`** in `training/`: iterate all 84 labeled
  `staged_raw_material` songs, score each `main.wav`, report top-1 color accuracy and
  per-color breakdown — determines whether a calibration head is needed
- `mix_score.yml` gains `chunk_count` and `chunk_stride_s` metadata fields
- The `padding=True` fix to `refractor._encode_audio` is retained (needed for chunk
  batches)

## Impact

- Affected specs: `audio-mix-scoring`
- Affected code: `app/generators/midi/production/score_mix.py`,
  `training/refractor.py` (padding fix already in place), new
  `training/validate_mix_scoring.py`
- Non-breaking: `mix_score.yml` schema gains fields but existing consumers only read
  temporal/spatial/ontological/confidence
- If validation accuracy is low (<70% top-1), a Phase 2 calibration head trained on
  the 84-song full-mix dataset should be proposed as a separate change
