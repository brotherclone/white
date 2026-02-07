# Change: Add Training Data Verification Tools

## Why

The training parquet files contain embedded audio waveforms and MIDI binary data, but there's no way to spot-check whether the data is correct without writing custom extraction code each time. After major pipeline changes (MIDI bug fix, structure fallback), we need to verify that segments sound right before committing to a 6-hour extraction + training run. The parquet files are too large to inspect manually.

## What Changes

- New verification script that extracts random or targeted segments from parquet and writes them as playable WAV/MIDI files
- Coverage report showing modality presence by album color
- Optional A/B comparison: play original audio file segment vs parquet-stored audio to verify fidelity

## Impact

- Affected files: new script in `training/` or `app/util/`
- Reads from: `training_segments_metadata.parquet`, `training_segments_media.parquet`
- No changes to existing extraction or training code
