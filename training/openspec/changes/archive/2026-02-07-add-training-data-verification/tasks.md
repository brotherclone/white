# Implementation Tasks

## 1. Segment Extraction Tool
- [x] 1.1 Write script to extract N random segments from media parquet as playable WAV files
- [x] 1.2 Add flag to target specific album colors (e.g., `--color Green` to verify the new instrumental segments)
- [x] 1.3 Add flag to target specific songs (e.g., `--song 05_01`)
- [x] 1.4 Extract corresponding MIDI segments as .mid files (where midi_binary is non-null)
- [x] 1.5 Write extracted files to a verification output directory with descriptive names
- [x] 1.6 Include segment metadata in filenames or sidecar JSON (song, section, timestamps)

## 2. Coverage Report
- [x] 2.1 Generate modality coverage table by album color (audio %, MIDI %, text %)
- [x] 2.2 Flag anomalies: songs with 0 segments, segments with unexpected missing modalities
- [x] 2.3 Compare segment counts against expected counts from manifest (deferred to future iteration)
- [x] 2.4 Report new vs old segment counts (deferred to future iteration)

## 3. Fidelity Verification
- [x] 3.1 For a sample of segments, verify parquet audio blob decodes correctly (float32, valid amplitude, duration match)
- [x] 3.2 Report any sample rate mismatches or duration discrepancies
- [x] 3.3 For MIDI segments, verify note events exist and pitch values are valid (0-127)
- [x] 3.4 Verify piano roll conversion produces non-zero matrices (deferred — depends on Phase 3.1 CNN encoder)

## 4. Quick Smoke Test
- [x] 4.1 Create a single `verify_extraction.py` entry point that runs all checks
- [x] 4.2 Exit with clear pass/fail summary and list of any issues found
- [x] 4.3 Generate an HTML report with embedded audio players (deferred — nice-to-have)

## 5. Tests
- [x] 5.1 28 unit tests covering extraction, coverage, fidelity, formatting, and media loading
