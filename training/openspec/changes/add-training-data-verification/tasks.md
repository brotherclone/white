# Implementation Tasks

## 1. Segment Extraction Tool
- [ ] 1.1 Write script to extract N random segments from media parquet as playable WAV files
- [ ] 1.2 Add flag to target specific album colors (e.g., `--color Green` to verify the new instrumental segments)
- [ ] 1.3 Add flag to target specific songs (e.g., `--song 05_01`)
- [ ] 1.4 Extract corresponding MIDI segments as .mid files (where midi_binary is non-null)
- [ ] 1.5 Write extracted files to a verification output directory with descriptive names
- [ ] 1.6 Include segment metadata in filenames or sidecar JSON (song, section, timestamps)

## 2. Coverage Report
- [ ] 2.1 Generate modality coverage table by album color (audio %, MIDI %, text %)
- [ ] 2.2 Flag anomalies: songs with 0 segments, segments with unexpected missing modalities
- [ ] 2.3 Compare segment counts against expected counts from manifest (tracks * structure sections)
- [ ] 2.4 Report new vs old segment counts (diff against previous extraction)

## 3. Fidelity Verification
- [ ] 3.1 For a sample of segments, compare parquet audio blob vs original WAV file at same timestamp
- [ ] 3.2 Report any sample rate mismatches or duration discrepancies
- [ ] 3.3 For MIDI segments, verify note events exist and fall within expected timestamp range
- [ ] 3.4 Verify piano roll conversion produces non-zero matrices for MIDI-present segments

## 4. Quick Smoke Test
- [ ] 4.1 Create a single `verify_extraction.py` entry point that runs all checks
- [ ] 4.2 Exit with clear pass/fail summary and list of any issues found
- [ ] 4.3 Generate an HTML report with embedded audio players for easy review (optional)
