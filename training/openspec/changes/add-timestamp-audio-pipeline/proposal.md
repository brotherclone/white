# Change: Add Timestamp Audio Extraction Pipeline

## Why
The White Album project requires a systematic approach to preparing training data from multi-track audio exports. Currently, the workflow involves manual conversion of SMPTE timestamps to LRC format, extraction of audio segments based on timestamps, and preparation of non-silent segments for model training. This proposal documents the integrated pipeline that transforms raw Logic Pro exports into training-ready audio segments.

## What Changes
- Document SMPTE to LRC timestamp conversion capability
- Document LRC-based audio segment extraction functionality
- Document non-silent audio segment extraction for training data
- Specify the integrated workflow that chains these capabilities together
- Define the expected file structure and data formats for staged raw material
- Implement special logic for handling time segment overlaps between 'structure' (in manifests) and lyrics (in LRC files)
  - For example when a lyric is within x range of a structure boundary, extend the segment to include the full structure section
  - This is to ensure that the model is trained on a full segment, not just a single word
- Implement special considerations for splitting MIDI files into audio segments using best practices
- Implement maximum segment length for audio segments using best practices

## Impact
- Affected specs: `timestamp-audio-pipeline` (new capability)
- Affected code:
  - `app/util/convert_smpte_to_lrc.py` - SMPTE timestamp conversion
  - `app/util/audio_io.py` - Audio loading utilities
  - `app/agents/tools/audio_tools.py` - Segment extraction and processing
  - `staged_raw_material/` - Training data organization - LRC is used in both manifests (01_01.yml and lyrics 01_01.lrc)
