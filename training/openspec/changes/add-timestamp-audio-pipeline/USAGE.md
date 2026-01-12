# Timestamp Audio Extraction Pipeline - Usage Guide

This guide demonstrates how to use the timestamp audio extraction pipeline to process Logic Pro exports into training-ready audio segments.

## Quick Start

### Process a Single Track

```python
from app.util.timestamp_pipeline import process_track_directory, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    max_segment_length_seconds=30.0,      # Maximum segment length
    structure_alignment_threshold_seconds=2.0,  # Snap to structure boundaries within 2s
    overlap_seconds=2.0,                   # Overlap for split segments
    extract_midi=True,                     # Extract MIDI files alongside audio
    output_metadata=True,                  # Generate JSON metadata
    sample_rate=None                       # Keep original sample rate (or set to 44100, etc.)
)

# Process a single track
result = process_track_directory(
    track_dir="staged_raw_material/01_01",
    output_dir="output/segments",
    config=config
)

print(f"Extracted {result['segments_extracted']} segments")
print(f"Total duration: {result['total_duration_seconds']:.1f}s")
print(f"Output: {result['output_directory']}")
```

### Process Multiple Tracks

```python
from app.util.timestamp_pipeline import process_staged_raw_material

# Process all tracks in staged_raw_material
result = process_staged_raw_material(
    staged_dir="staged_raw_material",
    output_dir="output/all_segments",
    config=config,
    track_filter="08_*"  # Optional: only process Violet album tracks
)

print(f"Processed {result['successful_tracks']} tracks")
print(f"Total segments: {result['total_segments_extracted']}")
```

## Pipeline Features

### 1. Structure-Aware Segmentation

The pipeline automatically aligns segment boundaries to song structure sections (verse, chorus, bridge) when they are within the threshold distance:

```python
# Example: Lyric at 00:28.5 near Verse 1 end at 00:30.0
# Will be adjusted to align with structure boundary

# Original: [00:28.500] - [00:32.000]
# Adjusted: [00:30.000] - [00:30.000] (aligned to Verse 1/Chorus 1 boundary)
```

### 2. Maximum Segment Length

Long segments are automatically split with overlap to maintain continuity:

```python
# Example: 60-second segment with 30s max and 2s overlap
# Split into:
# Segment 1: 0:00 - 0:30
# Segment 2: 0:28 - 0:58 (2s overlap with segment 1)
# Segment 3: 0:56 - 1:00 (2s overlap with segment 2)
```

### 3. MIDI Segmentation

MIDI files are automatically matched to audio tracks via the manifest and segmented to match:

```python
# MIDI files are time-shifted to start at 0:00 while preserving relative timing
# Matched via manifest audio_tracks with midi_file or midi_group_file fields
```

## Directory Structure

### Input (staged_raw_material)
```
staged_raw_material/
└── 01_01/
    ├── 01_01.yml              # Manifest with structure, metadata
    ├── 01_01.lrc              # LRC lyrics with timestamps
    ├── 01_01_main.wav         # Main audio file
    ├── 01_01_02.wav           # Individual track (vocals)
    ├── 01_01_synth_bass.mid   # MIDI file
    └── 01_01_drums.mid        # MIDI file (group)
```

### Output
```
output/segments/
└── 01_01/
    ├── 01_01_seg_0001.wav     # Audio segment 1
    ├── 01_01_seg_0002.wav     # Audio segment 2
    ├── 01_01_segments_metadata.json  # Metadata for all segments
    └── midi/
        ├── 01_01_seg_0001_synth_bass.mid
        ├── 01_01_seg_0001_drums.mid
        ├── 01_01_seg_0002_synth_bass.mid
        └── 01_01_seg_0002_drums.mid
```

## Metadata Format

The generated metadata file contains detailed information about each segment:

```json
[
  {
    "segment_id": "01_01_seg_0001",
    "track_id": "01_01",
    "audio_file": "output/segments/01_01/01_01_seg_0001.wav",
    "midi_files": [
      "output/segments/01_01/midi/01_01_seg_0001_synth_bass.mid",
      "output/segments/01_01/midi/01_01_seg_0001_drums.mid"
    ],
    "start_seconds": 15.0,
    "end_seconds": 17.107,
    "duration_seconds": 2.107,
    "lyric_text": "Hoisted high now",
    "segment_type": "combined",
    "structure_adjustments": [
      "Start aligned to Verse 1 boundary at 15.000s"
    ],
    "original_start": 15.0,
    "original_end": 17.107,
    "lrc_line_number": 5,
    "is_sub_segment": false,
    "sub_segment_info": null
  }
]
```

## Advanced Usage

### Custom Configuration

```python
config = PipelineConfig(
    max_segment_length_seconds=45.0,      # Longer segments for different model
    structure_alignment_threshold_seconds=3.0,  # More aggressive alignment
    overlap_seconds=3.0,                   # Larger overlap for context
    extract_midi=True,
    output_metadata=True,
    sample_rate=44100                      # Resample to 44.1kHz
)
```

### Process Specific Tracks

```python
from app.util.timestamp_pipeline import process_multiple_tracks

track_dirs = [
    "staged_raw_material/01_01",
    "staged_raw_material/01_02",
    "staged_raw_material/02_01"
]

result = process_multiple_tracks(
    track_dirs=track_dirs,
    output_dir="output/custom_batch",
    config=config
)
```

### Extract Segments Without Structure Alignment

```python
from app.util.timestamp_audio_extractor import create_segment_specs_from_lrc, extract_all_segments

# Create specs without manifest (no structure alignment)
specs = create_segment_specs_from_lrc(
    lrc_file_path="path/to/file.lrc",
    audio_file_path="path/to/audio.wav",
    manifest=None,  # No structure alignment
    max_segment_length=30.0
)

# Extract segments
paths = extract_all_segments(
    segment_specs=specs,
    output_dir="output/no_structure",
    filename_prefix="segment"
)
```

## Testing

Run the test suite to verify the pipeline:

```bash
# Run all timestamp pipeline tests
pytest tests/util/test_timestamp_pipeline.py -v

# Run specific test class
pytest tests/util/test_timestamp_pipeline.py::TestStructureBoundaryDetection -v

# Run integration tests (requires staged_raw_material)
pytest tests/util/test_timestamp_pipeline.py::TestRealDataProcessing -v
```

## Requirements

- Python 3.13+
- soundfile
- numpy
- pyyaml
- pydantic
- mido (optional, for MIDI segmentation)

## Troubleshooting

### "No entries found in LRC file"
- Check LRC file format: timestamps should be `[MM:SS.mmm]`
- Ensure LRC file is UTF-8 encoded
- Verify timestamps are followed by lyric text

### "Main audio file not found"
- Check manifest `main_audio_file` field matches actual filename
- Ensure audio file is in the same directory as manifest

### "MIDI segmentation skipped"
- Install mido: `pip install mido`
- Check that MIDI files are referenced in manifest `audio_tracks`

### Segments not aligned to structure
- Verify manifest has `structure` field populated
- Check `structure_alignment_threshold_seconds` is large enough
- Ensure structure timestamps overlap with lyric timestamps
