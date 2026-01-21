# Implementation Tasks

## 1. Documentation
- [x] 1.1 Document SMPTE to LRC conversion specification
- [x] 1.2 Document audio I/O utilities specification
- [x] 1.3 Document segment extraction specification
- [x] 1.4 Document integrated pipeline workflow
- [x] 1.5 Document staged raw material structure

## 2. Core Implementation
- [x] 2.1 Enhance LRC parsing utilities in `app/util/lrc_utils.py`
- [x] 2.2 Create timestamp audio extractor in `app/util/timestamp_audio_extractor.py`
- [x] 2.3 Implement structure-aware segment boundary adjustment
- [x] 2.4 Implement maximum segment length constraints with intelligent splitting
- [x] 2.5 Create MIDI segmentation utilities in `app/util/midi_segment_utils.py`
- [x] 2.6 Implement MIDI-to-audio alignment and time-shifting
- [x] 2.7 Create integrated pipeline in `app/util/timestamp_pipeline.py`

## 3. Testing
- [x] 3.1 Create test suite in `tests/util/test_timestamp_pipeline.py`
- [x] 3.2 Test duration conversion utilities
- [x] 3.3 Test structure boundary detection and alignment
- [x] 3.4 Test long segment splitting with overlap
- [x] 3.5 Test integration with real staged raw material
- [x] 3.6 All tests passing (13/13 passed)

## 4. Validation
- [x] 4.1 Validate proposal with openspec tooling
- [x] 4.2 Review spec completeness
- [x] 4.3 Ensure all scenarios are covered

## Implementation Summary

### New Files Created
1. **app/util/timestamp_audio_extractor.py** - Core audio segmentation with structure awareness
   - `AudioSegmentSpec` dataclass for segment specifications
   - `duration_to_seconds()` - Convert Duration to seconds
   - `find_nearest_structure_boundary()` - Detect nearby structure boundaries
   - `adjust_segment_to_structure()` - Align segments to structure sections
   - `split_long_segment()` - Split oversized segments with overlap
   - `create_segment_specs_from_lrc()` - Generate specs from LRC + manifest
   - `extract_audio_segment()` - Extract and save audio segment
   - `extract_all_segments()` - Batch extraction

2. **app/util/midi_segment_utils.py** - MIDI file segmentation support
   - `segment_midi_file()` - Time-shift and extract MIDI segments
   - `find_matching_midi_files()` - Match MIDI files to audio via manifest
   - `segment_midi_with_audio()` - Segment MIDI alongside audio
   - `extract_midi_note_density()` - Calculate notes per second (analysis)

3. **app/util/timestamp_pipeline.py** - Integrated end-to-end pipeline
   - `PipelineConfig` dataclass for configuration
   - `SegmentMetadata` dataclass for extracted segment metadata
   - `process_track_directory()` - Process single track with complete pipeline
   - `process_multiple_tracks()` - Batch processing
   - `process_staged_raw_material()` - Process entire staged_raw_material directory

4. **tests/util/test_timestamp_pipeline.py** - Comprehensive test suite
   - Duration conversion tests
   - Structure boundary detection tests
   - Segment splitting tests
   - Integration tests with real data

### Features Implemented
- ✅ LRC-based audio segmentation
- ✅ Structure-aware boundary alignment (2s threshold configurable)
- ✅ Maximum segment length enforcement (30s default, configurable)
- ✅ Intelligent segment splitting with overlap (2s default)
- ✅ MIDI file segmentation with time-shifting
- ✅ Automatic MIDI-to-audio file matching via manifest
- ✅ Comprehensive metadata generation (JSON output)
- ✅ Batch processing support for multiple tracks
- ✅ Full integration with existing manifest and audio utilities
