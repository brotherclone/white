## ADDED Requirements

### Requirement: SMPTE to LRC Timestamp Conversion
The system SHALL convert SMPTE-format timestamps (HH:MM:SS:FF) to LRC-format timestamps ([MM:SS.mmm]) for use in lyric timing and audio segmentation.

#### Scenario: Convert single SMPTE timestamp
- **WHEN** a SMPTE timestamp string "01:01:06:11.49" is provided
- **THEN** it is converted to LRC format "[01:06.011]"

#### Scenario: Batch convert timestamps in text file
- **WHEN** a text file containing multiple SMPTE timestamps is processed
- **THEN** all SMPTE timestamps are replaced with LRC format in-place

#### Scenario: Handle malformed timestamp gracefully
- **WHEN** a timestamp with insufficient colon-separated fields is encountered
- **THEN** the original timestamp is preserved without conversion

### Requirement: Audio File Loading
The system SHALL load audio files using soundfile (libsndfile) with optional resampling and mono conversion.

#### Scenario: Load audio file with resampling
- **WHEN** an audio file is loaded with a target sample rate different from the file's native rate
- **THEN** the audio data is resampled to the target rate using high-quality resampling

#### Scenario: Convert multi-channel to mono
- **WHEN** an audio file with multiple channels is loaded with mono=True
- **THEN** the channels are averaged to produce a single mono channel

#### Scenario: Return normalized float32 array
- **WHEN** any audio file is loaded
- **THEN** the returned data is a numpy float32 array with samples in the range [-1.0, 1.0]

### Requirement: Non-Silent Segment Extraction
The system SHALL extract non-silent audio segments from WAV files based on amplitude thresholding.

#### Scenario: Extract segments above noise floor
- **WHEN** an audio file is processed with a top_db threshold of 30
- **THEN** only segments where amplitude exceeds the threshold are extracted

#### Scenario: Filter by minimum duration
- **WHEN** extracted segments are filtered with min_duration=0.25 seconds
- **THEN** only segments meeting or exceeding the duration threshold are returned

#### Scenario: Prioritize vocal tracks
- **WHEN** searching for audio files with a "vocal" or "vox" keyword
- **THEN** files matching these keywords are processed first in the extraction queue

### Requirement: LRC-Based Audio Segmentation
The system SHALL extract audio segments corresponding to LRC timestamp ranges for phrase-level training data.

#### Scenario: Extract audio for single LRC timestamp range
- **WHEN** given a LRC file with timestamp "[01:06.011]" and the next timestamp "[01:12.500]"
- **THEN** the corresponding audio segment from 1:06.011 to 1:12.500 is extracted from the WAV file

#### Scenario: Handle final segment without end timestamp
- **WHEN** processing the last timestamp in a LRC file with no following timestamp
- **THEN** extract audio from the timestamp to the end of the audio file

#### Scenario: Align LRC file with corresponding WAV
- **WHEN** a LRC file "08_03.lrc" is paired with audio file "08_03_main.wav"
- **THEN** the system automatically matches files by common prefix for extraction

### Requirement: Staged Raw Material Organization
The system SHALL organize raw audio exports and metadata in a standardized directory structure for training pipelines.

#### Scenario: Directory structure per track
- **WHEN** a track "08_03" is exported from Logic Pro
- **THEN** a directory "staged_raw_material/08_03/" contains the LRC file, YML metadata, main WAV, and individual track WAVs

#### Scenario: Track naming convention
- **WHEN** individual tracks are exported
- **THEN** files follow the pattern "{track_id}_{track_number}_{track_name}.wav" (e.g., "08_03_12_drums.wav")

#### Scenario: LRC and YML metadata pairing
- **WHEN** audio files are staged
- **THEN** each track has a corresponding "{track_id}.lrc" and "{track_id}.yml" file for lyrics/metadata

### Requirement: Integrated Pipeline Workflow
The system SHALL provide an end-to-end workflow from Logic Pro export to training-ready audio segments.

#### Scenario: Complete pipeline execution
- **WHEN** raw Logic Pro exports are placed in staging directory
- **THEN** the system converts SMPTE to LRC, extracts timestamped segments, filters by silence, and outputs training-ready WAV files

#### Scenario: Preserve source attribution
- **WHEN** audio segments are extracted
- **THEN** metadata tracks the source track, timestamp range, and original file path for reproducibility

#### Scenario: Generate segment artifacts
- **WHEN** segments are extracted for training
- **THEN** AudioChainArtifactFile objects are created with thread_id, duration, sample_rate, and file path metadata

### Requirement: Structure and Lyric Overlap Handling
The system SHALL handle time segment overlaps between structure boundaries (from manifest YML files) and lyric timestamps (from LRC files) to ensure complete musical phrases are captured.

#### Scenario: Extend segment to structure boundary
- **WHEN** a lyric timestamp is within a configurable threshold (e.g., 2 seconds) of a structure boundary (verse, chorus, bridge)
- **THEN** the extracted segment is extended to include the full structure section

#### Scenario: Align with manifest structure sections
- **WHEN** processing a track with both manifest YML structure data and LRC lyric timestamps
- **THEN** segment boundaries respect structure sections to avoid splitting mid-phrase or mid-section

#### Scenario: Prioritize musical coherence
- **WHEN** a lyric segment would end at an arbitrary point within a musical section
- **THEN** the segment is extended to the next logical structure boundary to maintain musical context

### Requirement: MIDI File Segmentation
The system SHALL handle MIDI files alongside audio files when creating timestamped training segments, applying best practices for MIDI-to-audio alignment.

#### Scenario: Pair MIDI with corresponding audio segment
- **WHEN** extracting an audio segment for timestamp range [01:06.011] to [01:12.500]
- **THEN** the corresponding MIDI file (if present) is also segmented to the same time range

#### Scenario: Preserve MIDI timing accuracy
- **WHEN** segmenting MIDI files based on LRC timestamps
- **THEN** MIDI note events are time-shifted to start at 0:00 in the extracted segment while preserving relative timing

#### Scenario: Handle tracks with MIDI exports
- **WHEN** a track directory contains both WAV files (e.g., "08_03_08_chords.mid") and audio files
- **THEN** MIDI files matching the naming pattern are included in segment extraction metadata

### Requirement: Maximum Segment Length
The system SHALL enforce maximum segment length constraints to ensure training data compatibility and prevent memory issues.

#### Scenario: Apply configurable maximum duration
- **WHEN** extracting an audio segment that would exceed the maximum length (e.g., 30 seconds)
- **THEN** the segment is truncated at the maximum length or split into multiple sub-segments

#### Scenario: Warn on truncated segments
- **WHEN** a segment is truncated due to maximum length constraints
- **THEN** a warning is logged with the original and truncated durations for auditability

#### Scenario: Split long sections intelligently
- **WHEN** a structure section (e.g., extended instrumental) exceeds maximum length
- **THEN** it is split into overlapping sub-segments with configurable overlap duration (e.g., 2 seconds) to maintain continuity
