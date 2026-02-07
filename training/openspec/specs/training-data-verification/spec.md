# training-data-verification Specification

## Purpose
TBD - created by archiving change add-training-data-verification. Update Purpose after archive.
## Requirements
### Requirement: Segment Extraction for Verification
The system SHALL extract audio and MIDI segments from training parquet files as playable media files for human review.

#### Scenario: Random segment extraction
- **WHEN** the verification tool is run with `--random N`
- **THEN** N random segments are extracted as WAV files from the media parquet

#### Scenario: Targeted extraction by album color
- **WHEN** the verification tool is run with `--color Green`
- **THEN** only segments belonging to the Green album are extracted

#### Scenario: Targeted extraction by song
- **WHEN** the verification tool is run with `--song 05_01`
- **THEN** only segments from song 05_01 are extracted

#### Scenario: MIDI extraction
- **WHEN** a segment has non-null `midi_binary` in the media parquet
- **THEN** the MIDI data is written as a playable .mid file alongside the audio

#### Scenario: Descriptive output naming
- **WHEN** segments are extracted
- **THEN** output files are named with song ID, segment index, section name, and timestamps

### Requirement: Modality Coverage Report
The system SHALL generate a coverage report showing modality presence across the training dataset.

#### Scenario: Coverage by album color
- **WHEN** the coverage report is generated
- **THEN** it shows audio %, MIDI %, and text % for each album color

#### Scenario: Anomaly detection
- **WHEN** a song has 0 extracted segments
- **THEN** it is flagged as an anomaly in the report

#### Scenario: Comparison with previous extraction
- **WHEN** a previous extraction's metadata parquet is available
- **THEN** the report shows delta in segment counts per album color

### Requirement: Audio Fidelity Verification
The system SHALL verify that parquet-stored audio matches the original source files.

#### Scenario: Sample rate verification
- **WHEN** audio is decoded from the parquet binary
- **THEN** the sample rate matches the stored `audio_sample_rate` column

#### Scenario: Duration verification
- **WHEN** audio is decoded from the parquet binary for a segment
- **THEN** the decoded duration matches the expected segment duration within 0.1s tolerance

#### Scenario: MIDI content verification
- **WHEN** MIDI binary is decoded from the parquet
- **THEN** at least one note event exists and falls within the segment's timestamp range

