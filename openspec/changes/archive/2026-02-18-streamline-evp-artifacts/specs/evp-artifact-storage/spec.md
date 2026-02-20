# EVP Artifact Storage

## Overview

EVP artifacts store only the audio mosaic and transcript. Intermediate processing files (segments, blended audio) are handled in-memory and not persisted to disk.

## REMOVED Requirements

### Requirement: Audio Segment File Storage
Individual audio segment files are no longer saved to disk during EVP generation.

#### Scenario: Segments processed in memory
- Given the Black agent extracts audio segments from source material
- When the segments are collected for mosaic creation
- Then no `segment_*.wav` files are written to disk
- And the segment data exists only in memory until mosaic creation

### Requirement: Blended Audio File Storage
The noise-blended audio file is no longer saved to disk.

#### Scenario: Blended audio processed in memory
- Given the mosaic audio is blended with speech-like noise
- When the blended audio is prepared for speech-to-text
- Then no `blended*.wav` file is written to disk
- And the blended data is passed directly to the STT service

### Requirement: EVPArtifact Segment References
The `audio_segments` field is removed from EVPArtifact.

#### Scenario: No segment paths in artifact
- Given an EVPArtifact is created
- When the artifact is serialized to YAML
- Then no `audio_segments` key appears in the output
- And the artifact class has no `audio_segments` field

### Requirement: EVPArtifact Blended Reference
The `noise_blended_audio` field is removed from EVPArtifact.

#### Scenario: No blended path in artifact
- Given an EVPArtifact is created
- When the artifact is serialized to YAML
- Then no `noise_blended_audio` key appears in the output
- And the artifact class has no `noise_blended_audio` field

## MODIFIED Requirements

### Requirement: EVP Artifact Persistence
EVPArtifact SHALL save only the mosaic audio file and YAML metadata.

#### Scenario: Minimal file output
- Given an EVP generation completes successfully
- When `evp_artifact.save_file()` is called
- Then exactly one audio file is written: `mosiac.wav`
- And exactly one YAML file is written with transcript and mosaic path
- And no other audio files are created in the artifact directory

### Requirement: Audio Mosaic as Primary Artifact
The audio mosaic SHALL be the only audio file persisted for EVP artifacts.

#### Scenario: Mosaic file saved
- Given audio segments are combined into a mosaic
- When `create_audio_mosaic_chain_artifact` completes
- Then the mosaic WAV file is written to `chain_artifacts/<thread>/audio/mosiac.wav`
- And the mosaic path is stored in the EVPArtifact

### Requirement: In-Memory Audio Processing
Audio segment extraction and blending SHALL operate in memory without file I/O.

#### Scenario: Memory-efficient processing
- Given the EVP pipeline processes source audio
- When segments are extracted and blended with noise
- Then all intermediate audio data remains in numpy arrays
- And file I/O occurs only for the final mosaic output

## ADDED Requirements

### Requirement: Legacy Artifact Field Tolerance
EVPArtifact SHALL gracefully ignore legacy fields when loading old artifacts.

#### Scenario: Loading old EVP YAML
- Given an EVP YAML file contains `audio_segments` and `noise_blended_audio` keys
- When the artifact is loaded into an EVPArtifact instance
- Then no error is raised
- And the legacy fields are silently ignored
- And the `transcript` and `audio_mosiac` fields are loaded correctly

### Requirement: Cleanup Script for Intermediate Files
A cleanup script SHALL exist to remove legacy segment and blended files.

#### Scenario: Dry run cleanup
- Given existing EVP artifacts contain segment and blended files
- When `cleanup_evp_intermediates.py --dry-run` is executed
- Then a report shows files that would be deleted
- And no files are actually removed

#### Scenario: Execute cleanup
- Given existing EVP artifacts contain segment and blended files
- When `cleanup_evp_intermediates.py` is executed
- Then all `segment_*.wav` files are deleted
- And all `blended*.wav` files are deleted
- And mosaic files are preserved

## Related Capabilities

- Black Agent EVP generation (modified to use in-memory processing)
- Speech-to-text transcription (modified to accept in-memory audio)
