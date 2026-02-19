# Streamline EVP Artifact Storage

## Summary

Reduce EVP artifact disk footprint by only persisting the audio mosaic. Remove storage of intermediate segment files and the blended audio file. Add a cleanup script for existing artifacts.

## Motivation

The current EVP generation pipeline saves three types of audio files:

1. **Segments** (~9 files): Individual audio clips extracted from source material
2. **Mosaic** (1 file): Random slices reassembled into a cohesive audio collage
3. **Blended** (1 file): Mosaic mixed with speech-like noise for STT processing

In practice:
- **Segments** are only intermediate data used to build the mosaic - no standalone value
- **Blended** is only needed transiently for Speech-to-Text - the transcript is what matters
- **Mosaic** is the only potentially useful audio for music production

This creates unnecessary disk usage (~10x more files than needed) and clutters the artifact directory.

## Scope

### In Scope
- Stop saving segment audio files to disk
- Stop saving blended audio file to disk
- Keep segment/blend processing in memory for pipeline operation
- Remove `audio_segments` and `noise_blended_audio` fields from EVPArtifact
- Update EVPArtifact YAML to only reference mosaic path
- Create cleanup script to remove existing segment/blended files
- Update mock mode EVP generation

### Out of Scope
- Audio processing logic (segmentation, mosaic creation, blending algorithms)
- Speech-to-text functionality
- EVP evaluation logic
- Transcript storage

## Key Changes

### EVPArtifact Structure

**Before:**
```yaml
transcript: "some words detected"
audio_segments:
  - chain_artifacts/<thread>/audio/segment_1.wav
  - chain_artifacts/<thread>/audio/segment_2.wav
  # ... up to 9 segments
audio_mosiac: chain_artifacts/<thread>/audio/mosiac.wav
noise_blended_audio: chain_artifacts/<thread>/audio/blended.wav
```

**After:**
```yaml
transcript: "some words detected"
audio_mosiac: chain_artifacts/<thread>/audio/mosiac.wav
```

### Audio Tools

- `get_audio_segments_as_chain_artifacts`: Return in-memory audio data without saving files
- `create_audio_mosaic_chain_artifact`: Accept in-memory segments, continue saving mosaic
- `create_blended_audio_chain_artifact`: Return in-memory blended audio without saving

### Cleanup Script

New script `scripts/cleanup_evp_intermediates.py`:
- Scan `chain_artifacts/*/audio/` for segment and blended files
- Optionally delete or move to archive
- Report space savings

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Debugging harder without intermediate files | Add optional `EVP_DEBUG_MODE` env var to save all files |
| Existing artifact references break | Cleanup script handles migration; YAML still valid if files missing |
| Memory pressure from in-memory processing | Already loading audio into memory; no net change |

## Success Criteria

1. New EVP artifacts contain only mosaic audio file
2. EVPArtifact YAML only lists mosaic path
3. Disk usage per EVP reduced by ~90%
4. Existing tests pass after mock updates
5. Cleanup script successfully removes old intermediate files
