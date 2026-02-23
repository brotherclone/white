# Design: Streamline EVP Artifact Storage

## Current Architecture

### EVP Generation Pipeline

```
Source Audio Files (from MANIFEST_PATH)
         │
         ▼
┌─────────────────────────────────┐
│ get_audio_segments_as_chain_artifacts │
│ - Extract non-silent segments   │
│ - Save each segment as WAV      │  ◄── REMOVE file saving
│ - Return List[AudioChainArtifactFile] │
└─────────────────────────────────┘
         │
         ▼ (9 segment files saved)
┌─────────────────────────────────┐
│ create_audio_mosaic_chain_artifact │
│ - Shuffle and slice segments    │
│ - Concatenate into mosaic       │
│ - Save mosaic WAV               │  ◄── KEEP file saving
│ - Return AudioChainArtifactFile │
└─────────────────────────────────┘
         │
         ▼ (mosaic file saved)
┌─────────────────────────────────┐
│ create_blended_audio_chain_artifact │
│ - Mix with speech-like noise    │
│ - Save blended WAV              │  ◄── REMOVE file saving
│ - Return AudioChainArtifactFile │
└─────────────────────────────────┘
         │
         ▼ (blended file saved)
┌─────────────────────────────────┐
│ transcription_from_speech_to_text │
│ - Send to Whisper API           │
│ - Return transcript string      │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ EVPArtifact                     │
│ - audio_segments: [...]         │  ◄── REMOVE field
│ - audio_mosiac: path            │  ◄── KEEP
│ - noise_blended_audio: path     │  ◄── REMOVE field
│ - transcript: string            │  ◄── KEEP
└─────────────────────────────────┘
```

### Current Disk Output (per EVP)

```
chain_artifacts/<thread_id>/
├── audio/
│   ├── segment_1_<source>.wav     # ~2-5 seconds each
│   ├── segment_2_<source>.wav
│   ├── ...
│   ├── segment_9_<source>.wav     # 9 segment files
│   ├── mosiac.wav                 # ~10 seconds
│   └── blended.wav                # ~10 seconds
└── yml/
    └── evp_<thread_id>.yml        # References all above
```

---

## Proposed Architecture

### Streamlined Pipeline

```
Source Audio Files
         │
         ▼
┌─────────────────────────────────┐
│ get_audio_segments_in_memory    │
│ - Extract non-silent segments   │
│ - Return List[(np.ndarray, sr)] │  ◄── In-memory only
└─────────────────────────────────┘
         │
         ▼ (no files written)
┌─────────────────────────────────┐
│ create_audio_mosaic_chain_artifact │
│ - Accept in-memory segments     │
│ - Concatenate into mosaic       │
│ - Save mosaic WAV               │  ◄── ONLY file saved
│ - Return AudioChainArtifactFile │
└─────────────────────────────────┘
         │
         ▼ (mosaic file saved)
┌─────────────────────────────────┐
│ create_blended_audio_in_memory  │
│ - Mix with speech-like noise    │
│ - Return np.ndarray             │  ◄── In-memory only
└─────────────────────────────────┘
         │
         ▼ (no files written)
┌─────────────────────────────────┐
│ transcription_from_speech_to_text │
│ - Accept in-memory audio bytes  │
│ - Return transcript string      │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ EVPArtifact                     │
│ - audio_mosiac: path            │  ◄── ONLY audio reference
│ - transcript: string            │
└─────────────────────────────────┘
```

### Streamlined Disk Output (per EVP)

```
chain_artifacts/<thread_id>/
├── audio/
│   └── mosiac.wav                 # Only the mosaic
└── yml/
    └── evp_<thread_id>.yml        # References mosaic only
```

---

## Data Structure Changes

### EVPArtifact (Before)

```python
class EVPArtifact(ChainArtifact):
    audio_segments: Optional[List[AudioChainArtifactFile]]  # REMOVE
    transcript: Optional[str]
    audio_mosiac: Optional[AudioChainArtifactFile]
    noise_blended_audio: Optional[AudioChainArtifactFile]   # REMOVE
```

### EVPArtifact (After)

```python
class EVPArtifact(ChainArtifact):
    transcript: Optional[str]
    audio_mosiac: Optional[AudioChainArtifactFile]
```

### In-Memory Segment Type

```python
# New type for in-memory audio segments
AudioSegment = Tuple[np.ndarray, int]  # (audio_data, sample_rate)

def get_audio_segments_in_memory(...) -> List[AudioSegment]:
    """Extract segments without saving to disk."""
    ...
```

---

## API Changes

### audio_tools.py

| Function | Before | After |
|----------|--------|-------|
| `get_audio_segments_as_chain_artifacts` | Returns `List[AudioChainArtifactFile]`, saves WAVs | Returns `List[Tuple[np.ndarray, int]]`, no file I/O |
| `create_audio_mosaic_chain_artifact` | Reads from segment file paths | Accepts in-memory segment arrays |
| `create_blended_audio_chain_artifact` | Returns `AudioChainArtifactFile`, saves WAV | Returns `Tuple[np.ndarray, int]`, no file I/O |

### speech_tools.py

| Function | Before | After |
|----------|--------|-------|
| `transcription_from_speech_to_text` | Accepts `AudioChainArtifactFile` (reads from disk) | Accepts `bytes` or `np.ndarray` (in-memory) |

---

## Cleanup Script Design

### `scripts/cleanup_evp_intermediates.py`

```python
def cleanup_evp_intermediates(
    base_path: str = "chain_artifacts",
    dry_run: bool = True,
    archive_path: Optional[str] = None,
) -> CleanupReport:
    """
    Remove or archive segment and blended audio files.

    Scans for:
    - chain_artifacts/*/audio/segment_*.wav
    - chain_artifacts/*/audio/blended*.wav

    Optionally updates EVP YAML files to remove stale references.
    """
```

### CLI Interface

```bash
# Preview what would be deleted
python scripts/cleanup_evp_intermediates.py --dry-run

# Delete intermediate files
python scripts/cleanup_evp_intermediates.py

# Move to archive instead of delete
python scripts/cleanup_evp_intermediates.py --archive ./evp_archive

# Also update YAML files
python scripts/cleanup_evp_intermediates.py --update-yaml
```

---

## Migration Notes

1. **Backward Compatibility**: Existing EVP YAML files will have stale `audio_segments` and `noise_blended_audio` references. The `EVPArtifact` class should gracefully ignore these fields when loading old artifacts.

2. **File Deletion**: The cleanup script should be run once after deployment to free disk space from existing runs.

3. **Debug Mode**: For troubleshooting, set `EVP_DEBUG_MODE=true` to retain old file-saving behavior.
