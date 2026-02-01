# Tasks: Streamline EVP Artifact Storage

## Phase 1: EVPArtifact Model Changes

- [ ] **1.1** Remove `audio_segments` field from `EVPArtifact` class
- [ ] **1.2** Remove `noise_blended_audio` field from `EVPArtifact` class
- [ ] **1.3** Update `save_file()` to only save mosaic audio
- [ ] **1.4** Update `flatten()` to only include mosaic reference
- [ ] **1.5** Update `for_prompt()` to only reference mosaic (if needed)
- [ ] **1.6** Update YAML serialization to exclude removed fields

## Phase 2: Audio Tools Refactoring

- [ ] **2.1** Modify `get_audio_segments_as_chain_artifacts` to return in-memory data structure instead of saving files
- [ ] **2.2** Create new return type or modify `AudioChainArtifactFile` to support in-memory-only mode
- [ ] **2.3** Update `create_audio_mosaic_chain_artifact` to accept in-memory segments
- [ ] **2.4** Continue saving mosaic file (this is the keeper)
- [ ] **2.5** Modify `create_blended_audio_chain_artifact` to return in-memory audio bytes
- [ ] **2.6** Do not save blended file to disk
- [ ] **2.7** Update `transcription_from_speech_to_text` to accept in-memory audio bytes if needed

## Phase 3: Black Agent Integration

- [ ] **3.1** Update `generate_evp` method to work with in-memory segments/blended
- [ ] **3.2** Only pass mosaic to EVPArtifact constructor
- [ ] **3.3** Update mock mode EVP generation to match new structure
- [ ] **3.4** Remove segment and blended artifact creation from mock mode

## Phase 4: Cleanup Script

- [ ] **4.1** Create `scripts/cleanup_evp_intermediates.py`
- [ ] **4.2** Implement scan for `segment_*.wav` files in `chain_artifacts/*/audio/`
- [ ] **4.3** Implement scan for `blended*.wav` files in `chain_artifacts/*/audio/`
- [ ] **4.4** Add `--dry-run` flag to preview deletions
- [ ] **4.5** Add `--archive` flag to move files instead of delete
- [ ] **4.6** Report disk space to be freed
- [ ] **4.7** Update existing EVP YAML files to remove stale references

## Phase 5: Test Updates

- [ ] **5.1** Update EVP artifact tests to reflect new structure
- [ ] **5.2** Update mock YAML files to exclude segments/blended
- [ ] **5.3** Update Black agent tests for new EVP flow
- [ ] **5.4** Add test confirming only mosaic file is saved
- [ ] **5.5** Add test for cleanup script functionality

## Phase 6: Optional Debug Mode

- [ ] **6.1** Add `EVP_DEBUG_MODE` environment variable check
- [ ] **6.2** When enabled, save all intermediate files (old behavior)
- [ ] **6.3** Document debug mode in code comments

## Dependencies

- Phase 1 must complete before Phase 3
- Phase 2 must complete before Phase 3
- Phase 4 can run in parallel with Phases 1-3
- Phase 5 depends on Phases 1-3
- Phase 6 is optional and can run anytime after Phase 2

## Validation

After Phase 3:
```bash
# Run with mock mode
MOCK_MODE=true python run_white_agent.py start --mode single_agent --agent black

# Check artifact directory
ls -la chain_artifacts/*/audio/
# Should only show mosaic files, no segment_* or blended*
```

After Phase 4:
```bash
# Dry run cleanup on existing artifacts
python scripts/cleanup_evp_intermediates.py --dry-run

# Execute cleanup
python scripts/cleanup_evp_intermediates.py
```

Full validation:
```bash
pytest tests/ -v -k evp
python run_white_agent.py start --mode single_agent --agent black
```
