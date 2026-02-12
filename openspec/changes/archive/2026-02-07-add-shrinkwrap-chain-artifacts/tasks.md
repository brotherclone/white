# Implementation Tasks

## 1. Thread Discovery
- [x] 1.1 Scan `chain_artifacts/` for thread directories (UUID-named)
- [x] 1.2 Identify the final song proposal YAML (`all_song_proposals_{thread_id}.yml`)
- [x] 1.3 Parse the final `SongProposalIteration` to extract title, BPM, key, concept, rainbow_color
- [x] 1.4 Handle threads with no song proposals (incomplete runs)

## 2. Directory Cleanup
- [x] 2.1 Define keep-list: final song proposal YAML, final audio, final MIDI, chromatic synthesis
- [x] 2.2 Define delete-list: per-color rebracketing analyses, transformation traces, facet evolution, intermediate proposals; also EVP intermediate segment/blended files
- [x] 2.3 Add `--dry-run` flag showing what would be deleted without actually deleting
- [x] 2.4 Add `--archive` flag to move debug files to a `.debug/` subdirectory instead of deleting

## 3. Thread Renaming
- [x] 3.1 Extract final song title from last `SongProposalIteration`
- [x] 3.2 Slugify title for filesystem safety (lowercase, hyphens, no special chars)
- [x] 3.3 Prepend album color prefix (e.g., `white-the-prism-protocol`)
- [x] 3.4 Copy thread to output directory with slug name (outputs to `shrinkwrapped/`, chain_artifacts untouched)
- [x] 3.5 Handle name collisions (append `-2`, `-3`, etc.)

## 4. Summary Manifest
- [x] 4.1 Generate `manifest.yml` in each shrink-wrapped thread directory with:
  - title, bpm, key, tempo, concept, rainbow_color, mood, genres
  - agent_name (which agent generated final proposal)
  - thread_id (original UUID for traceability)
  - timestamp
- [x] 4.2 Generate a top-level `shrinkwrapped/index.yml` listing all shrink-wrapped threads
- [x] 4.3 Include iteration count (how many agents contributed)

## 5. Batch Processing
- [x] 5.1 Process all threads in `chain_artifacts/` in one command
- [x] 5.2 Skip already-shrinkwrapped threads (detect by directory name in output)
- [x] 5.3 Print summary: N threads processed, N skipped, N failed
