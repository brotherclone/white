# Implementation Tasks

## 1. Thread Discovery
- [ ] 1.1 Scan `chain_artifacts/` for thread directories (UUID-named)
- [ ] 1.2 Identify the final song proposal YAML (`all_song_proposals_{thread_id}.yml`)
- [ ] 1.3 Parse the final `SongProposalIteration` to extract title, BPM, key, concept, rainbow_color
- [ ] 1.4 Handle threads with no song proposals (incomplete runs)

## 2. Directory Cleanup
- [ ] 2.1 Define keep-list: final song proposal YAML, final audio, final MIDI, chromatic synthesis
- [ ] 2.2 Define delete-list: per-color rebracketing analyses, transformation traces, facet evolution, intermediate proposals
- [ ] 2.3 Add `--dry-run` flag showing what would be deleted without actually deleting
- [ ] 2.4 Add `--archive` flag to move debug files to a `.debug/` subdirectory instead of deleting

## 3. Thread Renaming
- [ ] 3.1 Extract final song title from last `SongProposalIteration`
- [ ] 3.2 Slugify title for filesystem safety (lowercase, hyphens, no special chars)
- [ ] 3.3 Prepend album color prefix (e.g., `black-the-phantom-limb-protocol`)
- [ ] 3.4 Rename thread directory from UUID to slug
- [ ] 3.5 Handle name collisions (append `-2`, `-3`, etc.)

## 4. Summary Manifest
- [ ] 4.1 Generate `manifest.yml` in each shrink-wrapped thread directory with:
  - title, bpm, key, tempo, concept, rainbow_color, mood, genres
  - agent_name (which agent generated final proposal)
  - thread_id (original UUID for traceability)
  - timestamp
- [ ] 4.2 Generate a top-level `chain_artifacts/index.yml` listing all shrink-wrapped threads
- [ ] 4.3 Include iteration count (how many agents contributed)

## 5. Batch Processing
- [ ] 5.1 Process all threads in `chain_artifacts/` in one command
- [ ] 5.2 Skip already-shrinkwrapped threads (detect by presence of `manifest.yml`)
- [ ] 5.3 Print summary: N threads processed, N skipped, N failed
