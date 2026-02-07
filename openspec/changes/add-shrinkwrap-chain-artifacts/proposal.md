# Change: Add Chain Artifact Shrink-Wrap

## Why

Chain artifact threads are identified by UUIDs and contain debug artifacts (rebracketing analyses, transformation traces, facet evolution docs) mixed with final outputs (song proposals, audio, MIDI). After a run completes, there's no clean way to:
1. Know what the thread produced (have to dig through UUID directories)
2. Get the final song metadata (BPM, key, title, concept) without parsing YAML
3. Clean up debug artifacts to reduce clutter
4. Feed results to downstream processes

This makes it hard to audit what the pipeline has produced and impossible to programmatically avoid repeating past results.

## What Changes

- New `shrinkwrap` command/script that processes a chain artifact thread
- Renames thread directory from UUID to final song title (slugified)
- Deletes or archives debug/intermediate artifacts
- Generates a clean summary manifest with core song metadata
- **BREAKING**: Thread directories may be renamed (existing UUID references in any logs would break)

## Impact

- Affected specs: `chain-artifacts` (new capability)
- Affected code: new utility, reads from `chain_artifacts/` directory
- Affected artifacts: `chain_artifacts/{thread_id}/yml/all_song_proposals_*.yml`
