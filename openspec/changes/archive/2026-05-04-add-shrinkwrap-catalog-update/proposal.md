# Change: Shrinkwrap Catalog Update

## Why
Shrinkwrap extracts `title`, `key`, `bpm`, `rainbow_color`, and `singer` from song
proposal YML files but silently drops `sounds_like`. As a result, the artist catalog has
no data to work from after shrinkwrap runs — the catalog update must be triggered manually
beforehand, or it misses new artists entirely. The fix is two steps: include `sounds_like`
in `manifest_bootstrap.yml` so the data is never lost, then call `generate_missing()` at
the end of each shrinkwrap run for any artists not yet in the catalog.

## What Changes
- `scaffold_song_productions()` in `shrinkwrap_chain_artifacts.py` extracts `sounds_like`
  from each proposal YML and writes it into `manifest_bootstrap.yml`
- After all threads are processed, `shrinkwrap()` collects the union of `sounds_like`
  entries across all newly scaffolded productions and calls `artist_catalog.generate_missing()`
- `generate_missing()` in `artist_catalog.py` gains a programmatic call path that accepts
  an explicit artist list (not just CLI `--thread` / `--from-training-data` modes)
- Catalog update failure is non-fatal — logged as a warning, shrinkwrap continues

## Impact
- Affected specs: `chain-artifacts` (modified), `artist-style-catalog` (modified)
- Affected code:
  - `packages/composition/src/white_composition/shrinkwrap_chain_artifacts.py`
  - `packages/generation/src/white_generation/artist_catalog.py`
