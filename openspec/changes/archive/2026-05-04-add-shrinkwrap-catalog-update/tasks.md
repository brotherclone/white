## 1. manifest_bootstrap.yml — add sounds_like
- [x] 1.1 In `scaffold_song_productions()`, extract `sounds_like` from each proposal YML
        (same field-extraction block that reads `title`, `key`, `bpm`, etc.)
        — default to empty list if field absent
- [x] 1.2 Write `sounds_like` into the `manifest_bootstrap.yml` produced per song

## 2. artist_catalog.py — programmatic call path
- [x] 2.1 Add `generate_missing(artists: list[str]) -> list[str]` overload (or optional
        parameter) that accepts an explicit artist list instead of scanning for
        production_plan.yml files
        — returns list of newly added artist slugs
- [x] 2.2 Deduplication logic already in `collect_sounds_like()` is reused or extracted
        so it works from either source

## 3. shrinkwrap — post-scaffold catalog hook
- [x] 3.1 After `scaffold_song_productions()` finishes for all threads, collect the union
        of all `sounds_like` values from newly written `manifest_bootstrap.yml` files
- [x] 3.2 Call `artist_catalog.generate_missing(artists)` with the collected list
- [x] 3.3 Wrap the call in try/except — log warning on failure, do not re-raise
        (mirrors the existing non-fatal pattern for post-run shrinkwrap failures)

## 4. Tests
- [x] 4.1 Unit test: `scaffold_song_productions()` with a proposal YML that has `sounds_like`
          — assert `manifest_bootstrap.yml` contains the field
- [x] 4.2 Unit test: `scaffold_song_productions()` with a proposal YML missing `sounds_like`
          — assert `manifest_bootstrap.yml` has `sounds_like: []`, no error
- [x] 4.3 Unit test: `generate_missing(artists=[...])` with artists already in catalog
          — assert no Claude API call made, returns empty list
- [x] 4.4 Unit test: catalog update failure does not propagate out of `shrinkwrap()`
