## 1. Catalog data file + schema

- [ ] 1.1 Create `app/data/artist_catalog.yml` (initially empty, with a header comment
       documenting the schema)
- [ ] 1.2 Document the schema in a top-level comment (slug, status, description,
       style_tags, chromatic_hint, chromatic_score, notes)

## 2. CLI — artist_catalog.py

- [ ] 2.1 Create `app/generators/artist_catalog.py`
- [ ] 2.2 Implement `collect_sounds_like(thread_dir=None, from_training_data=False) → list[tuple[str, int|None]]` —
       scan production plans and/or training parquet; parse `"Artist, discogs_id: N, ..."` format;
       return list of `(artist_name, discogs_id_or_None)` deduplicated
- [ ] 2.3 Implement `generate_description(artist_name, client) → str` — call Claude API
       with aesthetic-only prompt, return description text
- [ ] 2.4 Implement `generate_missing(thread_dir)` — load catalog, find uncatalogued
       artists, generate entries with `status: draft`, save
- [ ] 2.5 Implement `score_chromatic(catalog_path, onnx_path)` — load catalog, score
       non-null descriptions, write `chromatic_score` field
- [ ] 2.6 Implement `print_summary(catalog_path)` — counts by status, hint fill, score fill
- [ ] 2.7 CLI flags: `--thread`, `--generate-missing`, `--score-chromatic`, `--summary`,
       `--onnx-path`
- [ ] 2.8 Load `.env` via `python-dotenv` (same pattern as lyric_pipeline.py)

## 3. Pipeline injection — lyric_pipeline.py

- [ ] 3.1 Add `_load_artist_context(sounds_like: list[str]) → str` — reads catalog,
       returns formatted "STYLE REFERENCES" block (empty string if no matches)
- [ ] 3.2 Inject the block into `_build_prompt()` when non-empty
- [ ] 3.3 Print note for any `sounds_like` artists not found in catalog

## 4. Pipeline injection — chord_pipeline.py

- [ ] 4.1 Add same `_load_artist_context` call (import from lyric_pipeline or a shared util)
- [ ] 4.2 Inject style_tags + chromatic_hint into chord generation prompt when available

## 5. Tests

- [ ] 5.1 `test_collect_sounds_like_deduplicates` — multiple plans with overlapping artists
- [ ] 5.2 `test_generate_description_prompt_constraints` — verify prompt text excludes
       biographical instruction and includes copyright note
- [ ] 5.3 `test_generate_missing_idempotent` — second run makes no API calls
- [ ] 5.4 `test_load_artist_context_reviewed_preferred` — draft vs reviewed fallback
- [ ] 5.5 `test_load_artist_context_missing_artist_note` — missing entry prints note
- [ ] 5.6 `test_score_chromatic_skips_null_description`
