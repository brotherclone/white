## 1. Parquet Loading
- [ ] 1.1 Add `load_clap_index(parquet_path) -> pd.DataFrame` — loads
      `training_data_clap_embeddings.parquet` (local or via `hf_hub_download`);
      returns rows with segment_id, song_slug, color, clap_embedding, and Refractor
      scores (temporal/spatial/ontological/confidence) if present
- [ ] 1.2 Handle missing Refractor score columns gracefully — if absent, fall back to
      computing `compute_chromatic_match()` from CLAP distribution (requires Refractor)

## 2. Retrieval
- [ ] 2.1 Add `retrieve_by_color(df, color, top_n) -> list[dict]` — filters to the
      target color, sorts by chromatic match score descending, returns top_n rows as
      dicts with segment_id, song_slug, match, scores, audio_path (if known)
- [ ] 2.2 Add `retrieve_by_clap_similarity(df, query_embedding, top_n) -> list[dict]`
      — cosine similarity between query_embedding and all CLAP embeddings; returns top_n
      regardless of color label (cross-color retrieval for future use)

## 3. Output
- [ ] 3.1 Add `write_sample_map(results, output_dir, color)` — writes `sample_map.yml`
      with ranked list: rank, segment_id, song_slug, color, match, audio_path
- [ ] 3.2 Add optional `--copy-audio` flag: if local media cache exists, copies matched
      audio files into `output_dir/audio/`; skips silently if files not found

## 4. CLI
- [ ] 4.1 `--color` (required): target color name (Red, Blue, Violet, etc.)
- [ ] 4.2 `--top-n` (optional, default 10): number of results
- [ ] 4.3 `--output-dir` (optional, default `./sample_retrieval`): write destination
- [ ] 4.4 `--parquet` (optional): override default parquet path
- [ ] 4.5 `--copy-audio` (flag): copy audio files if available

## 5. Tests
- [ ] 5.1 Unit test: `retrieve_by_color()` returns correct top-N sorted by match
- [ ] 5.2 Unit test: `retrieve_by_clap_similarity()` returns correct cosine ranking
- [ ] 5.3 Unit test: `write_sample_map()` produces valid YAML with expected fields
- [ ] 5.4 Integration test: stub parquet; verify end-to-end CLI write with no media copy
