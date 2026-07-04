## 1. Analysis module
- [x] 1.1 Create `packages/generation/src/white_generation/lyric_negative_constraints.py`
      with word-frequency and overuse-threshold logic (reuse the `Counter` +
      threshold/severity pattern from `generate_negative_constraints.py`)
- [x] 1.2 Implement `collect_lyric_texts(album_dir)` — walks `*/production/*/melody/lyrics.txt`
      (promoted) under the album root, matching the existing `scan_songs`-style glob
- [x] 1.3 Implement `analyze_word_frequency(texts)` returning overused words/short-word
      clusters above a configurable fraction threshold (mirrors `analyze_title_vocabulary`)
- [x] 1.4 CLI entry point: `--album-dir` (default `$SHRINKWRAP_OUTPUT_DIR`),
      `--output lyrics_negative_constraints.yml`, `--dry-run`
- [x] 1.5 Unit tests for frequency analysis and threshold flagging with a small fixture set

## 2. Pipeline integration
- [x] 2.1 Add a loader in `lyric_pipeline.py` that reads `lyrics_negative_constraints.yml`
      from the album root if present; no error or behavior change if absent
- [x] 2.2 Format constraints into a prompt block (mirrors `format_for_prompt`) and inject
      into both `_build_prompt` and `_build_white_cutup_prompt`
- [x] 2.3 Add a `--refresh-constraints` CLI flag to `lyric_pipeline.py` that regenerates
      `lyrics_negative_constraints.yml` before generating candidates
- [x] 2.4 Tests confirming the prompt includes the avoidance block when the constraints
      file is present, and generation behavior is unchanged when it is absent
