## Phase 1 — MIDI rebracketing utilities

- [x] 1.1 Create `app/generators/midi/pipelines/white_rebracketing.py`:
      - `transpose_midi(midi_bytes: bytes, semitone_delta: int) -> bytes`
      - `set_midi_bpm(midi_bytes: bytes, bpm: int) -> bytes`
      - `extract_bars(midi_bytes: bytes, ticks_per_beat: int, beats_per_bar: int) -> list[bytes]`
      - `concatenate_bars(bars: list[bytes], ticks_per_beat: int, bpm: int) -> bytes`
- [x] 1.2 Write unit tests in `tests/generators/midi/test_white_rebracketing.py`:
      22 tests covering all four functions

## Phase 2 — Bar pool builder

- [x] 2.1 Add `build_bar_pool(sub_proposal_dirs, white_key, white_bpm) -> list[dict]`
      to `white_rebracketing.py`
- [x] 2.2 Bar metadata includes: source_dir, source_file, donor_color, donor_key, bar_index
- [x] 2.3 Integration tests: bar count, metadata, missing review skipped, empty approved skipped

## Phase 3 — White chord pipeline mode

- [x] 3.1 `is_white_mode(song_info)` helper in `chord_pipeline.py`
- [x] 3.2 `generate_white_candidates(...)` in `chord_pipeline.py`:
      cut-up draw + shuffle, Refractor scoring, bar_sources metadata per candidate
- [x] 3.3 White branch in `run_chord_pipeline()` — skips Markov, calls build_bar_pool +
      generate_white_candidates, writes MIDI + scratch beats + review.yml
- [x] 3.4 `bar_sources` included in review.yml candidate entries
- [x] 3.5 `sub_proposals` added to `load_song_proposal_unified()` return dict
- [x] 3.6 Tests: is_white_mode, existing chord pipeline unaffected; 41 passing

## Phase 4 — White lyric cut-up mode

- [x] 4.1 `collect_sub_lyrics(sub_proposal_dirs) -> list[dict]` in `lyric_pipeline.py`
- [x] 4.2 `_build_white_cutup_prompt(...)` in `lyric_pipeline.py` with ## Source Lyrics
      section and synthesis fallback when no sub-lyrics available
- [x] 4.3 White branch in `run_lyric_pipeline()` at step 5 (prompt building)
- [x] 4.4 7 tests: collect_sub_lyrics (4) + _build_white_cutup_prompt (3)

## Phase 5 — Integration test and docs

- [x] 5.1 Live integration test: ran against the-breathing-machine-learns-to-sing thread
      with 7 sub-proposals; 122 bars in pool, 5 candidates written, review.yml verified
- [x] 5.2 Workflow documented in WORKFLOW docstring at top of white_rebracketing.py
- [x] 5.3 `--sub-proposals` CLI argument added to `chord_pipeline.py`
