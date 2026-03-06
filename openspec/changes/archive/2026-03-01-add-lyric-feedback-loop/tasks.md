## 1. Draft Preservation — promote_part.py

- [ ] 1.1 In the `.txt` promotion path, copy source also to `part_dir / "lyrics_draft.txt"`
- [ ] 1.2 In `--clean`, unlink `lyrics_draft.txt` if it exists (alongside `lyrics.txt`)
- [ ] 1.3 Tests: `test_promote_txt_writes_draft`, `test_promote_clean_removes_draft`

## 2. Post-Edit Rescoring — song_evaluator.py

- [ ] 2.1 Add `--rescore-lyrics` CLI flag
- [ ] 2.2 Implement `_rescore_lyrics(production_dir, plan_concept)` — scores `lyrics.txt`
       and `lyrics_draft.txt` via Refractor text-only, returns dict with
       `draft_chromatic_match`, `edited_chromatic_match`, `lyrics_chromatic_delta`
- [ ] 2.3 When `--rescore-lyrics` is set, call `_rescore_lyrics` and merge results into
       `song_evaluation.yml` (load → update → write, do not discard existing fields)
- [ ] 2.4 Tests: `test_rescore_lyrics_happy_path`, `test_rescore_lyrics_missing_draft`,
       `test_rescore_lyrics_missing_lyrics_txt`

## 3. Feedback Dataset Export — lyric_feedback_export.py

- [ ] 3.1 Create `app/generators/midi/lyric_feedback_export.py`
- [ ] 3.2 Implement `collect_song_record(production_dir) → dict | None` — reads plan,
       lyrics_review.yml, lyrics.txt, lyrics_draft.txt, song_evaluation.yml; computes
       fitting metrics for both draft and edited text; returns structured record
- [ ] 3.3 Implement `export_feedback(thread_dir, output_path)` — walks production dirs,
       collects records, writes JSONL, prints summary with dataset-size advisory
- [ ] 3.4 CLI: `--thread`, `--production-dir` (single song), `--output` (default
       `lyric_feedback.jsonl`)
- [ ] 3.5 Tests: `test_collect_song_record_happy_path`, `test_collect_no_draft`,
       `test_collect_no_edits_detected`, `test_export_size_advisory`
