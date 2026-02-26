## 1. Lyric pipeline

- [ ] 1.1 Write `app/generators/midi/lyric_pipeline.py`
  - Read production plan + melody/review.yml (vocal sections + approved melody patterns)
  - Read song concept + color + time_sig + bpm from production_plan.yml
  - Compute per-section note counts from `melody/approved/<label>*.mid` × `section.repeat`
    (NOT from melody/melody.mid — that file is never written by the pipeline)
  - Syllable counting: vowel-cluster heuristic — count contiguous vowel groups per word,
    floor 1 per word; strip `#` comment lines and `[section]` headers before counting
  - Derive syllable targets (notes × 0.75 – notes × 1.05) per vocal pass
  - Build structured Claude API prompt (concept, sections, melody contours, syllable targets)
  - Call Anthropic SDK, default model `claude-sonnet-4-6` (overridable via `--model`)
  - Parse response → one .txt per candidate in `melody/candidates/lyrics_NN.txt`
  - Score each candidate with ChromaticScorer (lyric_text mode, concept_emb)
  - Compute fitting score per pass (syllables/notes ratio + verdict)
  - Append to `melody/lyrics_review.yml` (never clobber existing entries or human statuses)
- [ ] 1.2 Add CLI entry point consistent with other pipelines
  - Args: `--production-dir`, `--num-candidates` (default 3), `--singer`, `--model`
  - `--sync-candidates`: scan `melody/candidates/*.txt`, add untracked stubs to
    `lyrics_review.yml` without regenerating or scoring
- [ ] 1.3 Write unit tests `tests/generators/midi/test_lyric_pipeline.py`
  - Syllable counter (vowel-cluster heuristic, comment stripping, section header stripping)
  - Parse candidate lyrics format (section header + lines)
  - Note count from approved MIDI (mock mido)
  - Review yml append logic (existing entries preserved)
  - Sync-candidates stub registration
  - Prompt builder (no API call in unit tests — mock the Anthropic client)

## 2. Promotion support for .txt

- [ ] 2.1 Extend `promote_part.py` to handle `.txt` alongside `.mid`
  - Approved `.txt` candidates → `melody/lyrics.txt` (no renaming prefix needed)
  - Fail if two approved candidates point to the same section set (same constraint)
- [ ] 2.2 Update tests for `promote_part.py` to cover `.txt` promotion

## 3. Yellow song worked example

- [x] 3.1 Write three candidate lyric drafts for Yellow manually
- [x] 3.2 Score all three with ChromaticScorer, write lyrics_review.yml with real scores
- [ ] 3.3 After pipeline is built: regenerate Yellow candidates through the pipeline
  to verify end-to-end flow (optional — manual drafts remain valid)
