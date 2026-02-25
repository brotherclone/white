## 1. Lyric pipeline

- [ ] 1.1 Write `app/generators/midi/lyric_pipeline.py`
  - Read production plan + melody/review.yml (vocal sections + approved melody patterns)
  - Read song concept + color + time_sig + bpm from production_plan.yml
  - Compute per-section note counts from `melody/melody.mid` using plan section timings
  - Derive syllable targets (notes × 0.75 – notes × 1.05) per vocal pass
  - Build structured Claude API prompt (concept, sections, melody contours, syllable targets)
  - Call Anthropic SDK (claude-sonnet-4-6), generate N candidates
  - Parse response → one .txt per candidate in `melody/candidates/lyrics_NN.txt`
  - Score each candidate with ChromaticScorer (lyric_text mode, concept_emb)
  - Compute fitting score per pass (syllables/notes ratio + verdict)
  - Write `melody/lyrics_review.yml` with candidates, chromatic scores, fitting scores, status=pending
- [ ] 1.2 Add `--lyric-pipeline` CLI entry point consistent with other pipelines
  - Args: `--production-dir`, `--num-candidates` (default 3), `--singer`
- [ ] 1.3 Write unit tests `tests/generators/midi/test_lyric_pipeline.py`
  - Parse candidate lyrics format (section header + lines)
  - Review yml generation
  - Prompt builder (no API call in unit tests — mock the client)

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
