## 1. Implementation (complete)
- [x] 1.1 Add `Phrase` dataclass and `extract_phrases(midi_path, rest_threshold_beats=0.5)`
- [x] 1.2 Rewrite `_compute_fitting()` to accept `melody_dir`; per-phrase scoring when
      MIDI available; fallback to section-level when no MIDI
- [x] 1.3 Add `_verdict_rank()` helper; overall verdict driven by worst-case phrase
- [x] 1.4 Update `_build_prompt()` to include phrase counts and per-phrase syllable
      targets; instruct Claude to write one line per phrase
- [x] 1.5 Add `_TEMPORAL_KEYWORDS`, `_SPATIAL_KEYWORDS`, `_ONTOLOGICAL_KEYWORDS` dicts
- [x] 1.6 Add `_keyword_score(text)` returning Refractor-compatible distribution dict
- [x] 1.7 Add `_blend_scores(refractor_result, keyword_result, confidence)`
- [x] 1.8 Wire hybrid blending in `run_lyric_pipeline()` when confidence < 0.2
- [x] 1.9 Extract phrases into `vocal_sections` entries before prompt build
- [x] 1.10 Update test call sites: `_compute_fitting(text, sections, _NO_MIDI_DIR)`

## 2. Tests (complete)
- [x] 2.1 All 38 existing tests passing
