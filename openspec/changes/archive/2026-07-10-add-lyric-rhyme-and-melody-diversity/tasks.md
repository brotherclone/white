## 1. Lyrics: rhyme scheme + verify/revise loop
- [x] 1.1 Add `pronouncing` to `packages/generation/pyproject.toml`
- [x] 1.2 Add rhyme-scheme assignment helper to `lyric_pipeline.py` (phrase count →
      default scheme, `song_proposal.yml` `rhyme_scheme` override, base-label reuse
      for repeated sections)
- [x] 1.3 Inject rhyme-pairing instructions into `_build_prompt` and
      `_build_white_cutup_prompt`
- [x] 1.4 Add rhyme-check helper (CMUdict comparison via `pronouncing` + suffix-heuristic
      fallback that never counts as a failure)
- [x] 1.5 Add generate → verify → revise loop around `_call_api` (multi-turn
      `messages`, max 2 follow-up turns, lists only failing lines + reason)
- [x] 1.6 Record verify/revise outcome (syllable misses fixed, rhyme misses fixed,
      turns used) in `lyrics_review.yml` per candidate
- [x] 1.7 Add chorus/hook content guidance block to `_build_prompt` /
      `_build_white_cutup_prompt`, gated on `_infer_repeat_type(label) == EXACT`
- [x] 1.8 Tests: rhyme-scheme assignment (default + override + reuse), rhyme-check
      helper (dictionary hit, fallback maybe, real mismatch), revise-loop
      triggering/bounding with a mocked Anthropic client, chorus guidance
      present/absent by label

## 2. Melody: evolve-by-default + stronger mutation
- [x] 2.1 Flip `--evolve` default to `True` in `melody_pipeline.py` CLI, add
      `--no-evolve`; keep `--evolve` accepted as a no-op
- [x] 2.2 Raise `_MUTATION_PROB` to 0.35 and widen mutation magnitude
      (±2 semitones / ±0.5 beat) in `pattern_evolution.py`
- [x] 2.3 Randomize the bass/melody crossover splice point in `_crossover_melody`
      (and the shared bass crossover helper if applicable)
- [x] 2.4 Tests: default-evolve invocation includes evolved candidates,
      `--no-evolve` excludes them, mutation magnitude bounds, crossover split
      point varies across calls

## 3. Shared diversity penalty
- [x] 3.1 Steepen thresholds/factors in `diversity_tracker.py` per design.md formula
- [x] 3.2 Tests: updated `diversity_factor` boundary cases (0, 1, 2, 5+ uses)

## 4. Validation
- [x] 4.1 `openspec validate add-lyric-rhyme-and-melody-diversity --strict`
- [x] 4.2 Run `pytest` for `lyric_pipeline`, `melody_pipeline`, `pattern_evolution`,
      `diversity_tracker` test modules
