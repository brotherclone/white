# Change: Fix rhyme-less/samey lyrics and repetitive melodies

## Why
Lyrics never rhyme and drift into word-list fragments ("rust gone black thing now this
now that now red") because generation is a single one-shot Claude call with no rhyme
instruction and no verification — even though the pipeline already knows exact
phrase/line boundaries from real MIDI rest gaps and never uses them for anything beyond
a syllable-count target. Melody candidates come out nearly identical run to run because
`--evolve` defaults off, its mutation is too weak to matter on the rare run where it's
used, and the shared album diversity penalty is too soft to outweigh the pipeline's
70%-weighted chromatic scoring.

## What Changes
- Derive an explicit per-section rhyme scheme (line-letter assignment, e.g. ABAB) from
  existing phrase-boundary data and inject line-pairing rhyme instructions into the
  lyric generation prompt
- Add a bounded generate → verify → revise loop: after drafting, check syllable-target
  misses and rhyme-pair failures, and send up to 2 targeted follow-up turns asking
  Claude to fix only the offending lines
- Add `pronouncing` (CMUdict) as a new dependency for rhyme verification, with a
  same-family suffix-heuristic fallback for out-of-dictionary/invented words
- Add hook-style content guidance to the prompt for chorus/refrain/hook sections
  (short, repeatable, one strong central phrase) instead of treating them like verses,
  reusing the existing label-based `EXACT` repeat-type classification
- Default `melody_pipeline.py --evolve` to on, with `--no-evolve` to opt out
  (**BREAKING**: changes default CLI behavior — evolved candidates now appear in the
  standard pool unless explicitly disabled)
- Strengthen mutation in `pattern_evolution.py`: raise mutation probability, widen
  mutation magnitude, and randomize the bass/melody crossover splice point instead of
  always splitting at the bar midpoint
- Steepen the shared album diversity-tracker curve so template reuse is penalised
  sooner and harder

## Impact
- Affected specs: `lyric-generation`, `melody-generation`, `pattern-evolution`,
  `template-diversity-tracking` (new capability, documents previously-unspecified
  behavior in `diversity_tracker.py`)
- Affected code:
  - `packages/generation/src/white_generation/pipelines/lyric_pipeline.py`
  - `packages/generation/src/white_generation/pipelines/melody_pipeline.py`
  - `packages/generation/src/white_generation/patterns/pattern_evolution.py`
  - `packages/generation/src/white_generation/util/diversity_tracker.py`
  - `packages/generation/pyproject.toml` (new `pronouncing` dependency)
- Side effect: `bass_pipeline.py` imports the same `diversity_tracker.py`, so bass
  candidates also get the steeper diversity curve even though bass isn't otherwise
  touched by this change
- Cost/latency: up to 3x Claude calls per lyric candidate when revisions are needed
  (bounded to 2 extra turns, only fires when a check actually fails)
