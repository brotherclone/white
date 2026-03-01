# Change: Add Lyric Feedback Loop

## Why

The lyric pipeline generates drafts that humans edit in ACE Studio for syllable fit and
artistic quality. Once editing happens, the draft is overwritten and the signal is lost.
Capturing (draft → edited) pairs closes the feedback loop: it enables post-edit rescoring
to measure how much human judgment diverged from the initial Refractor ranking, and
accumulates structured training data that can later be used for few-shot prompt injection
or DPO-style fine-tuning when enough songs are complete.

## What Changes

- **Draft preservation**: `promote_part` copies the approved candidate to `lyrics.txt`
  and also writes `lyrics_draft.txt` — the pre-edit snapshot — so human edits can be
  diffed against the original.
- **Post-edit rescoring**: `song_evaluator.py` gains a `--rescore-lyrics` flag that runs
  Refractor on the current `lyrics.txt` (post-edit) and appends the score to
  `song_evaluation.yml` alongside the original draft score.
- **Feedback dataset export**: A new `lyric_feedback_export.py` CLI walks all production
  directories, collects (draft, edited) pairs with metadata, computes per-section fitting
  deltas, and writes a structured JSONL file suitable for prompt engineering or fine-tuning.

## Impact

- Affected specs: `lyric-generation` (MODIFIED — draft preservation), `lyric-feedback` (ADDED — new capability)
- Affected code:
  - `app/generators/midi/promote_part.py` — copy draft alongside `lyrics.txt`
  - `app/generators/midi/song_evaluator.py` — `--rescore-lyrics` flag
  - `app/generators/midi/lyric_feedback_export.py` — new file (~150 lines)
  - `tests/generators/midi/test_lyric_feedback.py` — new test file
