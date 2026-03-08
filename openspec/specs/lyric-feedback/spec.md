# lyric-feedback Specification

## Purpose
TBD - created by archiving change add-lyric-feedback-loop. Update Purpose after archive.
## Requirements
### Requirement: Post-Edit Lyric Rescoring

`song_evaluator.py` SHALL accept a `--rescore-lyrics` flag that runs Refractor on
the current `melody/lyrics.txt` (post-ACE-edit version) and writes the score alongside
the draft score in `song_evaluation.yml`.

#### Scenario: Rescore happy path

- **WHEN** `song_evaluator <production_dir> --rescore-lyrics` is run
- **AND** both `melody/lyrics.txt` and `melody/lyrics_draft.txt` exist
- **THEN** Refractor scores both files in text-only mode
- **AND** `song_evaluation.yml` is updated with `lyrics_draft_chromatic_match` and
  `lyrics_edited_chromatic_match` fields
- **AND** the delta (`edited − draft`) is recorded as `lyrics_chromatic_delta`

#### Scenario: Missing draft file

- **WHEN** `--rescore-lyrics` is run but `lyrics_draft.txt` does not exist
- **THEN** only `lyrics_edited_chromatic_match` is written (draft fields omitted)
- **AND** a warning is printed noting that the draft was not preserved at promotion time

#### Scenario: Missing lyrics.txt

- **WHEN** `--rescore-lyrics` is run but `melody/lyrics.txt` does not exist
- **THEN** rescoring is skipped with a clear warning
- **AND** no fields are written to `song_evaluation.yml`

#### Scenario: Rescore without Refractor

- **WHEN** `--rescore-lyrics` is run but Refractor is unavailable (wrong venv)
- **THEN** the pipeline exits with a clear error directing the user to use `.venv312`

---

### Requirement: Lyric Feedback Dataset Export

A CLI tool `lyric_feedback_export.py` SHALL walk one or more production directories,
collect (draft, edited) lyric pairs with metadata, compute per-section fitting metrics,
and write a JSONL file suitable for prompt engineering or future fine-tuning.

#### Scenario: Export from a thread

- **WHEN** `lyric_feedback_export --thread <shrink_wrapped_dir> --output feedback.jsonl` is run
- **THEN** every production directory under the thread that has both `lyrics.txt` and
  `lyrics_draft.txt` is included
- **AND** each output record contains: `song_slug`, `color`, `concept`, `bpm`, `time_sig`,
  `key`, `singer`, `vocal_sections` (list with `name`, `bars`, `repeat`, `total_notes`,
  `contour`), `draft_text`, `edited_text`, `draft_chromatic_match` (from
  `lyrics_review.yml`), `edited_chromatic_match` (from `song_evaluation.yml` if present,
  else null), `draft_fitting` (per-section), `edited_fitting` (per-section)

#### Scenario: No edits detected

- **WHEN** a song's `lyrics.txt` and `lyrics_draft.txt` have identical content
- **THEN** the record is still written but tagged with `"edited": false`
- **AND** a note is printed: "no edits detected for <song_slug>"

#### Scenario: Partial data tolerated

- **WHEN** a production directory has `lyrics.txt` but no `lyrics_draft.txt`
- **THEN** the record is written with `draft_text: null` and `edited: null`
- **AND** a warning is printed

#### Scenario: Output format

- **WHEN** the export completes
- **THEN** the JSONL file has one JSON object per song (not per section)
- **AND** a summary is printed: N songs exported, M with confirmed edits, K with null drafts

#### Scenario: Minimum viable dataset size warning

- **WHEN** the export completes with fewer than 20 songs with confirmed edits
- **THEN** the tool prints a note: "N pairs collected — suggest 20+ for reliable few-shot
  injection, 100+ for LoRA fine-tuning"

