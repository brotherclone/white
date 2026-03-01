## ADDED Requirements

### Requirement: Lyric Draft Preservation

When a lyric candidate is promoted to `lyrics.txt`, the pipeline SHALL also write the
pre-edit draft to `lyrics_draft.txt` in the same directory so that human edits can be
diffed against the original generated text.

#### Scenario: Draft written on promotion

- **WHEN** `promote_part` is run against a `lyrics_review.yml` with one approved candidate
- **THEN** the approved `.txt` is copied to `melody/lyrics.txt` as before
- **AND** the same source file is also copied to `melody/lyrics_draft.txt`

#### Scenario: Draft not overwritten on re-promotion

- **WHEN** `promote_part` is run a second time (e.g., after changing the approved entry)
- **THEN** `lyrics_draft.txt` is overwritten to match the newly promoted candidate
- **AND** `lyrics.txt` is also overwritten

#### Scenario: Clean removes draft

- **WHEN** `promote_part --clean` is run on a melody review file
- **THEN** both `lyrics.txt` and `lyrics_draft.txt` are removed if they exist
