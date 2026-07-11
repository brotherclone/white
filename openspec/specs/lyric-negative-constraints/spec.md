# lyric-negative-constraints Specification

## Purpose
TBD - created by archiving change add-lyric-negative-constraints. Update Purpose after archive.
## Requirements
### Requirement: Lyric Word/Imagery Frequency Analysis
The system SHALL provide a `lyric_negative_constraints.py` module that walks an album's
promoted lyric files and computes word-frequency statistics distinct from the
proposal-level constraints produced by `generate_negative_constraints.py`.

#### Scenario: Album-wide frequency scan
- **WHEN** `lyric_negative_constraints.py --album-dir <shrink_wrapped_dir>` is run
- **THEN** every `melody/lyrics.txt` found under `<shrink_wrapped_dir>/*/production/*/`
  is read and tokenized
- **AND** a per-word frequency count is computed across all songs

#### Scenario: Overused short word flagged
- **WHEN** a monosyllabic content word (e.g. "blue", "dead") appears in more than 30%
  of scanned songs' lyrics
- **THEN** it is recorded as an overused word with `severity: avoid` and a human-readable
  reason string, following the same threshold/severity shape as
  `generate_negative_constraints.analyze_title_vocabulary`

#### Scenario: Independent from proposal-level constraints
- **WHEN** `lyric_negative_constraints.py` runs
- **THEN** it does not read or write `negative_constraints.yml`
- **AND** `generate_negative_constraints.py` is unmodified and continues to operate only
  on song-proposal metadata (key, BPM, title, concept, dialogue openers)

### Requirement: Lyric Negative Constraints File
The system SHALL write `lyrics_negative_constraints.yml` to the album root
(`$SHRINKWRAP_OUTPUT_DIR`), following the same file-location convention as
`index.yml` and `negative_constraints.yml`.

#### Scenario: Constraints file written
- **WHEN** the analysis completes with at least one overused word
- **THEN** `lyrics_negative_constraints.yml` is written with `overused_words`, per-word
  `count`/`fraction`/`reason`, and a `generated_from` field recording the album dir scanned

#### Scenario: No constraints yet
- **WHEN** the album has fewer than 2 songs with promoted lyrics
- **THEN** the file is still written with an empty `overused_words` list and a note
  that too few songs exist for meaningful frequency analysis

### Requirement: Lyric Pipeline Constraint Injection
`lyric_pipeline.py` SHALL load `lyrics_negative_constraints.yml` from the album root
when present and inject a formatted avoidance block into the generation prompt for
both standard and White cut-up modes.

#### Scenario: Constraints present
- **WHEN** `lyrics_negative_constraints.yml` exists and lists overused words
- **THEN** the Claude prompt built by `_build_prompt` (or `_build_white_cutup_prompt`
  for White) includes a block listing those words as language to avoid

#### Scenario: Constraints absent
- **WHEN** `lyrics_negative_constraints.yml` does not exist for the album
- **THEN** the pipeline proceeds without an avoidance block and without error or warning
  beyond an informational log line

### Requirement: Refresh Constraints CLI Flag
`lyric_pipeline.py` SHALL accept a `--refresh-constraints` flag that regenerates
`lyrics_negative_constraints.yml` from the current album state before generating
new candidates.

#### Scenario: Refresh before generation
- **WHEN** `lyric_pipeline.py --production-dir <dir> --refresh-constraints` is run
- **THEN** `lyric_negative_constraints.py`'s analysis is re-run against the album root
  and `lyrics_negative_constraints.yml` is overwritten before the prompt is built

