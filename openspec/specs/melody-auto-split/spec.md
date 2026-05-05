# melody-auto-split Specification

## Purpose
TBD - created by archiving change add-melody-auto-split. Update Purpose after archive.
## Requirements
### Requirement: Syllable Segmentation
The system SHALL break each word in a lyric line into its constituent syllables using
rule-based hyphenation (`pyphen`, `en_US` locale). Punctuation SHALL be stripped before
lookup. If `pyphen` returns no split point for a word, the whole word is treated as one
syllable (safe fallback, never crashes).

#### Scenario: Multi-syllable word split
- **WHEN** `syllabify("beautiful")` is called
- **THEN** it returns `["beau", "ti", "ful"]` (3 syllables)

#### Scenario: Monosyllabic word unchanged
- **WHEN** `syllabify("dream")` is called
- **THEN** it returns `["dream"]` (1 syllable)

#### Scenario: Punctuation stripped before lookup
- **WHEN** `syllabify("falling,")` is called
- **THEN** punctuation is removed and the result is `["fall", "ing"]`

#### Scenario: Unknown word falls back to whole-word
- **WHEN** pyphen has no entry for the input token
- **THEN** the function returns `[token]` without raising an error

---

### Requirement: Syllable-to-Note Greedy Assignment
Within each MIDI phrase, syllables SHALL be assigned to notes left-to-right. When
syllables are exhausted before notes, remaining notes receive an empty string (melisma
continuation). When notes are exhausted before syllables, the excess syllables are
appended to the last note as a combined token (over-packed; flagged in the report).

#### Scenario: Equal count — clean 1:1 mapping
- **WHEN** a phrase has 4 notes and the lyric line has 4 syllables
- **THEN** each note receives exactly one syllable in order

#### Scenario: More notes than syllables — melisma
- **WHEN** a phrase has 6 notes and the lyric line yields 4 syllables
- **THEN** the first 4 notes receive syllables; notes 5 and 6 receive empty string

#### Scenario: More syllables than notes — over-pack flagged
- **WHEN** a phrase has 3 notes and the lyric line yields 5 syllables
- **THEN** the last note carries the remaining syllables concatenated
- **AND** the alignment report flags the phrase as `overpacked`

---

### Requirement: Long-Note Subdivision
The system SHALL subdivide any note that carries a multi-syllable token AND whose duration
≥ `min_split_ticks` (default: 1 beat = `ticks_per_beat`) into N equal-duration sub-notes,
where N is the syllable count. Each sub-note MUST inherit the original note's pitch,
channel, and velocity. The final sub-note absorbs any tick remainder from integer division.
Notes shorter than `min_split_ticks` MUST be left unsplit even if their token is
multi-syllable.
(default: 1 beat = `ticks_per_beat`) SHALL be split into N equal-duration sub-notes,
where N is the syllable count. Each sub-note inherits the original note's pitch, channel,
and velocity. The final sub-note absorbs any tick remainder from integer division. Notes
shorter than `min_split_ticks` are left unsplit even if their token is multi-syllable.

#### Scenario: Long note with di-syllabic word split into two
- **WHEN** a note has duration 960 ticks (1 beat at 960 tpb) and carries "fall-ing"
- **THEN** it is replaced by two notes of 480 ticks each at the same pitch

#### Scenario: Short note with multi-syllable word left unsplit
- **WHEN** a note has duration 120 ticks (< 1 beat at 960 tpb) and carries "beau-ti-ful"
- **THEN** the note is left unchanged and the phrase is flagged as `tight`

#### Scenario: Tick remainder absorbed by last sub-note
- **WHEN** a note of 100 ticks is split into 3
- **THEN** sub-notes are 33, 33, 34 ticks (last absorbs remainder)

---

### Requirement: Split MIDI Output
`auto_split_melody()` SHALL write the modified note grid to a new MIDI file at
`<source_stem>_split.mid` alongside the source file. The source MIDI SHALL NOT be
modified. All non-note events (tempo, time signature, program changes) are copied
unchanged.

#### Scenario: Output file written beside source
- **WHEN** `auto_split_melody("melody/approved/melody_verse.mid", ...)` completes
- **THEN** `melody/approved/melody_verse_split.mid` is created
- **AND** `melody/approved/melody_verse.mid` is unchanged

#### Scenario: Total note count non-decreasing
- **WHEN** any notes are split
- **THEN** the output MIDI has >= as many notes as the input MIDI

#### Scenario: No splits needed — output mirrors input
- **WHEN** every note carries a monosyllabic word or is below the duration threshold
- **THEN** the output MIDI is note-for-note identical to the input

---

### Requirement: Alignment Report
`auto_split_melody()` SHALL return a per-phrase alignment report as a list of dicts
describing each phrase's syllable count, note count, split count, and verdict
(`clean`, `tight`, `overpacked`).

#### Scenario: Report returned on success
- **WHEN** `auto_split_melody()` runs without error
- **THEN** the return value is a dict with keys `output_path` and `phrases`,
  where `phrases` is a list with one entry per phrase

#### Scenario: API endpoint surfaces report in response
- **WHEN** `POST /api/v1/production/auto-split-melody` succeeds
- **THEN** the JSON response contains `output_path` and the `phrases` alignment report

---

### Requirement: Auto-Split API Endpoint
The candidate server SHALL expose `POST /api/v1/production/auto-split-melody` accepting
`production_dir`, `phase_label` (the approved melody filename stem to split), and an
optional `min_split_beats` float (default 1.0). It SHALL invoke `auto_split_melody()` and
return the alignment report as JSON.

#### Scenario: Successful split via API
- **WHEN** a POST is sent with a valid `production_dir` and `phase_label`
- **THEN** the server responds 200 with `output_path` and `phrases` in JSON

#### Scenario: Missing approved MIDI returns 404
- **WHEN** `phase_label` does not correspond to an existing approved MIDI file
- **THEN** the server responds 404 with a descriptive error message

#### Scenario: Missing lyrics.txt returns 422
- **WHEN** no promoted `lyrics.txt` exists in the production directory
- **THEN** the server responds 422 explaining that lyrics must be promoted first

