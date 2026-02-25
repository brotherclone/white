## ADDED Requirements

### Requirement: Lyric Candidate Generation
The lyric pipeline SHALL generate N complete song lyric drafts (covering all vocal
sections) per run, calling the Claude API with a prompt built from the song concept,
color target, approved melody patterns, and BPM/time-signature.

#### Scenario: Standard generation run
- **WHEN** `lyric_pipeline.py --production-dir <dir> --num-candidates 3` is run
- **THEN** three candidate files are written to `melody/candidates/lyrics_01.txt`,
  `lyrics_02.txt`, `lyrics_03.txt`
- **AND** each file contains `[section_name]` headers for every vocal section in the
  production plan, followed by lyric lines

#### Scenario: Vocal sections derived from plan
- **WHEN** the production plan has sections with `vocals: true`
- **THEN** only those sections receive lyric content in each candidate
- **AND** instrumental sections are omitted from the lyrics files

---

### Requirement: Lyric Chromatic Scoring
The pipeline SHALL score each candidate using ChromaticScorer in text-only mode
(DeBERTa lyric embedding + concept embedding, null audio/MIDI inputs).

#### Scenario: Scoring produces chromatic output
- **WHEN** a candidate lyrics file is scored
- **THEN** the scorer returns `temporal`, `spatial`, `ontological` scores and a
  `match` value representing alignment with the song's color target

#### Scenario: Low CLAP confidence expected
- **WHEN** text-only scoring is used (no audio or MIDI)
- **THEN** `confidence` is expected to be low (the model uses null audio/MIDI inputs)
- **AND** this is flagged as informational, not a scoring failure

---

### Requirement: Lyric Fitting Score
The pipeline SHALL compute a fitting score for each candidate by comparing syllable
count against note count per vocal section, using the merged `melody/melody.mid` as
the note source.  The score indicates how much manual note-splitting will be required
in ACE Studio.

Fitting ratio = syllables / notes for each melody pass.
- **paste-ready**: 0.75–1.10 — syllables map directly to notes with minimal adjustment
- **tight but workable**: 1.10–1.30 — a few notes will need splitting
- **splits needed**: >1.30 — significant manual work in ACE Studio
- **spacious**: <0.75 — melody has held notes; ACE Studio handles this automatically

#### Scenario: Fitting computed per vocal pass
- **WHEN** a candidate lyrics file is scored
- **THEN** the pipeline counts notes per melody pass from `melody/melody.mid` using
  bar boundaries from the arrangement (derived from `production_plan.yml` section
  timings), and counts syllables per lyrics section using a vowel-group heuristic
- **AND** per-pass ratios and verdicts are stored in `lyrics_review.yml` under
  a `fitting` key on each candidate entry

#### Scenario: Paste-ready target
- **WHEN** all passes have ratio 0.75–1.10
- **THEN** the candidate is flagged as paste-ready (no splits expected in ACE Studio)

#### Scenario: Fitting informs prompt
- **WHEN** the Claude API prompt is built for lyric generation
- **THEN** it includes per-section syllable targets derived from note counts
  (e.g. "verse: 12–17 syllables, chorus: 26–37 syllables")
- **AND** the generated lyrics are constrained to those targets

---

### Requirement: Lyric Review File
The pipeline SHALL write `melody/lyrics_review.yml` listing all candidates with their
chromatic scores and `status: pending`, following the same review pattern as MIDI phases.

#### Scenario: Review file written after generation
- **WHEN** the pipeline completes
- **THEN** `melody/lyrics_review.yml` exists with one entry per candidate
- **AND** each entry has `id`, `file`, `chromatic` scores, and `status: pending`

#### Scenario: Human approval workflow
- **WHEN** a human sets one candidate's `status` to `approved`
- **THEN** `promote_part.py` copies that file to `melody/lyrics.txt`
- **AND** other candidates are rejected/ignored

---

### Requirement: Lyric File Format
Lyrics files SHALL use `[section_name]` headers (matching production plan section names)
with lyric lines below, and empty lines between phrases.

#### Scenario: Section header matching
- **WHEN** a lyrics file has `[verse]` and `[chorus]` headers
- **THEN** those names match the `name` field of the corresponding plan sections

#### Scenario: Editable after promotion
- **WHEN** a candidate is promoted to `melody/lyrics.txt`
- **THEN** the human MAY edit the file directly for final refinement
- **AND** the song evaluator reads the promoted file as-is

---

### Requirement: Lyric Promotion
`promote_part.py` SHALL support `.txt` files in addition to `.mid` files, copying the
approved candidate to `melody/lyrics.txt`.

#### Scenario: Promote approved lyrics
- **WHEN** a candidate with `status: approved` exists in `lyrics_review.yml`
- **THEN** `promote_part.py` copies the candidate `.txt` to `melody/lyrics.txt`

#### Scenario: Multiple approved candidates rejected
- **WHEN** two candidates in `lyrics_review.yml` both have `status: approved`
- **THEN** `promote_part.py` fails with a clear error message
