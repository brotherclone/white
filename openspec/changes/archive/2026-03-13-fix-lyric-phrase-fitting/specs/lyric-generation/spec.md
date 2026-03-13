## ADDED Requirements

### Requirement: Vocal Sections from Arrangement
The pipeline SHALL derive vocal sections from `arrangement.txt` (track 4 clips)
rather than from `production_plan.yml`. A clip's presence on track 4 is the sole
signal that it is a vocal section.

#### Scenario: Track 4 clips become vocal sections
- **WHEN** `arrangement.txt` exists and contains track 4 clips
- **THEN** the pipeline produces one lyric block per unique clip label in first-seen order
- **AND** `bars` is derived from clip duration and the song BPM/time_sig
- **AND** `repeat` equals the number of times that clip appears in the arrangement

#### Scenario: No production_plan.yml required
- **WHEN** the lyric pipeline runs and `production_plan.yml` is absent
- **THEN** the pipeline proceeds without error; all song metadata comes from the song
  proposal YAML

### Requirement: Keyword Hybrid Chromatic Scoring Fallback
When Refractor text-mode confidence is below 0.2, the pipeline SHALL blend the
Refractor output with a keyword-based chromatic score to improve differentiation
between candidates whose content signals (second-person address, future-tense verbs,
imagined/fabricated language) would otherwise be masked by low-confidence base-rate
predictions.

Blend weights:
- confidence < 0.1 → 30% Refractor + 70% keyword
- 0.1 ≤ confidence < 0.2 → 70% Refractor + 30% keyword
- confidence ≥ 0.2 → 100% Refractor (no blending)

#### Scenario: Low-confidence blending applied
- **WHEN** Refractor returns confidence < 0.2 for a lyric candidate
- **THEN** the keyword scorer is run on the candidate text and the two distributions
  are blended before `compute_chromatic_match()` is called

#### Scenario: High-confidence Refractor result not blended
- **WHEN** Refractor returns confidence ≥ 0.2
- **THEN** the keyword score is not computed; Refractor output is used directly

#### Scenario: Blended confidence raised
- **WHEN** blending is applied
- **THEN** the effective confidence passed to `compute_chromatic_match()` is increased
  by 0.15 (capped at 0.5) so the match weight reflects the hybrid signal strength

## MODIFIED Requirements

### Requirement: Lyric Fitting Score
The pipeline SHALL compute a per-phrase syllable fitting score for each candidate by
comparing syllable count per lyric line against note count per MIDI phrase group within
each vocal section. The overall verdict for a section is driven by the worst-case phrase,
not the section mean.

A MIDI phrase group is a sequence of note-on events separated from adjacent events by
a rest of at least 0.5 beats. Single-note phrases are permitted.

Fitting ratio = syllables / notes for each phrase.
- **paste-ready**: 0.75–1.10 — syllables map directly to notes with minimal adjustment
- **tight but workable**: 1.10–1.30 — a few notes will need splitting
- **splits needed**: >1.30 — significant manual work in ACE Studio
- **spacious**: <0.75 — melody has held notes; ACE Studio handles this automatically

When the approved MIDI file for a section is not available, the pipeline SHALL fall back
to section-level fitting (total syllables / total notes) with no error.

#### Note source: approved melody MIDIs per section
Note counts SHALL be derived from the approved melody MIDI files in `melody/approved/`,
not from a merged `melody/melody.mid` (which is never written by the pipeline).

#### Syllable counting algorithm
Syllable count SHALL use a vowel-cluster heuristic (no NLP dependency):
1. Strip comment lines (starting with `#`) and section header lines (`[name]`)
2. Split remaining text into words
3. For each word, count contiguous vowel-character groups (`[aeiouAEIOU]`) as
   syllables, with a floor of 1 syllable per word
4. Sum across all lines for that section

#### Scenario: Per-phrase fitting computed when MIDI available
- **WHEN** the approved MIDI for a section exists in `melody/approved/`
- **THEN** the pipeline extracts phrase groups separated by rests ≥ 0.5 beats,
  scores each lyric line against its corresponding phrase's note count, and records
  per-phrase ratios, verdicts, worst_ratio, worst_verdict, mean_ratio, and overall
  in `lyrics_review.yml`

#### Scenario: Worst-case phrase drives overall verdict
- **WHEN** a section has 4 phrases and 3 are paste-ready but 1 is splits-needed
- **THEN** the section's overall verdict is "splits needed"

#### Scenario: Fallback to section-level when no MIDI
- **WHEN** no approved MIDI exists for a section
- **THEN** fitting falls back to total syllables / total notes for that section;
  no error is raised

#### Scenario: Prompt includes phrase structure
- **WHEN** phrase data is available before the Claude API call
- **THEN** the generation prompt includes per-phrase note counts and syllable target
  ranges, and instructs Claude to write exactly one line per phrase

#### Scenario: Paste-ready target
- **WHEN** all phrases in all sections have ratio 0.75–1.10
- **THEN** the candidate is flagged as paste-ready (no splits expected in ACE Studio)

## ADDED Requirements

### Requirement: Keyword Hybrid Chromatic Scoring Fallback
When Refractor text-mode confidence is below 0.2, the pipeline SHALL blend the
Refractor output with a keyword-based chromatic score to improve differentiation
between candidates whose content signals (second-person address, future-tense verbs,
imagined/fabricated language) would otherwise be masked by low-confidence base-rate
predictions.

Blend weights:
- confidence < 0.1 → 30% Refractor + 70% keyword
- 0.1 ≤ confidence < 0.2 → 70% Refractor + 30% keyword
- confidence ≥ 0.2 → 100% Refractor (no blending)

#### Scenario: Low-confidence blending applied
- **WHEN** Refractor returns confidence < 0.2 for a lyric candidate
- **THEN** the keyword scorer is run on the candidate text and the two distributions
  are blended before `compute_chromatic_match()` is called

#### Scenario: High-confidence Refractor result not blended
- **WHEN** Refractor returns confidence ≥ 0.2
- **THEN** the keyword score is not computed; Refractor output is used directly

#### Scenario: Blended confidence raised
- **WHEN** blending is applied
- **THEN** the effective confidence passed to `compute_chromatic_match()` is increased
  by 0.15 (capped at 0.5) so the match weight reflects the hybrid signal strength
