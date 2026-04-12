# lyric-generation Specification

## Purpose
TBD - created by archiving change add-lyric-pipeline. Update Purpose after archive.
## Requirements
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
The pipeline SHALL score each candidate using Refractor in text-only mode
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

### Requirement: Lyric Review File
The pipeline SHALL write `melody/lyrics_review.yml` listing all candidates with their
chromatic scores and `status: pending`, following the same review pattern as MIDI phases.

#### Scenario: Review file written after generation
- **WHEN** the pipeline completes
- **THEN** `melody/lyrics_review.yml` exists with one entry per candidate
- **AND** each entry has `id`, `file`, `chromatic` scores, `fitting` scores, and `status: pending`

#### Scenario: Incremental append — existing entries are never clobbered
- **WHEN** `lyrics_review.yml` already exists with one or more entries
- **THEN** the pipeline appends new candidates only; existing entries (including any
  human-set `status` values) are preserved unchanged
- **AND** the next candidate `id` is derived from the highest existing `id` + 1

#### Scenario: Human approval workflow
- **WHEN** a human sets one candidate's `status` to `approved`
- **THEN** `promote_part.py` copies that file to `melody/lyrics.txt`
- **AND** other candidates are rejected/ignored

---

### Requirement: Sync Candidates
The pipeline SHALL support a `--sync-candidates` flag that registers manually-written
`.txt` files in `melody/candidates/` that are not yet tracked in `lyrics_review.yml`,
without regenerating or scoring anything.

#### Scenario: Orphan draft picked up by sync
- **WHEN** `lyric_pipeline.py --sync-candidates` is run
- **THEN** any `.txt` file in `melody/candidates/` with no matching entry in
  `lyrics_review.yml` is added as a stub entry with `status: pending` and no scores
- **AND** existing entries are not modified

---

### Requirement: API Model
The pipeline SHALL use `claude-sonnet-4-6` as the default generation model, with a
`--model` CLI argument to override.  The model name SHALL be passed directly to the
Anthropic SDK `model=` parameter.

#### Scenario: Default model
- **WHEN** `--model` is not supplied
- **THEN** `claude-sonnet-4-6` is used

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

### Requirement: White Lyric Cut-Up Mode
When the song is White, the lyric pipeline SHALL collect approved lyric text from each
sub-proposal listed in the song context and supply it to Claude as explicit source material,
instructing a cut-up: extract phrases and images from the source lyrics, recombine and
transform them into a coherent new lyric that feels synthesised rather than collaged.

Sub-lyric collection SHALL:
1. For each sub-proposal directory, check `melody/candidates/lyrics_review.yml` for entries
   with `status: approved` and load those files.
2. If no review file exists, load all `melody/candidates/lyrics_*.txt` files as candidates.
3. If a sub-proposal has no `melody/candidates/` directory, skip it silently.

The prompt MUST preserve the section structure (vocal section headers) of the White
production plan. Non-White lyric generation is unchanged.

#### Scenario: approved sub-lyrics used as cut-up source

- **WHEN** a White lyric pipeline run has three sub-proposals with approved lyric files
- **THEN** the Claude prompt includes a `## Source Lyrics` section with labeled excerpts
  from each sub-song
- **AND** Claude is instructed to cut up and recombine the phrases, not generate from scratch
- **AND** the output lyric files follow the same section-header format as non-White candidates

#### Scenario: fallback when no approved sub-lyrics available

- **WHEN** the sub-proposal directories have no approved lyric files
- **THEN** the pipeline falls back to standard lyric generation (concept-driven)
- **AND** a note is included in the prompt identifying this as a White synthesis song

#### Scenario: non-White lyric pipeline unchanged

- **WHEN** the lyric pipeline is run for any color other than White
- **THEN** sub-lyric collection is not performed; the standard prompt is used

### Requirement: Lyric Repeat Type
The lyric pipeline SHALL classify each vocal section instance with a `lyric_repeat_type`
indicating whether repeated plays of the same melody loop should receive identical,
structurally-related, or fully independent lyric content.

Three types are defined:
- **`exact`** — the section's lyrics are written once and repeated verbatim across all
  plays (chorus, refrain, hook). Only the first instance appears in the generation prompt;
  subsequent plays reuse the same lyric block.
- **`variation`** — each play of the loop receives its own lyric lines, but Claude is
  instructed to preserve rhyme scheme, meter, and core imagery across instances (verse 2
  vs verse 1). Each instance appears in the prompt with a numbered variation note.
- **`fresh`** — each instance is treated as fully independent with no structural
  relationship to other plays of the same loop (bridge, outro, climax). This is the
  default when no type is inferred or specified.

The `lyric_repeat_type` SHALL be:
1. Auto-detected from the loop label when no override exists:
   - Label contains `chorus`, `refrain`, or `hook` (case-insensitive) → `exact`
   - Label contains `verse` or `pre_chorus`/`pre-chorus` (case-insensitive) → `variation`
   - All other labels → `fresh`
2. Overridable via `lyric_repeat_type` in the corresponding `production_plan.yml` section.

#### Scenario: Chorus repeats verbatim
- **WHEN** a loop labelled `melody_chorus` appears three times in `arrangement.txt`
- **THEN** the generation prompt contains exactly one `[melody_chorus]` block instruction
- **AND** that block carries a note that it repeats verbatim
- **AND** the output lyrics file has one `[melody_chorus]` block
- **AND** ACE Studio receives the same lyric content for all three arrangement instances

#### Scenario: Verse instances vary
- **WHEN** a loop labelled `melody_verse` appears twice in `arrangement.txt`
- **THEN** the generation prompt contains a `[melody_verse]` block and a `[melody_verse_2]` block
- **AND** the second block carries a variation note referencing the first
- **AND** Claude writes distinct content for each, sharing rhyme scheme and meter

#### Scenario: Fresh instance generated independently
- **WHEN** a loop labelled `melody_bridge` appears once
- **THEN** the generation prompt contains one `[melody_bridge]` block with no repeat notes
- **AND** behaviour is identical to the current pipeline

#### Scenario: Manual override in production plan
- **WHEN** a section in `production_plan.yml` has `lyric_repeat_type: exact`
- **AND** the loop label would otherwise infer `variation` (e.g., `melody_verse`)
- **THEN** the pipeline uses the explicit value from the plan, not the inferred value

#### Scenario: Syllable fitting for exact repeats
- **WHEN** fitting is computed for a section with `exact_repeat` instances
- **THEN** the fitting result for each `exact_repeat` instance matches the first
  instance (same MIDI, same lyrics block)

