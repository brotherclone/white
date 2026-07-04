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

#### Short-phrase target widening
For phrases with a note count at or below `SHORT_PHRASE_NOTE_THRESHOLD` (default: 3),
the syllable target range presented in the generation prompt SHALL NOT be the strict
`floor(notes*0.8)`–`ceil(notes*1.15)` window. The pipeline SHALL widen the lower bound
of the range so that a single word or short phrase sustained across the notes (melisma)
is a legal, encouraged option, and a "spacious" ratio (fewer syllables than notes) on
one of these phrases SHALL NOT be treated as a fitting concern or drive the section's
overall verdict. This prevents the prompt from pinning every short phrase to a 1–2
syllable target, which previously converged candidates on a small pool of monosyllabic
words across songs.

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
- **AND** for phrases at or below `SHORT_PHRASE_NOTE_THRESHOLD`, the instruction notes
  that a single word or short phrase may be sustained across the notes (melisma)
  rather than requiring one syllable per note

#### Scenario: Paste-ready target
- **WHEN** all phrases in all sections have ratio 0.75–1.10
- **THEN** the candidate is flagged as paste-ready (no splits expected in ACE Studio)

#### Scenario: Short phrase does not force a monosyllabic line
- **WHEN** a phrase has 2 notes
- **THEN** the syllable target range presented to Claude is wider than
  `floor(2*0.8)`–`ceil(2*1.15)` (i.e. wider than 1–3 syllables)
- **AND** a lyric line with fewer syllables than notes for that phrase is not flagged
  as a fitting problem or included in the worst-case verdict calculation
