## ADDED Requirements

### Requirement: Rhyme Scheme Guidance
The lyric pipeline SHALL derive a per-section rhyme scheme (line-letter assignment,
e.g. `ABAB`, `AABB`) from the section's phrase count (from `extract_phrases`) before
building the generation prompt, and SHALL include explicit line-pairing rhyme
instructions in the prompt.

Scheme notation: each line is assigned a letter (rhyme group) or `X` (free — no rhyme
required). Lines sharing the same letter MUST rhyme with each other; `X` lines have no
rhyme constraint and are not checked by the verify/revise loop.

Default scheme by rhyme-eligible line count:
- 2 lines → `AA`
- 4 lines → `XAXA` (lines 2 and 4 rhyme; lines 1 and 3 are free)
- Any other count → unrhymed (no scheme enforced)

A `rhyme_scheme` map in `song_proposal.yml`, keyed by section label (any string in the
letter/`X` notation, or `none` for no rhyme), SHALL override the default for that
section.

Sections sharing a base label (e.g. `verse_1`, `verse_2`) SHALL reuse the same rhyme
scheme so repeated sections read consistently.

#### Scenario: Default XAXA for a four-line section
- **WHEN** a section has 4 rhyme-eligible phrases and no `rhyme_scheme` override
- **THEN** the prompt instructs lines 2 and 4 to rhyme, and leaves lines 1 and 3 free

#### Scenario: Proposal override
- **WHEN** `song_proposal.yml` sets `rhyme_scheme: {chorus: AABB}`
- **THEN** the chorus section's prompt uses AABB instead of the count-based default

#### Scenario: Partial (X) scheme override
- **WHEN** `song_proposal.yml` sets `rhyme_scheme: {verse_1: AXAX}`
- **THEN** the prompt instructs lines 1 and 3 to rhyme, and leaves lines 2 and 4 free

#### Scenario: Explicit no-rhyme
- **WHEN** `song_proposal.yml` sets `rhyme_scheme: {bridge: none}`
- **THEN** no rhyme instruction is included for the bridge section

#### Scenario: Repeated section reuses scheme
- **WHEN** `verse_1` is assigned `ABAB`
- **THEN** `verse_2` also uses `ABAB`

### Requirement: Syllable and Rhyme Verify-and-Revise Loop
After the initial draft is generated, the lyric pipeline SHALL check each phrase's
syllable count against its target range and each rhyme-scheme line pair for an actual
rhyme, using CMUdict-based rhyme comparison (`pronouncing`) with a same-family suffix
heuristic fallback for words absent from the dictionary. Fallback "maybe" matches SHALL
NOT be treated as failures.

When any phrase misses its syllable target or any rhyme pair fails, the pipeline SHALL
send up to 2 follow-up revision turns to Claude (same conversation, prior draft in
context) listing only the failing lines and the specific reason, requesting a full
corrected lyrics text in response.

After the revision budget is exhausted, the pipeline SHALL accept the best available
draft and continue to scoring — this loop does not block candidate generation from
completing.

#### Scenario: Syllable miss triggers a revision turn
- **WHEN** a phrase's line has a syllable count outside its target range
- **THEN** a follow-up turn is sent naming that line and its target range

#### Scenario: Rhyme miss triggers a revision turn
- **WHEN** a rhyme-scheme line pair's end words do not rhyme per CMUdict
- **THEN** a follow-up turn is sent naming both lines and their assigned rhyme letter

#### Scenario: Revision budget is bounded
- **WHEN** issues remain after 2 revision turns
- **THEN** the pipeline stops requesting revisions and uses the latest draft

#### Scenario: Dictionary-fallback words don't force a revision
- **WHEN** a line-final word is absent from CMUdict and the suffix heuristic finds a
  plausible (not confirmed) match
- **THEN** that rhyme pair is not treated as a failure

#### Scenario: Clean first draft skips revision entirely
- **WHEN** all phrases hit their syllable target and all rhyme pairs pass
- **THEN** no follow-up API calls are made

### Requirement: Chorus/Hook Content Style Guidance
The lyric pipeline SHALL include additional hook-style prompt guidance for any section
classified as `EXACT` repeat type (via `_infer_repeat_type` — labels containing
`chorus`, `refrain`, or `hook`), instructing Claude to write a short, highly
repeatable hook rather than narrative/descriptive verse content: prefer a single
strong central phrase (optionally the song or section title), permit repeating the
same line or phrase more than once within the section, and favor simpler/more
repetitive vocabulary than surrounding verses.

This guidance SHALL only be added when generating the first (`EXACT`) instance of a
repeated section; `EXACT_REPEAT` instances are copied verbatim from the first
instance and do not receive their own prompt.

#### Scenario: Chorus label gets hook guidance
- **WHEN** a section labeled `chorus_1` is generated (repeat_type EXACT)
- **THEN** the prompt includes hook-style guidance in addition to its syllable target
  and rhyme scheme

#### Scenario: Refrain/hook labels also qualify
- **WHEN** a section is labeled `refrain` or `hook`
- **THEN** it receives the same hook-style guidance as `chorus`

#### Scenario: Verse sections do not get hook guidance
- **WHEN** a section labeled `verse_1` is generated (repeat_type VARIATION)
- **THEN** the prompt does not include hook-style guidance

#### Scenario: Repeated chorus instance is not re-prompted
- **WHEN** `chorus_2` has repeat_type EXACT_REPEAT (second occurrence)
- **THEN** no separate prompt block is generated for it — the pipeline reuses the
  `chorus_1` text verbatim, matching existing (unchanged) repeat-copy behavior
