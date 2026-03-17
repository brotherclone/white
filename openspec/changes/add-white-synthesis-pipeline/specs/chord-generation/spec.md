## ADDED Requirements

### Requirement: White Donor Mode — Bar Pool Construction
When the song proposal specifies `rainbow_color: White`, the chord pipeline SHALL build a
bar pool from approved chord MIDIs in the listed `sub_proposals` production directories
rather than generating progressions from a Markov chain.

For each sub-proposal directory, the pipeline SHALL:
1. Read `chords/review.yml` to obtain the donor song's key and BPM.
2. Load all MIDI files from `chords/approved/`.
3. Transpose every note by the semitone delta from the donor key root to the White key root;
   clamp resulting note values to [21, 108].
4. Replace the MIDI tempo message with one matching the White song's BPM.
5. Slice the transposed MIDI into individual bars using tick arithmetic
   (`bar_ticks = ticks_per_beat × beats_per_bar`); truncate notes at bar boundaries.

Each bar in the pool carries metadata: source production directory, donor color, approved
MIDI filename, and bar index within that file.

#### Scenario: bar pool built from multiple sub-proposals

- **WHEN** a White song proposal lists three sub-proposal directories
- **THEN** the bar pool contains bars from all approved chord MIDIs across all three directories
- **AND** each bar's metadata identifies its donor directory, color, and bar index
- **AND** all bar notes are transposed to the White key and the tempo is set to the White BPM

#### Scenario: transposition clamps out-of-range notes

- **WHEN** transposing a bar produces a note value below 21 or above 108
- **THEN** the note is clamped to the nearest boundary (21 or 108) and a warning is logged
- **AND** generation continues without error

#### Scenario: sub-proposal with no approved chords is skipped

- **WHEN** a sub-proposal directory has an empty `chords/approved/` folder
- **THEN** that directory contributes zero bars to the pool and a warning is logged
- **AND** the pipeline continues with bars from the remaining sub-proposals

---

### Requirement: White Donor Mode — Cut-Up Candidate Generation
The pipeline SHALL generate White chord candidates by randomly drawing bars from the bar
pool and shuffling them (the cut-up step). This replaces Markov generation for White.

Each candidate is constructed as:
1. Draw `progression_length` bars from the pool uniformly at random with replacement.
2. Shuffle the drawn bars into a random order.
3. Concatenate bars to form a complete candidate MIDI.

The same seed SHALL produce identical candidates. Theory and chromatic scoring,
MIDI output, and review.yml format are unchanged from non-White candidates.
Each candidate entry in `review.yml` SHALL include a `bar_sources` list recording
the donor directory, color, source filename, and bar index for each bar position.

#### Scenario: cut-up produces reproducible candidates

- **WHEN** White chord generation is run with `--seed 42`
- **THEN** the same bar draws and shuffle order are produced on every run with that seed

#### Scenario: bar_sources metadata recorded per candidate

- **WHEN** a White candidate is generated
- **THEN** `review.yml` contains a `bar_sources` list with one entry per bar,
  each recording `source_dir`, `donor_color`, `source_file`, and `bar_index`

#### Scenario: non-White pipeline unchanged

- **WHEN** the pipeline is run for any color other than White
- **THEN** Markov generation proceeds exactly as before; the donor mode is not activated
