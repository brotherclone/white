# chord-generation Specification

## Purpose
TBD - created by archiving change add-music-production-pipeline. Update Purpose after archive.
## Requirements
### Requirement: Chord Pipeline Input

The chord generation pipeline SHALL accept a song proposal YAML file path and a shrink_wrapped thread directory as input. It SHALL extract key, mode, BPM, time signature, concept text, and rainbow color from the song proposal and thread manifest.

#### Scenario: Load song proposal from shrink_wrapped thread

- **WHEN** the pipeline is invoked with a thread directory and song proposal filename
- **THEN** it SHALL parse the song proposal YAML for key, BPM, time signature, and concept
- **AND** parse the thread manifest for the full concept text and rainbow color
- **AND** reject proposals with missing key or rainbow color fields

#### Scenario: Derive chromatic target from rainbow color

- **WHEN** the rainbow color is extracted from the song proposal
- **THEN** the pipeline SHALL map it to target mode distributions (temporal, spatial, ontological)
- **AND** use uniform distributions for White and Black proposals

### Requirement: Candidate Generation

The pipeline SHALL generate chord primitive candidates by combining a Markov chord progression
with a randomly-sampled harmonic rhythm (HR) distribution and strum articulation pattern. Each
candidate is a complete chord primitive — voicings, rhythm, and articulation — ready for
promotion without further post-processing.

#### Scenario: Graph-guided generation

- **WHEN** the pipeline generates candidates
- **THEN** it SHALL use the function transition graph for weighted Markov sampling
- **AND** generate at least 50 candidates per invocation (configurable)
- **AND** constrain all candidates to the target key and mode from the song proposal

#### Scenario: HR and strum baked into each candidate

- **WHEN** a chord progression is generated
- **THEN** the pipeline SHALL randomly sample a harmonic rhythm distribution (from the half-bar
  duration grid) and a strum articulation pattern (from the strum template library)
- **AND** apply both to the progression's voicings before writing the candidate MIDI
- **AND** the same seed SHALL produce identical HR + strum pairings for reproducibility

#### Scenario: Progression length from time signature

- **WHEN** the song proposal specifies a time signature
- **THEN** the pipeline SHALL use an appropriate default progression length (e.g., 4 bars for 4/4, 7 bars for 7/8)
- **AND** allow the user to override the length via CLI parameter

#### Scenario: Reproducible generation

- **WHEN** a random seed is provided
- **THEN** the same seed SHALL produce identical candidates for the same song proposal

### Requirement: Composite Scoring

The pipeline SHALL score each candidate using both music theory metrics (from the chord prototype) and chromatic fitness (from Refractor), producing a single composite ranking.

#### Scenario: Music theory scoring

- **WHEN** a chord progression candidate is scored
- **THEN** the pipeline SHALL compute melody score, voice leading score, variety score, and graph probability score using the existing scoring functions

#### Scenario: Chromatic scoring

- **WHEN** a chord progression candidate is scored
- **THEN** the pipeline SHALL convert the candidate to MIDI bytes, encode the concept text via `Refractor.prepare_concept()`, and score with `Refractor.score()`
- **AND** the concept embedding SHALL be computed once and reused across all candidates in the batch

#### Scenario: Composite ranking

- **WHEN** all candidates are scored
- **THEN** the pipeline SHALL compute a weighted composite score (default: 30% theory, 70% chromatic)
- **AND** rank candidates by composite score descending
- **AND** allow the user to configure scoring weights via CLI or config

### Requirement: MIDI Output

The pipeline SHALL export each top-ranked candidate as a standard MIDI file alongside a scratch
beat MIDI for auditioning.

#### Scenario: MIDI file generation

- **WHEN** the top N candidates are selected (default N=10)
- **THEN** the pipeline SHALL write each as a `.mid` file in the song's production directory
- **AND** the MIDI file SHALL use the song proposal's BPM for tempo
- **AND** chord notes SHALL reflect the baked-in HR distribution and strum articulation

#### Scenario: Output directory structure

- **WHEN** MIDI files are generated
- **THEN** they SHALL be placed in `<thread>/production/<song_slug>/chords/candidates/`
- **AND** the directory SHALL be created if it does not exist

#### Scenario: Scratch beat generation

- **WHEN** a candidate MIDI file is written
- **THEN** the pipeline SHALL also write a companion scratch beat MIDI named
  `<candidate>_scratch.mid` in the same candidates directory
- **AND** the scratch beat SHALL use the lowest-energy template from the genre family inferred
  from the song proposal, matching the candidate's bar length and BPM
- **AND** scratch files SHALL be listed in `review.yml` with `scratch: true` and SHALL NOT be
  eligible for promotion

### Requirement: CLI Interface

The chord pipeline SHALL be invocable from the command line.

#### Scenario: Basic invocation

- **WHEN** the user runs the pipeline CLI
- **THEN** it SHALL accept `--thread` (shrink_wrapped thread directory), `--song` (song proposal filename), and optional `--seed`, `--num-candidates`, `--top-k`, `--theory-weight`, `--chromatic-weight` parameters

#### Scenario: Progress output

- **WHEN** the pipeline is running
- **THEN** it SHALL print progress (loading, generating, scoring, writing) to stdout
- **AND** print the top candidates with their composite scores and score breakdowns

### Requirement: CLI Interface — HR and Strum Parameters

The chord pipeline CLI SHALL expose controls for HR and strum generation.

#### Scenario: HR and strum seed propagation

- **WHEN** the user provides `--seed`
- **THEN** the seed SHALL deterministically control Markov generation, HR distribution sampling,
  and strum pattern sampling together

#### Scenario: Strum pattern override

- **WHEN** the user provides `--strum-patterns` (comma-separated list)
- **THEN** only the specified strum patterns SHALL be used when pairing with chord progressions
- **AND** if the flag is omitted, all patterns applicable to the song's time signature are eligible

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

### Requirement: Diatonic Workhorse Candidates

After Markov candidates are scored and ranked, `chord_pipeline.py` SHALL
generate an additional set of diatonic workhorse candidates from the chord
bank and append them to `review.yml`.

Diatonic candidates are assembled using the existing
`ChordProgressionGenerator.get_chord_by_function()` method — no new MIDI
synthesis or Markov traversal is required. They are intended as grounded,
guitar-playable contrast options, annotated so the reviewer knows to assign
them to verse sections. (Section labels are assigned by the human reviewer
after generation; they are not known at pipeline time.)

The following patterns SHALL be attempted. Major-key songs use the Major set;
minor-key songs use the Minor set:

**Major patterns**
| pattern_name | degrees |
|---|---|
| `I_V_vi_IV` | I – V – vi – IV |
| `I_IV_V` | I – IV – V |
| `I_vi_IV_V` | I – vi – IV – V |
| `ii_V_I` | II – V – I |

**Minor patterns**
| pattern_name | degrees |
|---|---|
| `i_VII_VI_VII` | i – VII – VI – VII |
| `i_VI_III_VII` | i – VI – III – VII |
| `i_iv_v` | i – iv – v |
| `i_VI_VII_i` | i – VI – VII – i |

For each pattern, one chord per degree SHALL be selected from the bank. If no
chord is found for a given degree the pattern SHALL be skipped silently.

Each diatonic candidate SHALL be written to `candidates/` as a MIDI file and
added to `review.yml` with:
- `id`: `diatonic_{pattern_name}`
- `source`: `diatonic`
- `scores`: `null`
- `label`: `null`
- `status`: `pending`
- `notes`: `"Diatonic workhorse — assign to verse sections"`
- `rank`: `null` (listed after all scored Markov candidates)

#### Scenario: A minor song gets diatonic candidates appended

- **GIVEN** a song in A minor
- **WHEN** chord_pipeline runs
- **THEN** `review.yml` contains Markov candidates followed by up to 4
  diatonic candidates with `source: diatonic` and `scores: null`
- **AND** their IDs follow the `diatonic_{pattern_name}` convention

#### Scenario: Pattern degree missing from bank is skipped

- **GIVEN** a pattern degree that has no matching chord in the bank for the
  current key
- **WHEN** diatonic candidate assembly runs
- **THEN** that pattern is silently skipped
- **AND** remaining patterns are still added

#### Scenario: White cut-up mode is unaffected

- **GIVEN** a song using White donor cut-up mode
- **WHEN** chord_pipeline runs
- **THEN** no diatonic candidates are added (White mode has its own candidate logic)


### Requirement: Constrained Chord Generation Path

When the loaded song proposal contains a `musical_constraints.harmonic_sequence`, the
chord pipeline SHALL use a constrained generation path instead of Markov sampling.

A new function `build_constrained_candidates` SHALL be added to `chord_pipeline.py`.
It SHALL:

1. Parse `harmonic_sequence` into an ordered list of Roman numeral tokens
   (split on whitespace; e.g. `"i iv i"` → `["i", "iv", "i"]`).
2. For each token, look up chords from the chord bank via
   `gen.get_chord_by_function(key_root, mode, token, category="triad")`,
   falling back to any category if no triad exists.
3. If any token resolves to zero chords in the bank, skip that token with a warning
   rather than failing the whole pipeline.
4. Produce multiple candidate progressions by independently sampling one chord per
   token position from the available pool, up to `num_candidates` total. Each
   candidate is a different voicing combination of the fixed sequence.
5. Score each candidate through the same composite pipeline (theory + Refractor) used
   for Markov candidates.
6. Return the top `top_k` as ranked candidate dicts in the same format as
   `build_diatonic_candidates`, with `source: "constrained"` and
   `id: "constrained_NNN"`.

When `harmonic_sequence` is present, Markov generation SHALL still run alongside the
constrained path (same `num_candidates` budget), so the human reviewer always sees
both organic Markov results and the explicitly-directed sequence. Constrained
candidates appear at the top of `review.yml` ranked by composite score.

When `harmonic_sequence` is absent, behavior is unchanged from current.

#### Scenario: Constrained generation produces candidates for "i IV i"

- **GIVEN** a song proposal with `harmonic_sequence: "i IV i"` in C minor
- **WHEN** `run_chord_pipeline` is invoked
- **THEN** the pipeline SHALL call `build_constrained_candidates` with tokens
  `["i", "IV", "i"]`
- **AND** each constrained candidate SHALL be a three-chord progression where
  chord 1 and 3 use a minor tonic chord and chord 2 uses a major IV chord
- **AND** constrained candidates SHALL appear in `review.yml` with `source: "constrained"`
- **AND** Markov candidates SHALL also appear (source: "markov") in the same file

#### Scenario: Single-chord "one chord for two minutes" case

- **GIVEN** a proposal with `harmonic_sequence: "i"`
- **WHEN** the pipeline runs
- **THEN** `build_constrained_candidates` SHALL produce candidates each containing
  exactly one chord (the minor tonic in various voicings)
- **AND** Markov generation SHALL also run and produce its normal multi-chord results
- **AND** both appear in `review.yml` — the human can promote either

#### Scenario: Unknown function token — graceful skip

- **GIVEN** a `harmonic_sequence` containing a token with no matching chords in the
  bank (e.g. `"bVII"` in a key where that degree is absent)
- **WHEN** `build_constrained_candidates` processes it
- **THEN** a warning SHALL be printed but the pipeline SHALL NOT raise
- **AND** the token SHALL be skipped; remaining tokens proceed normally

#### Scenario: harmonic_sequence absent — unchanged behaviour

- **GIVEN** a proposal with no `musical_constraints` or no `harmonic_sequence`
- **WHEN** `run_chord_pipeline` is invoked
- **THEN** it SHALL behave exactly as before this change: Markov generation only,
  no constrained candidates added

### Requirement: Constrained Candidate Labelling in review.yml

Constrained candidates SHALL be visually distinguishable in `review.yml` from Markov
and diatonic candidates so the reviewer understands their origin.

Each constrained candidate entry SHALL include:
- `source: "constrained"`
- `harmonic_sequence` field echoing the token string from the proposal
  (e.g. `harmonic_sequence: "i iv i"`)

#### Scenario: review.yml distinguishes constrained from Markov

- **GIVEN** a pipeline run with `harmonic_sequence: "i iv i"`
- **WHEN** `review.yml` is written
- **THEN** constrained entries SHALL have `source: constrained` and
  `harmonic_sequence: "i iv i"`
- **AND** Markov entries SHALL have `source: markov` and no `harmonic_sequence` field
