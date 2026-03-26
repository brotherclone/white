# Design: White Synthesis Pipeline

## Context

The White color is defined in the chromatic framework as the synthesis of all colors — it has
no single modal target but rather spans all temporal, spatial, and ontological modes. The
existing Markov chord generator produces chromatic-fit progressions for a single color;
for White the source of "chromatic truth" is the body of approved work from the sub-songs
themselves.

This design also draws on William Burroughs and Brion Gysin's cut-up technique: slice
existing text/musical material and reassemble it in a new order. The resulting work is
both derived from and estranged from its sources.

## Goals / Non-Goals

**Goals:**
- White chord candidates are built from real approved material, not Markov generation
- Key and BPM of donor material are normalised to the White song before splicing
- Candidates are still scored by Refractor and ranked — the selection mechanism is unchanged
- Lyric cut-up gives Claude sub-lyrics as explicit source material, not just a prompt
- No changes to drum, bass, or melody pipelines
- Full backward compat: non-White color songs are unaffected

**Non-Goals:**
- Automatic discovery of sub-proposals — the producer lists them explicitly in the proposal
- Structural MIDI analysis (e.g., detecting chord function from notes) — we treat donor
  bars as opaque blocks; they are transposed but not reharmonised
- Polyphonic voice separation — transposition is applied uniformly to all notes

## Decisions

### 1. Sub-proposal references as explicit paths in the song proposal

**Decision:** `sub_proposals` is a list of production directory paths, written explicitly
in the White song proposal YAML.

**Why:** Paths are unambiguous and traceable. Auto-discovery (e.g., scan thread for all
color dirs) is fragile when some colors have multiple production dirs or are not yet
complete. The producer chooses which sub-songs to include.

**Alternative considered:** Glob-style pattern matching over a thread directory. Rejected
— too implicit, hard to audit.

### 2. Transposition by semitone delta, BPM by tempo message replacement

**Decision:** Transpose donor MIDI notes by `white_root_midi - donor_root_midi` semitones.
Rescale BPM by replacing the MIDI tempo metadata message: set it to
`round(60_000_000 / white_bpm)` microseconds per beat. Do not stretch tick values.

**Why:** Tick-value stretching requires floating-point rounding across every event, which
accumulates errors at bar boundaries. Replacing the tempo message is lossless — only the
playback speed interpretation changes, not the note positions.

**Consequence:** If a donor MIDI has no explicit tempo message it defaults to 120 BPM.
The donor's BPM MUST be read from its `chords/review.yml` (not assumed) so the tempo
replacement is correct.

### 3. Bar extraction via tick arithmetic

**Decision:** Extract bars as slices: a bar starts at tick `n * ticks_per_beat *
beats_per_bar` and ends at tick `(n+1) * ticks_per_beat * beats_per_bar`. Include any
note that starts within the bar's tick range. Re-zero start times relative to bar start.

**Handling overlapping notes:** Notes that start in bar N but end in bar N+1 are truncated
at the bar boundary (duration capped to the remaining ticks in that bar). This is
acceptable for a cut-up — some note truncation at splice points is part of the aesthetic.

### 4. Cut-up candidate generation

**Decision:** A single White chord candidate is constructed as follows:
1. Draw N bars uniformly at random from the full bar pool (with replacement — the same
   bar may appear multiple times, like a riff).
2. Shuffle the N bars (the cut-up step).
3. Concatenate bars in the new order to form a complete progression MIDI.

N is the same progression length as for other colors (4 bars for 4/4, 7 bars for 7/8).
The seed controls the random draw and shuffle so candidates are reproducible.

**Alternative considered:** Markov-guided bar selection (sample bars that transition
well harmonically). Rejected for the initial implementation — the cut-up's power is in
its randomness. A future Markov-guided variant can be added as a separate flag.

### 5. Lyric cut-up: sub-lyrics as explicit source material in the Claude prompt

**Decision:** When `rainbow_color` is White, the lyric pipeline reads approved lyric
files from each sub-proposal (from `melody/candidates/lyrics_NN.txt` that were labelled
`approved` in `lyrics_review.yml`, or all candidates if no review exists). These are
concatenated and included in the Claude prompt under a `## Source Lyrics` section.
Claude is instructed to cut them up — extract phrases, recombine them, transform them —
rather than generating from the concept alone.

**Why:** Pure concept-driven generation for White would produce a generic synthesis.
Forcing Claude to work with the actual sub-lyrics produces lyrics with shared vocabulary,
themes, and cadences from the sub-songs — true synthesis.

**Edge case:** If no sub-lyrics are available (sub-proposals have no approved lyrics),
fall back to standard lyric generation with a note in the prompt that it is a White
synthesis song.

### 6. Scoring White chord candidates

**Decision:** White uses the existing Refractor scoring with the uniform distribution
target (same as Black). This is already defined in `CHROMATIC_TARGETS`. No change needed
to scoring.

**Why:** The White chromatic axes span all modes — it has no directional target. The
uniform distribution target (1/3 each) means Refractor acts as a quality / coherence
filter rather than a directional selector. Theory scoring (melody, voice leading, variety)
still ranks candidates meaningfully.

## Data Flow

```
White song proposal (key, bpm, sub_proposals: [...])
        │
        ▼
white_rebracketing.build_bar_pool(sub_proposals, white_key, white_bpm)
        │   for each sub-proposal:
        │     - load chords/review.yml → donor key, donor bpm
        │     - load chords/approved/*.mid
        │     - transpose notes by semitone delta
        │     - replace tempo message with white_bpm
        │     - extract individual bars → bar pool
        ▼
bar pool: list of (bar_midi_events, source_label)
        │
        ▼
generate N candidates:
  for i in range(num_candidates):
    bars = rng.choices(bar_pool, k=progression_length)
    shuffle(bars)
    candidate_midi = concatenate(bars)
    score(candidate_midi)
        │
        ▼
review.yml (same format as color songs)
```

## Risks / Trade-offs

- **Octave clamping after transposition**: large semitone deltas (e.g., donor in F, White
  in B — 6 semitones) may push some notes out of the standard 21–108 MIDI range. Clamp
  notes to [21, 108] with a warning.
- **Bar pool is small if few sub-proposals**: if only 2–3 sub-proposals with 1 approved
  chord each (4 bars each), the bar pool has 8–12 bars. With replacement sampling this
  is fine for 4-bar progressions, but variety is limited. Expected use case is 7–8
  sub-proposals × 3–5 approved chords each = 84–160 bars in the pool.
- **Lyric cut-up quality**: Claude is good at surface-level cut-up but may produce
  grammatically incoherent output. The prompt should explicitly ask for coherent verses
  that _borrow phrases_ rather than random word-splicing.

## Open Questions

- Should the White proposal automatically discover sub-proposals from the thread if
  `sub_proposals` is omitted? (Deferred — explicit paths for now.)
- Should the cut-up order be fully random or biased toward adjacent-bar transitions?
  (Random for now; Markov-guided variant is a follow-on.)
- Should bars retain their source label (color tag) for traceability in review.yml?
  (Yes — include as metadata in the review.yml candidate entries.)
