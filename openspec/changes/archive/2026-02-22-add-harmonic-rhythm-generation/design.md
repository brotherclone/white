# Design: Harmonic Rhythm Generation

## Pipeline Position

```
chords → drums → HARMONIC RHYTHM → strums → bass → melody
         ↓              ↑
    approved drums ─────┘
```

The harmonic rhythm phase requires both approved chords (what notes) and approved drums (where accents fall). It produces a duration map that the strum pipeline consumes.

## Half-Bar Grid Model

All chord durations are expressed in half-bar units. For a 4/4 bar at 480 tpb, one half-bar = 2 beats = 960 ticks. For 7/8 at 480 tpb, one half-bar = 1.75 beats = 840 ticks.

**Minimum chord duration**: 0.5 bars (one grid unit)
**Maximum chord duration**: uncapped, but practically limited by the number of chords

### Duration Distribution Generation

Given N chords, generate all valid distributions where each chord gets at least 0.5 bars and durations are multiples of 0.5 bars.

Example with 4 chords, targeting ~4 bars total:
```
[1.0, 1.0, 1.0, 1.0]  →  4.0 bars (uniform, baseline)
[1.5, 0.5, 1.0, 1.0]  →  4.0 bars (chord 1 lingers)
[1.0, 1.0, 0.5, 1.5]  →  4.0 bars (chord 4 lingers)
[2.0, 0.5, 0.5, 1.0]  →  4.0 bars (chord 1 dominates)
[1.5, 1.0, 1.0, 1.5]  →  5.0 bars (expanded)
[0.5, 1.0, 0.5, 1.0]  →  3.0 bars (contracted)
```

### Expansion/Contraction Bounds

To keep sections musically reasonable:
- **Minimum total**: `N * 0.5` bars (every chord at minimum duration)
- **Maximum total**: `N * 2.0` bars (every chord at double duration)
- **Step**: 0.5 bars per chord, so total section lengths vary in 0.5-bar increments

The candidate count grows combinatorially. For practical generation, we enumerate within bounds and cap at ~200 candidates before scoring.

## Drum Accent Extraction

From an approved drum MIDI file, extract a list of "strong beat" positions within one bar:

1. Parse the MIDI for note-on events
2. Identify accented hits (velocity >= VELOCITY["accent"] threshold, e.g., 100+)
3. Quantize to the half-bar grid: for each half-bar boundary, check if an accent falls within a tolerance window (e.g., ± 1/8 note)
4. Produce an **accent mask** — a list of half-bar positions that are "strong"

For a 7/8 bar with 3+2+2 grouping:
- Half-bar boundaries at beats 0.0 and 1.75
- Kick accent at 0.0 → strong at position 0.0 (half-bar 0)
- Snare accent at 1.5 → near half-bar boundary 1.75 → strong at position 1.75 (half-bar 1)

For a 4/4 bar:
- Half-bar boundaries at beats 0.0 and 2.0
- Kick accent at 0.0 → strong at half-bar 0
- Snare accent at 2.0 → strong at half-bar 1

## Scoring Model

### Drum Alignment Score (0.0–1.0)

For a given chord duration distribution, compute what fraction of chord onsets land on strong drum beats:

```
alignment = (chord onsets landing on strong beats) / (total chord onsets)
```

The first chord always starts at beat 0 (always strong), so alignment starts with a floor of `1/N`.

### Chromatic Temporal Score

Generate MIDI bytes for the distribution (chords with variable durations) and score with `ChromaticScorer.score()`. Extract the temporal mode match against the color's chromatic target.

### Composite Score

```
composite = 0.3 * drum_alignment + 0.7 * chromatic_temporal_match
```

Same weighting philosophy as drums (30% structural, 70% chromatic).

## MIDI Generation

To produce scoreable MIDI from a duration distribution:

1. For each chord in the progression, write its voicing notes
2. Duration of each chord = `distribution[i] * bar_ticks`
3. Notes sustain for the full chord duration (whole-note style per chord)
4. Set tempo from song proposal BPM

This produces a simple block-chord MIDI — the strum pipeline later applies rhythm patterns to the approved distribution.

## Output Structure

```
<song>/harmonic_rhythm/
  candidates/          # MIDI files for each distribution
    hr_001.mid
    hr_002.mid
    ...
  approved/            # promoted winners
    <label>.mid
  review.yml           # human annotation
```

## Strum Pipeline Modification

The strum pipeline currently assumes 1 bar per chord in `strum_to_midi_bytes()`. After this change:

1. If `harmonic_rhythm/approved/` exists for the section, read the approved duration map
2. Each chord gets its approved duration instead of a uniform bar
3. Strum patterns scale to fill the chord's actual duration (a 1.5-bar chord gets 1.5 bars of the pattern, repeating the pattern for longer durations or truncating for shorter)
4. If no harmonic rhythm is approved, fall back to uniform 1-bar-per-chord (backward compatible)

## Candidate Count Management

With 4 chords and a range of 2.0–8.0 total bars in 0.5 steps, the combinatorial space is manageable but grows with chord count. Strategy:

1. Enumerate all distributions within bounds
2. If count > 200, randomly sample 200 (seeded)
3. Score all candidates
4. Present top-k in review (default k=20)
