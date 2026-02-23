# Proposal: Add Harmonic Rhythm Generation

## Summary

Add a harmonic rhythm generation phase to the music production pipeline that varies chord durations within a section, snapped to a half-bar grid. This phase sits between drums and strums in the pipeline order: **chords → drums → harmonic rhythm → strums → bass → melody**.

Currently every chord gets exactly one bar. This phase generates candidates where chord durations are redistributed — e.g., chord 1 holds for 1.5 bars while chord 2 plays for 0.5 bars — creating more musical variety in how the harmony moves through time.

## Motivation

Uniform one-bar-per-chord progressions sound mechanical. In real music, harmonic rhythm (how quickly chords change) is a major expressive tool:
- A chord lingering for 2 bars creates tension/anticipation
- A quick half-bar chord change creates urgency
- Aligning chord changes with drum accents creates groove coherence

## Approach

1. **Read inputs**: approved chord voicings + approved drum pattern(s) for each section
2. **Extract drum accent grid**: identify strong beat positions (kick accents, snare hits) from the approved drum MIDI
3. **Generate duration candidates**: enumerate valid chord duration distributions on a half-bar grid, where chord changes snap to strong drum beats
4. **Score with ChromaticScorer**: each distribution produces different MIDI (same notes, different durations), scored primarily on temporal mode
5. **Write candidates + review.yml**: same human gate pattern as other phases
6. **Downstream**: the strum pipeline reads the approved harmonic rhythm to know how long each chord lasts, instead of assuming 1 bar per chord

## Key Design Decisions

- **Half-bar grid**: durations snap to multiples of half a bar (e.g., 0.5, 1.0, 1.5, 2.0 bars). Provides meaningful variety without excessive granularity.
- **Section length may expand or contract**: a 4-chord section doesn't have to be exactly 4 bars. It could be 3 bars (compressed) or 6 bars (expanded).
- **Drum accent alignment**: chord changes score higher when they land on strong beats in the approved drum pattern (kick accents, snare hits, group boundaries in 7/8).
- **ChromaticScorer temporal mode**: the primary chromatic signal — different chord durations change the temporal feel of the MIDI.
- **Composite scoring**: weighted blend of drum alignment score + chromatic temporal match, similar to the drum pipeline's energy + chromatic blend.

## Affected Capabilities

| Capability       | Action   | Notes                                                         |
|------------------|----------|---------------------------------------------------------------|
| harmonic-rhythm  | NEW      | Core generation, scoring, review                              |
| strum-generation | MODIFIED | Consume approved harmonic rhythm for variable chord durations |

## References

- Drum generation spec: `openspec/specs/drum-generation/spec.md`
- Strum generation spec: `openspec/specs/strum-generation/spec.md`
- ChromaticScorer: `training/chromatic_scorer.py`
- Drum pipeline: `app/generators/midi/drum_pipeline.py`
- Strum pipeline: `app/generators/midi/strum_pipeline.py`
