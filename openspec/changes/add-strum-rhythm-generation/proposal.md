# Change: Add Strum/Rhythm Generation for Approved Chords

## Why

The chord pipeline generates progressions as whole-note blocks (one chord per bar). While this is correct for harmonic selection and scoring, real music plays chords with rhythmic patterns — half notes, quarter notes, syncopated hits, arpeggiated figures. The approved chords sound like a harmonic sketch; they need rhythmic life before they're useful as production material.

This is a distinct musical decision from chord selection. The *what* (which chords) is already decided and approved. The *how* (rhythmic pattern, note density, voicing articulation) is a new creative layer that deserves its own generate-review cycle.

## What Changes

- New **strum pattern generator** that takes approved chord MIDI files and produces rhythmic variations — same harmony, different time feel
- Strum patterns include: whole notes (existing), half notes, quarter notes, eighth notes, syncopated patterns, and arpeggiated figures
- Multiple pattern candidates per approved chord, presented for human review using the existing review YAML format
- Respects the song proposal's time signature and BPM
- Operates on the `approved/` chords directory — downstream of chord approval, upstream of drums

## Impact

- Affected specs: `production-review` (reuses existing review/promote pattern)
- New specs: `strum-generation`
- Affected code:
  - New `app/generators/midi/strum_pipeline.py` — main orchestrator
  - Modifies `app/generators/midi/chord_pipeline.py:progression_to_midi_bytes()` only if shared rhythm utilities are extracted (likely not needed)
  - Reuses `app/generators/midi/promote_chords.py` as-is for strum approval
- No changes to ChromaticScorer or chord generation

## Architecture Overview

```
Approved Chords (from chord pipeline)
    │
    └── verse.mid, chorus.mid, bridge.mid (whole-note blocks)
         │
         ▼
    Strum Generation
         │  1. Parse approved MIDI → extract chord voicings + durations
         │  2. Apply rhythm patterns (half, quarter, eighth, syncopated, arpeggiated)
         │  3. Generate N variations per approved chord per pattern type
         │  4. Write candidates + review.yml
         │
         ▼
    Review File (YAML)
         │  Human listens to strummed MIDI renders
         │  Labels: verse-halves, verse-quarters, chorus-driving, etc.
         │  Status: approved / rejected
         │
         ▼
    Approved Strums
         │  Ready for drum generation phase (rhythmic context now available)
```

## Non-Goals (this iteration)

- Velocity dynamics (all notes same velocity for now; dynamics are a mixing concern)
- Swing/groove quantization (future enhancement)
- Per-note articulation (staccato vs legato — defer to DAW)
- Multi-track chord voicing (splitting chord across left/right hand — defer)
- ChromaticScorer re-scoring of strummed versions (harmony unchanged, score unchanged)
