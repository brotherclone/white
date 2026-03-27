## Context
Four-part writing traditionally requires resolving dissonances, avoiding parallels, and
keeping voices within singable ranges. The pipeline must do this automatically with
enough musical credibility to be a useful starting point for human refinement.

## Goals / Non-Goals
- Goals: generate plausible SATB (or brass/string quartet) counterpoint from soprano; keep
  voices independent (no doubling for more than 2 consecutive beats); enforce range limits
- Non-Goals: full academic counterpoint enforcement (parallel 5ths tolerated in non-adjacent
  voices at low energy); real-time generation; expressive articulation (handled in Logic)

## Decisions
- **Interval-offset approach**: rather than independent voice leading, express alto/tenor/bass-voice
  as signed semitone offsets from the soprano note at each beat. Simple, fast, testable.
  Alternatives: full voice-leading search (complex, slow), rule-based grammar (more flexible
  but harder to validate).
- **Parallel-5th detection**: compare consecutive soprano→voice intervals; flag when two
  consecutive note-pairs produce the same interval (P5 = 7, P8 = 12). Re-roll by shifting
  the offending voice by ±1 semitone.
- **Multi-channel MIDI**: single file, 4 channels. Simpler than 4 files for Logic import
  (one drag). Each channel maps to a Logic track via channel strip filter.

## Risks / Trade-offs
- Interval-offset approach can produce awkward leaps if soprano has large interval jumps.
  Mitigation: cap per-beat offset change to ≤4 semitones.
- Voice crossing (alto below tenor) possible. Mitigation: post-check, swap if crossing detected.

## Open Questions
- Should bass-voice shadow the bass-line phase or be fully independent?
  (Current: independent — bass-line is rhythmic/chord-tone, bass-voice is melodic/linear)
