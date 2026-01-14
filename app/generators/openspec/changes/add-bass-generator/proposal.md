# Change: Add Bass Line Generator

## Why
Bass lines provide the rhythmic and harmonic foundation connecting chords, melody, and drums. The bass must lock with the kick drum for groove, outline harmonic movement, support (not compete with) the melody, and emphasize ontological pivot points through register, rhythm, and harmonic choices.

## What Changes
- Add `BassLineGenerator` class for generating bass patterns over chord progressions
- Implement chord-tone emphasis (roots, fifths, thirds on strong beats)
- Add walking bass patterns (stepwise chromatic approaches between chords)
- Implement groove patterns (syncopation, anticipation, ghost notes)
- Add rebracketing marker emphasis (low register, rhythmic density, pedal tones)
- Support for bass line scoring (harmonic fit, rhythmic coherence, playability)
- Integration with chord progressions and melody (avoid clashing intervals)
- MIDI export with bass track separate from other instruments

## Impact
- Affected specs: bass-generator (new capability)
- Affected code:
  - `app/generators/midi/bass/` (new directory)
  - `app/generators/midi/bass/generator.py` - main bass line generation
  - `app/generators/midi/bass/patterns.py` - groove and walking bass patterns
  - `app/generators/midi/bass/scoring.py` - bass quality scoring
  - Integration with existing `ChordProgressionGenerator` and `MelodyGenerator`
- Dependencies: mido (MIDI), integration with chord/melody generators
- Complexity: Medium - bass has fewer melodic constraints than vocals, but strict harmonic/rhythmic requirements

## Design Considerations

### Bass-First Harmonic Function
Unlike melody (which follows lyrics), bass **defines** harmonic progression:
1. Accept chord progression as input
2. Extract harmonic rhythm (when chords change)
3. Place root notes on chord changes
4. Fill between chord tones with passing/approach tones
5. Add rhythmic variation within harmonic constraints

### Rebracketing Marker Emphasis
Ontological pivot points should be harmonically grounded:
- **Low register emphasis** - Drop to lowest comfortable note (E1-E2 range)
- **Pedal tones** - Sustain root note through marker for stability
- **Rhythmic shift** - Change pattern density or syncopation
- **Harmonic tension** - Use tritone substitutions or chromatic approaches before markers

### Groove Integration
Bass must lock with drums (future integration):
- Align with kick drum patterns (strong beat coincidence)
- Support snare backbeat (anticipations, ghost notes)
- Create syncopated pocket (16th note subdivisions)
- Leave space for other instruments (rhythmic silence is groove)

### Playability Constraints
Generated bass lines must be performable by human bassists:
- Reasonable range (typically E1-G3, 4-string standard tuning)
- Avoid impossible position shifts (max 5-fret jumps per beat)
- Respect string/fret limitations (open strings, chromatic runs)
- Natural fingering patterns (alternate fingers, hammer-ons/pull-offs marked)

### Harmonic Movement Patterns

**Walking Bass** (jazz/funk):
- Stepwise motion between chord tones
- Chromatic approach tones (half-step below next root)
- Target chord tones on downbeats

**Pedal Tones** (rock/ambient):
- Sustain root while chords change above
- Creates harmonic tension/release

**Arpeggiated** (pop/electronic):
- Cycle through chord tones (R-5-8-5 patterns)
- Rhythmic consistency, melodic simplicity

**Syncopated** (funk/R&B):
- Anticipate chord changes (play root on "and" of 4)
- Ghost notes, muted strings, rhythmic complexity

### Integration with Melody (Avoiding Clashes)
Bass must support melody without creating dissonance:
- When melody is on chord tone, bass can use any chord tone or passing tone
- When melody is on non-chord tone (suspension, passing), bass stays on chord tones
- Avoid minor 2nd or major 7th intervals between bass and melody
- Counterpoint: when melody ascends, bass descends (and vice versa)

### Integration with Training Models (Future)
Eventually, trained models can guide bass:
- Chromatic style model suggests groove character (e.g., Black = heavy pedal tones, Violet = walking bass)
- Temporal sequence model suggests when to increase/decrease rhythmic density
- Rebracketing classifier validates ontological emphasis

But for now, use music theory + pattern templates + scoring.

## Example Bass Line Generation Flow

```
Input:
  - Chord progression: [Cmaj7, Am7, Fmaj7, G7] (4 bars, 4/4)
  - Melody: [C5 E5 G5 E5, A4 C5 E5 C5, F4 A4 C5 A4, G4 B4 D5 B4]
  - Style: "walking bass"

Generation:
  1. Extract chord tones:
     - Cmaj7: [C, E, G, B]
     - Am7: [A, C, E, G]
     - Fmaj7: [F, A, C, E]
     - G7: [G, B, D, F]

  2. Place roots on downbeats:
     - Bar 1 beat 1: C2
     - Bar 2 beat 1: A1
     - Bar 3 beat 1: F1
     - Bar 4 beat 1: G1

  3. Fill with walking pattern (quarter notes):
     - Bar 1: C2, E2, G2, A2 (chromatic approach to Am)
     - Bar 2: A1, C2, E2, F2 (approach to Fmaj)
     - Bar 3: F1, A1, C2, E2 (approach to G7)
     - Bar 4: G1, B1, D2, F2

  4. Check melody intervals:
     - Bar 1 beat 1: Melody C5, Bass C2 → octave (perfect, good)
     - Bar 1 beat 2: Melody E5, Bass E2 → octave (good)
     - No clashes detected

  5. Score:
     - Harmonic fit: 0.95 (all chord tones or chromatic approaches)
     - Rhythmic coherence: 0.85 (consistent quarter notes, good for walking)
     - Playability: 0.90 (all within comfortable range, no big jumps)
     - Melody support: 0.92 (no clashes, good counterpoint)
     - Total: 0.905

Output:
  - MIDI track with bass notes
  - Pattern metadata: "walking_bass_quarter_notes"
```

## Scoring Criteria

Bass lines will be scored on:

1. **Harmonic Fit** (0-1):
   - Chord tones on strong beats: +1.0
   - Passing tones on weak beats: +0.8
   - Chromatic approaches before chord changes: +0.9
   - Non-harmonic tones on strong beats: -0.5

2. **Rhythmic Coherence** (0-1):
   - Consistent subdivision pattern: +1.0
   - Syncopation aligned with style: +0.9
   - Random/chaotic rhythm: -0.5

3. **Playability** (0-1):
   - All notes in range (E1-G3): +1.0
   - Max 5-fret position shifts per beat: +1.0
   - Impossible fingerings: -0.8

4. **Melody Support** (0-1):
   - No harsh intervals with melody: +1.0
   - Good counterpoint (contrary motion): +0.9
   - Interval clashes (m2, M7): -0.7

5. **Groove** (0-1):
   - Syncopation appropriate for style: +1.0
   - Ghost notes/mutes for texture: +0.8
   - Stiff/robotic feel: -0.5

6. **Variety** (0-1):
   - Mix of roots, fifths, other chord tones: +1.0
   - Rhythmic variation across bars: +0.9
   - Too repetitive: -0.5

## Implementation Phases

**Phase 1: Chord-Following Bass** (Basic)
- Root notes on chord changes
- Simple quarter or half note rhythms
- Stay in safe range (E1-E2)

**Phase 2: Walking Bass** (Intermediate)
- Stepwise motion between chords
- Chromatic approach tones
- Consistent quarter note pulse

**Phase 3: Groove Patterns** (Advanced)
- Syncopation templates (funk, R&B, rock)
- Ghost notes and articulations
- Integration with drum patterns

**Phase 4: Rebracketing Emphasis** (Ontological)
- Detect markers from lyrics
- Pedal tones, low register, rhythmic shifts
- Harmonic tension/release around markers

**Phase 5: Brute-Force Search** (Quality)
- Generate N candidates per progression
- Score on all criteria
- Return top-K with score breakdowns
