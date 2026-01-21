# Change: Add Drum Pattern Generator

## Why
Drums provide the rhythmic backbone, defining groove, tempo, and energy. The drum generator must lock with bass for pocket, support harmonic rhythm, provide dynamic contrast between sections, and emphasize rebracketing markers through fills, cymbal crashes, and rhythmic disruption.

## What Changes
- Add `DrumPatternGenerator` class for generating drum patterns synchronized with chord/bass/melody
- Implement multi-voice drum patterns (kick, snare, hi-hat, toms, cymbals, percussion)
- Add groove templates for different styles (rock, funk, jazz, electronic, hip-hop)
- Implement rebracketing marker emphasis (fills, crashes, rhythmic breaks, dynamic shifts)
- Support for dynamic variation (ghost notes, accents, velocity curves)
- Add drum pattern scoring (groove, variety, playability, bass synchronization)
- Integration with bass patterns for rhythmic pocket
- MIDI export with drum map (General MIDI percussion standard)

## Impact
- Affected specs: drum-generator (new capability)
- Affected code:
  - `app/generators/midi/drums/` (new directory)
  - `app/generators/midi/drums/generator.py` - main drum pattern generation
  - `app/generators/midi/drums/patterns.py` - groove templates and fills
  - `app/generators/midi/drums/kit.py` - drum kit voice definitions
  - `app/generators/midi/drums/scoring.py` - drum quality scoring
  - Integration with `ChordProgressionGenerator`, `BassLineGenerator`
- Dependencies: mido (MIDI), integration with chord/bass/melody generators
- Complexity: High - drums are multi-voice, highly stylistic, and define feel

## Design Considerations

### Multi-Voice Drum Programming
Unlike monophonic instruments (bass, melody), drums are **polyphonic** with independent voices:
1. **Kick drum** - Harmonic foundation, locks with bass roots
2. **Snare drum** - Backbeat (beats 2 and 4 in 4/4), fills
3. **Hi-hat** - Subdivision (8ths, 16ths), open/closed articulation
4. **Ride cymbal** - Alternative to hi-hat, jazz/rock variations
5. **Crash cymbal** - Sectional emphasis, rebracketing markers
6. **Toms** - Fills, transitions, dramatic moments
7. **Percussion** - Shakers, tambourine, cowbell (style-dependent)

Each voice has independent rhythm, velocity, articulation.

### Rebracketing Marker Emphasis
Ontological pivot points should be rhythmically highlighted:
- **Crash cymbal** - Mark marker with cymbal crash
- **Fill** - Tom/snare fill leading into marker
- **Rhythmic break** - Drop out before marker, re-enter with emphasis
- **Dynamic surge** - Increase velocity/density around marker
- **Pattern disruption** - Change groove pattern at marker

### Bass-Drum Synchronization
Drums must lock with bass for rhythmic pocket:
- Kick drum aligns with bass root notes (especially on beat 1)
- Ghost notes on kick/snare fill rhythmic gaps
- Syncopated kicks match bass anticipations
- Hi-hat provides subdivision context for bass groove

### Groove Templates
Different musical styles have canonical drum patterns:

**Rock**:
- Kick: 1, 3 (and 4 sometimes)
- Snare: 2, 4 (backbeat)
- Hi-hat: Straight 8ths or 16ths

**Funk**:
- Kick: Syncopated (1, "and of 2", 3, "and of 4")
- Snare: 2, 4 with ghost notes on 16ths
- Hi-hat: 16th notes with accents on syncopations

**Jazz**:
- Kick: Sparse, conversational (on "and" beats)
- Snare: Ghost notes, rim shots
- Ride: Swing 8ths or "ding-ding-a-ding" pattern

**Electronic/Hip-Hop**:
- Kick: Four-on-the-floor (every beat) or trap patterns (double-time hi-hats)
- Snare: Beat 3 (often clap or electronic snare)
- Hi-hat: 8th or 16th triplets, trap rolls

**Breakbeat**:
- Classic drum breaks (Amen, Funky Drummer, Apache)
- Syncopated kicks, ghost snares, complex hi-hat patterns

### Playability Constraints
Generated drum patterns must be performable by human drummers:
- Max 2 hands + 1-2 feet simultaneously (no 6-limb polyphony)
- Physically possible coordination (e.g., can't play hi-hat and ride at same time with same hand)
- Reasonable tempos for subdivision (no 32nd notes at 180 BPM)
- Recovery time after fills (hands return to groove)

### Dynamic Variation
Drums are highly expressive through dynamics:
- **Ghost notes** - Very low velocity (20-40), textural
- **Normal hits** - Medium velocity (60-90), groove
- **Accents** - High velocity (100-127), emphasis
- **Crescendo** - Gradual velocity increase over bars
- **Diminuendo** - Gradual velocity decrease

### Fills and Transitions
Drums signal sectional changes and build anticipation:
- **Pre-chorus fill** - Build into chorus (tom roll, snare roll)
- **Turnaround fill** - End of 8-bar phrase (snare/tom combination)
- **Crash cymbal** - Emphasize downbeat after fill
- **Breakdown** - Stop or simplify drums before re-entry

### Integration with Training Models (Future)
Eventually, trained models can guide drums:
- Chromatic style model suggests groove character (e.g., Red = heavy backbeat, Yellow = light, sparse jazz)
- Temporal sequence model suggests when to add fills or change patterns
- Rebracketing classifier validates ontological emphasis

But for now, use music theory + pattern templates + scoring.

## Example Drum Pattern Generation Flow

```
Input:
  - Chord progression: [Cmaj7, Am7, Fmaj7, G7] (4 bars, 4/4, 120 BPM)
  - Bass pattern: Root notes on 1, syncopation on "and of 4"
  - Style: "funk"

Generation:
  1. Select groove template for funk:
     - Kick: 1, "and of 2", "and of 4"
     - Snare: 2, 4 (backbeat) + ghost notes on 16ths
     - Hi-hat: Closed 16ths with accents on syncopations

  2. Align kick with bass:
     - Bass plays root on beat 1 → Kick on beat 1
     - Bass anticipates on "and of 4" → Kick on "and of 4"

  3. Add dynamic variation:
     - Snare ghost notes: velocity 30-40
     - Normal snare hits: velocity 90
     - Kick: velocity 100
     - Hi-hat: velocity 60-80

  4. Generate fill at bar 4 (end of phrase):
     - Beats 1-3: Continue groove
     - Beat 4: Snare roll (16th notes, crescendo 70→110)
     - Downbeat of bar 5 (next section): Crash cymbal

  5. Score:
     - Groove: 0.92 (tight pocket, funky syncopation)
     - Bass sync: 0.95 (kick perfectly aligned with bass)
     - Variety: 0.85 (good mix of hits, some ghost notes)
     - Playability: 0.90 (humanly performable)
     - Total: 0.905

Output:
  - MIDI track with General MIDI drum map
  - Kick: Note 36, Snare: Note 38, Hi-hat closed: Note 42, Crash: Note 49
  - Velocity and timing data
```

## Scoring Criteria

Drum patterns will be scored on:

1. **Groove** (0-1):
   - Consistent pocket (kick/snare align with strong/weak beats): +1.0
   - Syncopation appropriate for style: +0.9
   - Stiff/robotic feel: -0.5

2. **Bass Synchronization** (0-1):
   - Kick aligns with bass roots on beat 1: +1.0
   - Kick matches bass anticipations: +0.9
   - Kick/bass conflict (clashes): -0.7

3. **Variety** (0-1):
   - Mix of voices (kick, snare, hats, crashes): +1.0
   - Dynamic variation (ghost notes, accents): +0.9
   - Too repetitive (exact loop for entire song): -0.5

4. **Playability** (0-1):
   - Max 4 simultaneous voices (2 hands, 2 feet): +1.0
   - Physically possible coordination: +1.0
   - Impossible limb combinations: -0.8

5. **Fills** (0-1):
   - Fills at phrase boundaries: +1.0
   - Smooth return to groove after fill: +0.9
   - Missing or awkward fills: -0.5

6. **Dynamic Expression** (0-1):
   - Ghost notes, accents, crescendos: +1.0
   - Velocity variation appropriate for style: +0.9
   - Flat dynamics (all hits same velocity): -0.6

## Implementation Phases

**Phase 1: Basic Backbeat** (Simple)
- Kick on beats 1 and 3
- Snare on beats 2 and 4
- Hi-hat eighth notes
- Single velocity, no variation

**Phase 2: Groove Templates** (Intermediate)
- Style-specific patterns (rock, funk, jazz)
- Hi-hat/ride variations
- Basic syncopation

**Phase 3: Dynamic Variation** (Advanced)
- Ghost notes, accents, velocity curves
- Open/closed hi-hat articulation
- Crescendo/diminuendo

**Phase 4: Fills and Transitions** (Musical)
- Generate fills at phrase boundaries
- Crash cymbals on sectional changes
- Tom patterns, snare rolls

**Phase 5: Rebracketing Emphasis** (Ontological)
- Detect markers from lyrics
- Fills, crashes, breaks at markers
- Dynamic surges, pattern disruption

**Phase 6: Bass Integration** (Pocket)
- Synchronize kick with bass roots
- Match syncopations
- Create rhythmic pocket together

**Phase 7: Brute-Force Search** (Quality)
- Generate N candidates per progression
- Score on all criteria
- Return top-K with score breakdowns
