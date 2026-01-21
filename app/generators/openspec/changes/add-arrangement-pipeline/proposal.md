# Change: Add Full Arrangement Pipeline

## Why
Individual generators (chords, melody, bass, drums) create musical layers, but a complete song requires orchestration: song structure (intro/verse/chorus/bridge/outro), sample/loop integration, dynamics and energy flow, transitions, and final MIDI arrangement ready for human performance. The arrangement pipeline is the conductor that brings all layers together into a cohesive musical experience.

## What Changes
- Add `ArrangementPipeline` class orchestrating all generators in sequence
- Implement song structure templates (verse-chorus, AABA, through-composed)
- Add sectional variation (verse lighter, chorus heavier)
- Implement transition generation (fills, builds, breakdowns)
- Add sample/loop selection and integration
- Support for dynamics and energy mapping across song structure
- Integration with all existing generators (chord, melody, bass, drums)
- MIDI export with complete multi-track arrangement
- Agent-based creative direction (LangGraph agents critique and guide)

## Impact
- Affected specs: arrangement-pipeline (new capability)
- Affected code:
  - `app/generators/midi/arrangement/` (new directory)
  - `app/generators/midi/arrangement/pipeline.py` - main orchestration
  - `app/generators/midi/arrangement/structure.py` - song structure templates
  - `app/generators/midi/arrangement/transitions.py` - fills, builds, breakdowns
  - `app/generators/midi/arrangement/samples.py` - sample selection and integration
  - Integration with `ChordProgressionGenerator`, `MelodyGenerator`, `BassLineGenerator`, `DrumPatternGenerator`
- Dependencies: All previous generators, LangGraph (agent orchestration), sample libraries
- Complexity: Very High - orchestrates all subsystems, adds creative decision-making layer

## Design Considerations

### Song Structure Templates
Songs follow conventional structures:

**Verse-Chorus (Pop/Rock)**:
```
Intro → Verse 1 → Chorus → Verse 2 → Chorus → Bridge → Chorus → Outro
```

**AABA (Jazz Standard)**:
```
A (theme) → A (repeat) → B (bridge/contrast) → A (return)
```

**Through-Composed (Progressive/Experimental)**:
```
Section A → Section B → Section C → ... (no repetition)
```

**Verse-Prechorus-Chorus (Modern Pop)**:
```
Intro → Verse 1 → Prechorus → Chorus → Verse 2 → Prechorus → Chorus → Bridge → Chorus (2x) → Outro
```

Each section type has different characteristics:
- **Intro**: Sparse, builds anticipation (chords + light drums, no vocal)
- **Verse**: Narrative, lower energy (simpler patterns, focus on lyrics)
- **Prechorus**: Build anticipation for chorus (add energy, build-up fill)
- **Chorus**: Hook, highest energy (full arrangement, melodic climax)
- **Bridge**: Contrast, break from verse/chorus (different chords, different groove)
- **Outro**: Wind down or fade out (simplify, reduce voices)

### Sectional Variation Strategy
Each section needs appropriate orchestration:

**Verse**:
- Chord progression: Simple, supportive
- Melody: Lower tessitura, narrative rhythm
- Bass: Root-focused, minimal syncopation
- Drums: Basic groove (kick/snare/hats), no crashes

**Chorus**:
- Chord progression: Same as verse OR more intense (add 7ths, extensions)
- Melody: Higher tessitura, memorable hook
- Bass: Busier pattern, more chord tones
- Drums: Full kit, crashes, more fills

**Bridge**:
- Chord progression: Modulate OR different function (move to IV, vi, etc.)
- Melody: Different contour (ascending if verse descends)
- Bass: Walking or pedal tone
- Drums: Different groove (half-time, double-time, or breakdown)

### Transition Generation
Smooth transitions between sections require:

**Fills**:
- Pre-chorus fill: 1-2 bars of snare/tom build
- Pre-bridge fill: 1 bar of dramatic tom descent
- Turnaround fills: Every 8 bars in verse/chorus

**Build-Ups**:
- Snare rolls with crescendo
- Bass density increase (quarter → eighth → sixteenth)
- Cymbal swells
- Filter sweeps (sample-based)

**Breakdowns**:
- Drop to kick + bass only
- Silence for 1-2 beats, then crash re-entry
- Gradual voice removal (toms → snare → hats, leaving kick)

### Energy Mapping
Song energy should flow naturally:

```
Energy Level Over Time:

High |                   ╱╲     ╱╲
     |                  ╱  ╲   ╱  ╲
     |          ╱╲    ╱     ╲╱     ╲
Med  |    ╱╲  ╱  ╲  ╱               ╲╱╲
     |   ╱  ╲╱    ╲╱
Low  |  ╱                              ╲
     |─┴─────────────────────────────────┴──
      Intro V1  Ch  V2  Ch  Br  Ch  Ch Outro
```

Energy control via:
- **Instrumentation**: Intro (chords only) → Verse (+ bass + light drums) → Chorus (+ crashes + toms)
- **Dynamics**: Verse (velocity 60-80) → Chorus (velocity 90-110)
- **Rhythmic density**: Verse (quarter notes) → Chorus (eighth/sixteenth notes)
- **Register**: Verse (mid-range melody) → Chorus (high melody, low bass)

### Sample/Loop Integration
Samples and loops add texture and modernity:

**Sample Types**:
- **Ambient pads**: Sustaining synth/string textures for intros/bridges
- **Rhythmic loops**: Electronic beats, shakers, percussion
- **Sound effects**: Risers, impacts, whooshes for transitions
- **Vocal chops**: Processed vocal fragments for hooks
- **Field recordings**: Environmental sounds for texture

**Integration Strategy**:
- Samples placed in separate MIDI or audio tracks
- Triggered at specific song positions (e.g., riser before chorus)
- Volume automation (fade in/out, swell)
- Synchronized to tempo and key

### Rebracketing Marker Integration
Arrangement must honor ontological structure:
- Detect rebracketing markers from lyrics
- Ensure ALL layers (chords, melody, bass, drums, samples) emphasize markers
- Coordinate emphasis: crash + bass drop + melody peak + chord change
- Post-marker resolution across all layers

### LangGraph Agent Integration (Creative Direction)
Agents provide qualitative feedback:

**White Agent** (conceptual critic):
- "Does the bridge contrast sufficiently with the verse?"
- "Is the rebracketing marker adequately emphasized?"
- "Does the energy flow match the ontological arc?"

**Violet Agent** (musical critic):
- "Is the voice leading smooth?"
- "Does the melody sit well over the chords?"
- "Is the bass-drum pocket tight?"

**Black Agent** (creative provocateur):
- "What if the chorus drops to silence instead of building?"
- "Should the bridge be in a different time signature?"

Agents run after arrangement generation, provide feedback, trigger regeneration with adjusted parameters.

### Multi-Track MIDI Export
Final output is a complete MIDI file:
- Track 1: Chord progression (piano or pad)
- Track 2: Melody + lyrics (MIDI lyric meta-events)
- Track 3: Bass line (fingered or slap bass patch)
- Track 4: Drums (General MIDI channel 10)
- Track 5+: Samples/loops (if MIDI-triggerable)

Each track has:
- Appropriate MIDI patch/channel
- Dynamics (velocity, expression)
- Timing (quantized or humanized)
- Metadata (tempo, key, time signature)

Human musicians receive this MIDI, perform real instruments, record, mix, master → final "suck on that, Spotify" quality track.

## Example Arrangement Pipeline Flow

```
Input:
  - Concept text: "The room is not a container" (ontological rebracketing)
  - Lyrics: "Beautiful ladies they're solving mysteries / The room [(is not)] a container"
  - Song structure: Verse-Chorus
  - Style: "Indie rock"
  - Chromatic mode: "Violet" (introspective, complex harmony)

Pipeline Execution:

1. Structure Planning:
   - Intro (4 bars)
   - Verse 1 (8 bars)
   - Chorus (8 bars)
   - Verse 2 (8 bars)
   - Chorus (8 bars)
   - Bridge (8 bars)
   - Chorus (8 bars, final)
   - Outro (4 bars)

2. Generate Chord Progressions:
   - Intro: Fmaj7 → Am7 (2 bars each, ambient)
   - Verse: Fmaj7 → Am7 → Cmaj7 → G7 (repeating)
   - Chorus: Cmaj7 → Em7 → Fmaj7 → G7 (more resolved)
   - Bridge: Dm7 → Em7 → Am7 → Cmaj7 (different function)

3. Generate Melody (over Verse/Chorus):
   - Verse melody: Lower tessitura (C4-G4), narrative rhythm
   - Chorus melody: Higher tessitura (E4-C5), hook on "is not"
   - Rebracketing marker "is not" → melodic peak + duration extension

4. Generate Bass:
   - Intro: Roots on whole notes (F → A)
   - Verse: Root-fifth pattern, quarter notes
   - Chorus: Walking bass with chromatic approaches
   - Bridge: Pedal tone on D (while chords change above)

5. Generate Drums:
   - Intro: No drums (just chords)
   - Verse: Basic rock groove (kick 1,3 / snare 2,4 / hats 8ths)
   - Chorus: Full kit with crash on beat 1, busier fills
   - Bridge: Half-time feel (kick on 1, snare on 3)
   - Transition fills: Tom fill before each chorus

6. Add Samples:
   - Intro: Ambient pad (sustained Fmaj7)
   - Pre-chorus: Riser (filter sweep, 2 bars)
   - Marker "is not": Impact sound effect (cymbal crash + bass drop)

7. Coordinate Rebracketing Emphasis:
   - "is not" at Verse bar 6, beat 3
   - Melody: Peak note (G5), extended duration
   - Bass: Drop to E1 (low register)
   - Drums: Crash cymbal + fill before marker
   - Chords: Cmaj7 (harmonic stability)

8. Agent Critique:
   - White Agent: "The marker emphasis is strong. Good use of low bass drop."
   - Violet Agent: "The melody sits well. Consider adding 7th to chorus chords for more color."
   - Pipeline regenerates chorus chords with 7ths based on feedback.

9. Export Multi-Track MIDI:
   - Track 1 (Chords): MIDI patch 0 (Acoustic Grand Piano)
   - Track 2 (Melody): MIDI patch 53 (Voice Oohs) + lyric meta-events
   - Track 3 (Bass): MIDI patch 33 (Fingered Bass)
   - Track 4 (Drums): Channel 10, General MIDI drum map
   - Tempo: 120 BPM, Key: F Major, Time Signature: 4/4

Output:
  - Complete MIDI file: "the_room_is_not_a_container_arrangement.mid"
  - Ready for human musicians to perform and record
```

## Scoring Criteria

Arrangements will be scored on:

1. **Structural Coherence** (0-1):
   - Sections follow logical order: +1.0
   - Transitions are smooth: +0.9
   - Energy flow is natural: +0.9
   - Chaotic or abrupt structure: -0.6

2. **Sectional Contrast** (0-1):
   - Verse/chorus differ appropriately: +1.0
   - Bridge provides contrast: +0.9
   - All sections sound identical: -0.7

3. **Layer Integration** (0-1):
   - Chords, melody, bass, drums work together: +1.0
   - Bass locks with drums: +0.9
   - Melody fits harmony: +0.9
   - Layers clash or compete: -0.8

4. **Rebracketing Emphasis** (0-1):
   - All layers emphasize markers: +1.0
   - Coordinated emphasis (crash + bass + melody peak): +1.0
   - Markers not emphasized: -0.8

5. **Dynamics and Energy** (0-1):
   - Energy builds and releases naturally: +1.0
   - Intro/outro appropriate: +0.9
   - Flat energy throughout: -0.6

6. **Performability** (0-1):
   - All parts humanly performable: +1.0
   - MIDI exports cleanly: +1.0
   - Parts too complex or impossible: -0.8

## Implementation Phases

**Phase 1: Structure Templating** (Foundation)
- Define song structure types (verse-chorus, AABA, etc.)
- Map sections to bar counts
- Create sectional variation rules

**Phase 2: Sequential Generation** (Basic Pipeline)
- Generate chords for entire song structure
- Generate melody for vocal sections
- Generate bass for all sections
- Generate drums for all sections

**Phase 3: Transition Generation** (Musicality)
- Detect section boundaries
- Generate fills, builds, breakdowns
- Add crashes and emphasis

**Phase 4: Rebracketing Coordination** (Ontological)
- Detect markers from lyrics
- Ensure all layers emphasize markers
- Coordinate timing and intensity

**Phase 5: Sample Integration** (Texture)
- Select samples for intros, builds, transitions
- Place samples at song positions
- Add volume automation

**Phase 6: Agent Critique Loop** (Quality)
- Run White/Violet/Black agents
- Collect feedback
- Regenerate with adjusted parameters
- Iterate until agents approve

**Phase 7: Multi-Track Export** (Deliverable)
- Combine all layers into single MIDI file
- Assign tracks, patches, channels
- Add metadata (tempo, key, time signature)
- Export human-performable arrangement
