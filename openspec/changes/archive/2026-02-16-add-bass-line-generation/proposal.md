# Proposal: Add Bass Line Generation

## Summary

Add a bass line generation phase to the music production pipeline. Bass lines are generated from approved chords, harmonic rhythm, and drum patterns, scored with a theory + chromatic composite, and gated through the same human review workflow as all other phases.

Pipeline position: chords → drums → harmonic rhythm → strums → **bass** → melody

## Motivation

Bass is the bridge between rhythm and harmony. It locks to the kick drum rhythmically while outlining the chord progression harmonically. Without bass, the drum and chord layers exist independently — bass is what makes them sound like a cohesive arrangement.

The bass phase reads from more upstream phases than any prior instrument:
- **Chords**: root notes and harmonic context
- **Harmonic rhythm**: when chords change (variable durations)
- **Drums**: kick pattern for rhythmic alignment

This makes it the ideal next step — it integrates everything built so far.

## Approach

1. **Template library** (`bass_patterns.py`): Structured data templates organized by style (root, walking, pedal, arpeggiated, octave, syncopated) and energy level (low/medium/high). Each template defines note-selection rules and rhythmic positions relative to the bar, similar to `drum_patterns.py`. Genre-family mapping reuses the same families as drums.

2. **Bass pipeline** (`bass_pipeline.py`): Reads approved chords + harmonic rhythm + drums. For each section:
   - Extracts chord roots and available chord tones from approved MIDI
   - Reads kick pattern from approved drums for rhythmic alignment
   - Applies each applicable bass template, generating MIDI on channel 0 in the bass register (MIDI octaves 1-3, roughly notes 24-60)
   - Scores with composite: theory (root adherence, kick alignment, voice leading) + chromatic

3. **Composite scoring**: Weighted blend of theory score (30%) and ChromaticScorer (70%), same weights as other phases.

4. **Output**: `<song>/bass/candidates/` + `review.yml` → human labels → promote to `approved/`

## Key Design Decisions

- **Note selection from chord tones**: Bass templates specify *which* chord tone to play (root, 5th, 3rd, etc.) rather than absolute pitches. The pipeline resolves these to actual MIDI notes from the approved chord voicings. This keeps templates reusable across any key.
- **Register**: All bass notes clamped to MIDI octaves 1-3 (notes 24-60). If a chord root is above this range, it's transposed down by octaves.
- **Kick alignment scoring**: Similar to how harmonic rhythm scores against drum accents, bass scores higher when note onsets coincide with kick drum hits. This is the primary theory component alongside root adherence.
- **Harmonic rhythm awareness**: Bass note durations follow the approved harmonic rhythm distribution — a chord lasting 2 bars gets 2 bars of the bass pattern. Falls back to 1 bar per chord if no harmonic rhythm exists.
- **No genre-family filtering** (unlike drums): All bass templates are applicable to all genres. The ChromaticScorer and theory scoring will naturally select the best fit. This keeps the template count manageable.
- **Single MIDI channel**: Bass on channel 0 (standard), no velocity dynamics beyond the template's accent/normal/ghost levels.

## Affected Capabilities

| Capability      | Action | Notes                                              |
|-----------------|--------|----------------------------------------------------|
| bass-generation | NEW    | Template library, pipeline, scoring, review output |

## References

- Drum generation spec: `openspec/specs/drum-generation/spec.md`
- Strum generation spec: `openspec/specs/strum-generation/spec.md`
- Harmonic rhythm pipeline: `app/generators/midi/harmonic_rhythm_pipeline.py`
- Drum pipeline: `app/generators/midi/drum_pipeline.py`
- Strum pipeline: `app/generators/midi/strum_pipeline.py`
- ChromaticScorer: `training/chromatic_scorer.py`
- Chord pipeline (shared utilities): `app/generators/midi/chord_pipeline.py`
- Promote tool: `app/generators/midi/promote_chords.py`
