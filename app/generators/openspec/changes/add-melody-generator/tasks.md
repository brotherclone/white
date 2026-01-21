# Implementation Tasks

## 1. Lyrics Processing
- [ ] 1.1 Create `LyricsParser` class for syllable extraction
- [ ] 1.2 Implement stress pattern detection (stressed vs unstressed syllables)
- [ ] 1.3 Add rebracketing marker detection in lyrics
- [ ] 1.4 Implement phrase boundary detection (commas, periods, line breaks)
- [ ] 1.5 Add syllable-to-beat alignment

## 2. Rhythm Generation
- [ ] 2.1 Create `RhythmGenerator` for lyric-based rhythm patterns
- [ ] 2.2 Implement prosody-to-rhythm conversion (stressed = longer/downbeat)
- [ ] 2.3 Add common rhythm patterns database (quarter notes, eighths, syncopation)
- [ ] 2.4 Implement rhythm scoring (groove, variety, predictability)
- [ ] 2.5 Add time signature support (4/4, 3/4, 6/8, etc.)

## 3. Melodic Contour Generation
- [ ] 3.1 Create `MelodyGenerator` class
- [ ] 3.2 Implement contour templates (ascending, descending, arch, valley, plateau)
- [ ] 3.3 Add random walk with constraints (max interval, range limits)
- [ ] 3.4 Implement harmonic alignment (chord tones vs passing tones)
- [ ] 3.5 Add neighbor tone and passing tone generation
- [ ] 3.6 Implement approach patterns to chord tones

## 4. Rebracketing Marker Emphasis
- [ ] 4.1 Detect rebracketing markers in lyrics
- [ ] 4.2 Implement melodic emphasis strategies (pitch raise, duration extend, harmonic tone)
- [ ] 4.3 Add contour disruption at markers (change direction, leap)
- [ ] 4.4 Implement pre-marker preparation (ascending approach)
- [ ] 4.5 Add post-marker resolution

## 5. Melodic Scoring Functions
- [ ] 5.1 Implement contour smoothness scoring (prefer stepwise motion)
- [ ] 5.2 Add range scoring (penalize too wide or too narrow)
- [ ] 5.3 Implement harmonic fit scoring (chord tone preference on strong beats)
- [ ] 5.4 Add climax placement scoring (highest note in meaningful position)
- [ ] 5.5 Implement variety scoring (avoid too much repetition)
- [ ] 5.6 Add interval quality scoring (avoid awkward leaps)

## 6. Phrase Structure
- [ ] 6.1 Implement phrase detection from lyrics
- [ ] 6.2 Add cadence generation (melodic endings matching phrase boundaries)
- [ ] 6.3 Implement call-and-response patterns
- [ ] 6.4 Add verse-chorus melodic contrast
- [ ] 6.5 Implement bridge differentiation

## 7. Singability Constraints
- [ ] 7.1 Add vocal range enforcement (configurable, typically C3-C5)
- [ ] 7.2 Implement breathing point insertion
- [ ] 7.3 Add leap recovery (after large leap, move stepwise)
- [ ] 7.4 Implement tessitura analysis (average pitch appropriate for voice type)
- [ ] 7.5 Add awkward interval avoidance

## 8. Integration with Chord Progressions
- [ ] 8.1 Accept `ChordProgressionGenerator` output as input
- [ ] 8.2 Align melody timing to chord changes
- [ ] 8.3 Implement chord tone preference on strong beats
- [ ] 8.4 Add passing tone generation between chord tones
- [ ] 8.5 Implement suspension and anticipation handling

## 9. Brute-Force Search and Scoring
- [ ] 9.1 Implement candidate melody generation (N melodies per phrase)
- [ ] 9.2 Add composite scoring (combine all scoring dimensions)
- [ ] 9.3 Implement top-K selection
- [ ] 9.4 Add score breakdown reporting
- [ ] 9.5 Implement configurable scoring weights

## 10. MIDI Export with Lyrics
- [ ] 10.1 Create MIDI files with melody as Note events
- [ ] 10.2 Embed lyrics as MIDI Lyric meta-events
- [ ] 10.3 Add tempo and time signature metadata
- [ ] 10.4 Implement multi-track export (melody + chords)
- [ ] 10.5 Add synchronization verification

## 11. Testing & Validation
- [ ] 11.1 Write unit tests for syllable parsing
- [ ] 11.2 Test rhythm generation with known lyrics
- [ ] 11.3 Validate harmonic alignment (melodies fit chords)
- [ ] 11.4 Test rebracketing marker emphasis
- [ ] 11.5 Verify MIDI export and lyrics embedding
- [ ] 11.6 Human review: singability and ontological correctness

## 12. Documentation & Examples
- [ ] 12.1 Document lyrics format and rebracketing marker syntax
- [ ] 12.2 Add melody generation usage examples
- [ ] 12.3 Document scoring criteria and weighting strategies
- [ ] 12.4 Create example: generate melody for existing Rainbow Table lyrics
- [ ] 12.5 Document integration with chord progression generator
