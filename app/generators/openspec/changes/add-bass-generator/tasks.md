# Implementation Tasks

## 1. Chord Progression Integration
- [ ] 1.1 Accept `ChordProgressionGenerator` output as input
- [ ] 1.2 Extract harmonic rhythm (chord change timing)
- [ ] 1.3 Parse chord tones from chord symbols (root, third, fifth, seventh, extensions)
- [ ] 1.4 Synchronize bass timing with chord durations
- [ ] 1.5 Handle chord inversions (if present in progression)

## 2. Basic Bass Pattern Generation
- [ ] 2.1 Create `BassLineGenerator` class
- [ ] 2.2 Implement root-note placement on chord changes
- [ ] 2.3 Add simple rhythm patterns (whole, half, quarter notes)
- [ ] 2.4 Implement chord tone selection (roots, fifths, thirds)
- [ ] 2.5 Add octave selection within playable range

## 3. Walking Bass Patterns
- [ ] 3.1 Implement stepwise motion between chord roots
- [ ] 3.2 Add chromatic approach tones (half-step below next root)
- [ ] 3.3 Implement diatonic passing tones (scale tones between chords)
- [ ] 3.4 Add neighbor tones (upper/lower neighbors to chord tones)
- [ ] 3.5 Implement target-tone anticipation (approach from below or above)

## 4. Groove Pattern Templates
- [ ] 4.1 Create groove pattern database (funk, rock, jazz, R&B)
- [ ] 4.2 Implement syncopation patterns (anticipations, delayed attacks)
- [ ] 4.3 Add ghost note generation (muted/percussive notes)
- [ ] 4.4 Implement rhythmic subdivision (8th, 16th note patterns)
- [ ] 4.5 Add rests and silence (rhythmic breathing)

## 5. Arpeggio Patterns
- [ ] 5.1 Implement chord tone arpeggiation (R-3-5-8 patterns)
- [ ] 5.2 Add common arpeggio variations (R-5-8-5, R-5-3-5)
- [ ] 5.3 Implement ascending/descending arpeggios
- [ ] 5.4 Add arpeggio rhythm variations (swing, straight, dotted)

## 6. Pedal Tones and Drones
- [ ] 6.1 Detect opportunities for pedal tones (static bass under changing chords)
- [ ] 6.2 Implement sustained root notes
- [ ] 6.3 Add octave doubling (bass + higher octave for thickness)
- [ ] 6.4 Implement pedal tone release (when to move off pedal)

## 7. Melody Integration and Clash Avoidance
- [ ] 7.1 Accept `MelodyGenerator` output as input
- [ ] 7.2 Analyze melody-bass intervals at each beat
- [ ] 7.3 Detect harsh intervals (minor 2nd, major 7th)
- [ ] 7.4 Adjust bass notes to avoid clashes (octave shifts, alternate chord tones)
- [ ] 7.5 Implement counterpoint rules (contrary motion preference)

## 8. Playability Constraints
- [ ] 8.1 Enforce bass range (configurable, typically E1-G3 for 4-string)
- [ ] 8.2 Detect impossible position shifts (max fret distance per time unit)
- [ ] 8.3 Add fingering simulation (4 fingers, 4 strings, chromatic frets)
- [ ] 8.4 Implement string preference (favor lower strings for lower notes)
- [ ] 8.5 Add open string detection (E, A, D, G on standard 4-string)

## 9. Rebracketing Marker Emphasis
- [ ] 9.1 Detect rebracketing markers from lyrics
- [ ] 9.2 Implement low register emphasis (drop to E1-G1 range)
- [ ] 9.3 Add pedal tone strategy (sustain through marker)
- [ ] 9.4 Implement rhythmic density shift (more/fewer notes around marker)
- [ ] 9.5 Add harmonic tension (tritone subs, chromatic approaches before marker)
- [ ] 9.6 Implement post-marker resolution (return to root after marker)

## 10. Bass Scoring Functions
- [ ] 10.1 Implement harmonic fit scoring (chord tones vs non-chord tones)
- [ ] 10.2 Add rhythmic coherence scoring (pattern consistency, groove)
- [ ] 10.3 Implement playability scoring (range, position shifts, fingering)
- [ ] 10.4 Add melody support scoring (interval quality, counterpoint)
- [ ] 10.5 Implement groove scoring (syncopation, ghost notes, feel)
- [ ] 10.6 Add variety scoring (root/fifth/third distribution, rhythmic variation)

## 11. Chromatic Approaches and Passing Tones
- [ ] 11.1 Implement chromatic approach from below (half-step leading tone)
- [ ] 11.2 Add chromatic approach from above (half-step descending)
- [ ] 11.3 Implement double chromatic approach (encircling target)
- [ ] 11.4 Add diatonic passing tones (scale-wise fills)
- [ ] 11.5 Implement passing chord insertion (diminished, augmented passing chords)

## 12. Rhythmic Articulation
- [ ] 12.1 Add note duration control (staccato, legato, sustained)
- [ ] 12.2 Implement ghost notes (low velocity, muted)
- [ ] 12.3 Add slides/glissandi (pitch bends between notes)
- [ ] 12.4 Implement hammer-ons and pull-offs (MIDI articulation marks)
- [ ] 12.5 Add rest placement (breathing, groove pockets)

## 13. Style-Specific Pattern Generation
- [ ] 13.1 Implement "rock" style (root-fifth patterns, driving eighths)
- [ ] 13.2 Add "funk" style (syncopated 16ths, ghost notes, slap markers)
- [ ] 13.3 Implement "jazz" style (walking quarters, chromatic approaches)
- [ ] 13.4 Add "R&B" style (syncopated anticipations, smooth melodic bass)
- [ ] 13.5 Implement "electronic" style (repetitive arpeggio patterns, long sustains)

## 14. Harmonic Movement Quality
- [ ] 14.1 Detect strong-to-weak function movement (I→V, IV→I)
- [ ] 14.2 Implement voice leading (minimize bass note jumps between chords)
- [ ] 14.3 Add bass line melodic smoothness (prefer stepwise to leaps)
- [ ] 14.4 Implement leap recovery (after leap, move stepwise)
- [ ] 14.5 Add bass climax detection (avoid competing with melody climax)

## 15. Integration with Drum Patterns (Future)
- [ ] 15.1 Accept drum pattern timing as input
- [ ] 15.2 Align bass attacks with kick drum hits
- [ ] 15.3 Add ghost notes aligned with snare backbeat
- [ ] 15.4 Implement hi-hat synchronization (16th note subdivision alignment)
- [ ] 15.5 Add groove pocket analysis (intentional timing ahead/behind beat)

## 16. Brute-Force Search and Scoring
- [ ] 16.1 Implement candidate bass line generation (N lines per progression)
- [ ] 16.2 Add composite scoring (combine all scoring dimensions)
- [ ] 16.3 Implement top-K selection
- [ ] 16.4 Add score breakdown reporting
- [ ] 16.5 Implement configurable scoring weights

## 17. MIDI Export with Bass Track
- [ ] 17.1 Create MIDI files with bass as separate track
- [ ] 17.2 Add bass-specific MIDI channels and patches (fingered bass, slap bass, etc.)
- [ ] 17.3 Implement velocity mapping (ghost notes, accents, normal)
- [ ] 17.4 Add articulation meta-events (slides, hammer-ons)
- [ ] 17.5 Implement multi-track synchronization (bass + chords + melody)

## 18. Pattern Variation and Evolution
- [ ] 18.1 Implement pattern repetition (verse 1 = verse 2 bass line)
- [ ] 18.2 Add variation on repeat (slight rhythmic/melodic changes)
- [ ] 18.3 Implement chorus intensification (busier patterns in chorus)
- [ ] 18.4 Add bridge contrast (different pattern style for bridge)
- [ ] 18.5 Implement build-up and breakdown (increase/decrease density)

## 19. Testing & Validation
- [ ] 19.1 Write unit tests for chord tone extraction
- [ ] 19.2 Test walking bass generation with known progressions
- [ ] 19.3 Validate melody-bass interval quality
- [ ] 19.4 Test rebracketing marker emphasis
- [ ] 19.5 Verify MIDI export and playability
- [ ] 19.6 Human review: bassist playability and groove feel

## 20. Documentation & Examples
- [ ] 20.1 Document bass pattern syntax and style options
- [ ] 20.2 Add bass line generation usage examples
- [ ] 20.3 Document scoring criteria and weighting strategies
- [ ] 20.4 Create example: generate bass for existing Rainbow Table chord progressions
- [ ] 20.5 Document integration with chord and melody generators
