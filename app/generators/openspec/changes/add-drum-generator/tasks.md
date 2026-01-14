# Implementation Tasks

## 1. Drum Kit Definition
- [ ] 1.1 Create `DrumKit` class with General MIDI drum map
- [ ] 1.2 Define voice types (kick, snare, hi-hat, ride, crash, toms, percussion)
- [ ] 1.3 Map MIDI note numbers to drum voices (e.g., 36=kick, 38=snare)
- [ ] 1.4 Add articulation types (open/closed hi-hat, rim shot, cross stick)
- [ ] 1.5 Define velocity ranges for ghost notes, normal, accents

## 2. Basic Pattern Generation
- [ ] 2.1 Create `DrumPatternGenerator` class
- [ ] 2.2 Implement simple backbeat (kick 1,3 / snare 2,4 / hats 8ths)
- [ ] 2.3 Add time signature support (4/4, 3/4, 6/8, 7/8)
- [ ] 2.4 Implement pattern duration (1, 2, 4, 8 bar loops)
- [ ] 2.5 Add basic velocity mapping

## 3. Groove Template Database
- [ ] 3.1 Create groove template structure (kick/snare/hat patterns per style)
- [ ] 3.2 Implement "rock" grooves (backbeat, straight 8ths)
- [ ] 3.3 Add "funk" grooves (syncopated kicks, 16th hats, ghost snares)
- [ ] 3.4 Implement "jazz" grooves (swing ride, sparse kicks, ghost snares)
- [ ] 3.5 Add "electronic" grooves (four-on-floor, quantized)
- [ ] 3.6 Implement "hip-hop/trap" grooves (half-time feel, trap hats)
- [ ] 3.7 Add "breakbeat" grooves (Amen, Funky Drummer patterns)

## 4. Multi-Voice Pattern Generation
- [ ] 4.1 Implement independent rhythm generation per voice
- [ ] 4.2 Add kick drum patterns (roots, syncopations)
- [ ] 4.3 Implement snare patterns (backbeat, ghost notes)
- [ ] 4.4 Add hi-hat patterns (8ths, 16ths, open/closed)
- [ ] 4.5 Implement ride cymbal patterns (alternative to hi-hat)
- [ ] 4.6 Add tom patterns (fills, accents)
- [ ] 4.7 Implement crash cymbal placement (sectional emphasis)
- [ ] 4.8 Add percussion (shakers, tambourine, cowbell)

## 5. Kick-Bass Synchronization
- [ ] 5.1 Accept `BassLineGenerator` output as input
- [ ] 5.2 Detect bass root note placements
- [ ] 5.3 Align kick drum with bass roots (especially beat 1)
- [ ] 5.4 Match bass anticipations with kick syncopations
- [ ] 5.5 Add "pocket" offset (intentional timing variation)

## 6. Dynamic Variation
- [ ] 6.1 Implement ghost note generation (velocity 20-40)
- [ ] 6.2 Add accent marking (velocity 100-127)
- [ ] 6.3 Implement normal hits (velocity 60-90)
- [ ] 6.4 Add crescendo (gradual velocity increase)
- [ ] 6.5 Implement diminuendo (gradual velocity decrease)
- [ ] 6.6 Add velocity humanization (random variation ±5-10)

## 7. Hi-Hat Articulation
- [ ] 7.1 Implement closed hi-hat patterns
- [ ] 7.2 Add open hi-hat accents
- [ ] 7.3 Implement hi-hat "chick" (foot pedal close)
- [ ] 7.4 Add open/closed alternation patterns
- [ ] 7.5 Implement hi-hat dynamics (soft, medium, hard)

## 8. Fill Generation
- [ ] 8.1 Detect phrase boundaries (every 4, 8, or 16 bars)
- [ ] 8.2 Generate snare fills (rolls, accents)
- [ ] 8.3 Add tom fills (descending/ascending patterns)
- [ ] 8.4 Implement combination fills (snare + toms)
- [ ] 8.5 Add fill crescendo (build intensity toward downbeat)
- [ ] 8.6 Implement fill duration (1 beat, 2 beats, 1 bar)

## 9. Cymbal Crashes and Rides
- [ ] 9.1 Place crash cymbals on sectional downbeats (chorus, verse)
- [ ] 9.2 Add ride cymbal patterns (jazz, rock variations)
- [ ] 9.3 Implement crash-ride combination (hit both simultaneously)
- [ ] 9.4 Add crash decay simulation (sustained cymbal)
- [ ] 9.5 Implement ride bell accents

## 10. Rebracketing Marker Emphasis
- [ ] 10.1 Detect rebracketing markers from lyrics
- [ ] 10.2 Implement crash cymbal at marker
- [ ] 10.3 Add fill leading into marker (1-2 beats before)
- [ ] 10.4 Implement rhythmic break (drop out before marker)
- [ ] 10.5 Add dynamic surge (increase velocity around marker)
- [ ] 10.6 Implement pattern disruption (change groove at marker)
- [ ] 10.7 Add post-marker resolution (return to groove)

## 11. Playability Constraints
- [ ] 11.1 Enforce max 4 simultaneous notes (2 hands, 2 feet)
- [ ] 11.2 Detect impossible limb combinations (e.g., hi-hat + ride same hand)
- [ ] 11.3 Add tempo-based subdivision limits (no 32nds at high BPM)
- [ ] 11.4 Implement recovery time after fills (hands return to groove)
- [ ] 11.5 Add stick/hand crossing detection (avoid awkward motions)

## 12. Groove Scoring Functions
- [ ] 12.1 Implement pocket scoring (kick/snare align with strong/weak beats)
- [ ] 12.2 Add syncopation appropriateness scoring (style-dependent)
- [ ] 12.3 Implement subdivision consistency scoring
- [ ] 12.4 Add groove feel scoring (swing, straight, shuffle)

## 13. Bass Synchronization Scoring
- [ ] 13.1 Score kick alignment with bass roots on beat 1
- [ ] 13.2 Add kick-bass anticipation matching scoring
- [ ] 13.3 Implement kick-bass conflict detection (clashes)
- [ ] 13.4 Add pocket offset scoring (tight vs loose feel)

## 14. Variety Scoring
- [ ] 14.1 Score voice usage (kick, snare, hats, crashes, toms)
- [ ] 14.2 Add dynamic range scoring (ghost notes, accents)
- [ ] 14.3 Implement pattern variation scoring (avoid exact repetition)
- [ ] 14.4 Add fill frequency scoring (too many or too few)

## 15. Playability Scoring
- [ ] 15.1 Score limb coordination (max 4 simultaneous)
- [ ] 15.2 Add physically impossible pattern detection
- [ ] 15.3 Implement tempo/subdivision feasibility scoring
- [ ] 15.4 Add recovery time scoring (after fills)

## 16. Fill Quality Scoring
- [ ] 16.1 Score fill placement at phrase boundaries
- [ ] 16.2 Add fill-to-groove transition smoothness scoring
- [ ] 16.3 Implement fill intensity appropriateness
- [ ] 16.4 Add fill frequency scoring (not too many)

## 17. Dynamic Expression Scoring
- [ ] 17.1 Score velocity variation (ghost notes, accents)
- [ ] 17.2 Add crescendo/diminuendo scoring
- [ ] 17.3 Implement humanization scoring (slight timing/velocity variation)
- [ ] 17.4 Add flat dynamics penalty

## 18. Syncopation and Subdivision
- [ ] 18.1 Implement straight 8th note patterns
- [ ] 18.2 Add 16th note patterns
- [ ] 18.3 Implement swing/shuffle feel (triplet subdivision)
- [ ] 18.4 Add syncopated anticipations ("and" of beats)
- [ ] 18.5 Implement polyrhythms (3 over 4, etc.) (advanced)

## 19. Style-Specific Pattern Selection
- [ ] 19.1 Map groove templates to style parameter
- [ ] 19.2 Implement rock drum generation
- [ ] 19.3 Add funk drum generation
- [ ] 19.4 Implement jazz drum generation
- [ ] 19.5 Add electronic/dance drum generation
- [ ] 19.6 Implement hip-hop/trap drum generation

## 20. Sectional Variation
- [ ] 20.1 Implement verse pattern (simpler, lower energy)
- [ ] 20.2 Add chorus pattern (busier, crashes, higher energy)
- [ ] 20.3 Implement bridge contrast (different groove)
- [ ] 20.4 Add intro/outro patterns (stripped down)
- [ ] 20.5 Implement breakdown patterns (minimal, sparse)

## 21. Build-Up and Breakdown
- [ ] 21.1 Detect build-up sections (pre-chorus, pre-drop)
- [ ] 21.2 Implement snare roll build-ups
- [ ] 21.3 Add tom roll build-ups
- [ ] 21.4 Implement crash cymbal swell
- [ ] 21.5 Add breakdown patterns (drop to kick only, or silence)

## 22. Humanization
- [ ] 22.1 Add slight timing variation (±5-15ms per hit)
- [ ] 22.2 Implement velocity humanization (±5-10 per hit)
- [ ] 22.3 Add "flamming" (very slight note doubling)
- [ ] 22.4 Implement natural acceleration/deceleration (slight tempo drift)

## 23. Brute-Force Search and Scoring
- [ ] 23.1 Implement candidate drum pattern generation (N patterns per progression)
- [ ] 23.2 Add composite scoring (combine all scoring dimensions)
- [ ] 23.3 Implement top-K selection
- [ ] 23.4 Add score breakdown reporting
- [ ] 23.5 Implement configurable scoring weights

## 24. MIDI Export with Drum Track
- [ ] 24.1 Create MIDI files with drums as channel 10 (General MIDI standard)
- [ ] 24.2 Map drum voices to General MIDI note numbers
- [ ] 24.3 Implement velocity export
- [ ] 24.4 Add open/closed hi-hat controller messages
- [ ] 24.5 Implement multi-track synchronization (drums + bass + chords + melody)

## 25. Integration with Chord Progressions
- [ ] 25.1 Accept `ChordProgressionGenerator` output as input
- [ ] 25.2 Detect harmonic rhythm (chord change timing)
- [ ] 25.3 Emphasize chord changes with kick or crash
- [ ] 25.4 Add fills before major chord changes (IV → V → I)

## 26. Integration with Melody
- [ ] 26.1 Accept `MelodyGenerator` output as input
- [ ] 26.2 Detect melody phrase boundaries
- [ ] 26.3 Add fills at melody phrase endings
- [ ] 26.4 Leave space for melody (avoid crash during vocal climax)

## 27. Tempo and Time Signature Handling
- [ ] 27.1 Support common time signatures (4/4, 3/4, 6/8, 5/4, 7/8)
- [ ] 27.2 Implement tempo-dependent subdivision selection
- [ ] 27.3 Add time signature change handling (if progression includes changes)
- [ ] 27.4 Implement metric modulation (advanced)

## 28. Testing & Validation
- [ ] 28.1 Write unit tests for groove template parsing
- [ ] 28.2 Test kick-bass synchronization with known bass patterns
- [ ] 28.3 Validate playability (no impossible limb combinations)
- [ ] 28.4 Test rebracketing marker emphasis
- [ ] 28.5 Verify MIDI export and General MIDI compliance
- [ ] 28.6 Human review: drummer playability and groove feel

## 29. Documentation & Examples
- [ ] 29.1 Document groove template syntax and style options
- [ ] 29.2 Add drum pattern generation usage examples
- [ ] 29.3 Document scoring criteria and weighting strategies
- [ ] 29.4 Create example: generate drums for existing Rainbow Table progressions
- [ ] 29.5 Document integration with chord, bass, and melody generators
