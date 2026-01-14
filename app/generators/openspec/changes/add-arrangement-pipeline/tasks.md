# Implementation Tasks

## 1. Song Structure Definition
- [ ] 1.1 Create `SongStructure` class representing section sequence
- [ ] 1.2 Define section types (intro, verse, prechorus, chorus, bridge, outro)
- [ ] 1.3 Implement structure templates (verse-chorus, AABA, through-composed)
- [ ] 1.4 Add section duration specification (bars per section)
- [ ] 1.5 Implement structure validation (no invalid sequences)

## 2. Arrangement Pipeline Core
- [ ] 2.1 Create `ArrangementPipeline` class
- [ ] 2.2 Implement sequential generator orchestration (chords → melody → bass → drums)
- [ ] 2.3 Add section-by-section generation
- [ ] 2.4 Implement cross-section coordination (verse 1 = verse 2 pattern reuse)
- [ ] 2.5 Add arrangement state management (track what's been generated)

## 3. Sectional Variation Rules
- [ ] 3.1 Define verse characteristics (lower energy, simpler patterns)
- [ ] 3.2 Define chorus characteristics (higher energy, full arrangement)
- [ ] 3.3 Define bridge characteristics (contrast, different patterns)
- [ ] 3.4 Define intro/outro characteristics (sparse, build/reduce)
- [ ] 3.5 Implement parameter mapping (section type → generator parameters)

## 4. Chord Progression per Section
- [ ] 4.1 Generate chord progression for intro
- [ ] 4.2 Generate chord progression for verse (use verse parameters)
- [ ] 4.3 Generate chord progression for chorus (same OR different from verse)
- [ ] 4.4 Generate chord progression for bridge (contrast with verse/chorus)
- [ ] 4.5 Generate chord progression for outro (resolve to tonic)
- [ ] 4.6 Handle chord progression repetition (verse 1 = verse 2)

## 5. Melody Generation per Vocal Section
- [ ] 5.1 Generate melody for verse with lyrics
- [ ] 5.2 Generate melody for chorus with lyrics
- [ ] 5.3 Generate melody for bridge with lyrics
- [ ] 5.4 Handle melodic repetition (verse 1 = verse 2 with variation)
- [ ] 5.5 Ensure chorus melody is memorable (melodic hook)

## 6. Bass Line Generation per Section
- [ ] 6.1 Generate bass for intro (sparse, roots on whole notes)
- [ ] 6.2 Generate bass for verse (simple root-fifth patterns)
- [ ] 6.3 Generate bass for chorus (busier, more chord tones or walking)
- [ ] 6.4 Generate bass for bridge (pedal tone or contrasting pattern)
- [ ] 6.5 Generate bass for outro (simplify, resolve to root)

## 7. Drum Pattern Generation per Section
- [ ] 7.1 Generate drums for intro (no drums OR minimal kick)
- [ ] 7.2 Generate drums for verse (basic groove, no crashes)
- [ ] 7.3 Generate drums for chorus (full kit, crashes, fills)
- [ ] 7.4 Generate drums for bridge (different groove or half-time)
- [ ] 7.5 Generate drums for outro (simplify, fade out)

## 8. Transition Detection
- [ ] 8.1 Detect section boundaries in song structure
- [ ] 8.2 Identify transition types (verse → chorus, chorus → bridge, etc.)
- [ ] 8.3 Mark transition points for fill/build generation
- [ ] 8.4 Add transition intensity levels (subtle, moderate, dramatic)

## 9. Fill Generation at Transitions
- [ ] 9.1 Generate snare fill before chorus (1-2 beats)
- [ ] 9.2 Generate tom fill before bridge (1 bar)
- [ ] 9.3 Generate turnaround fills (every 8 bars in verse/chorus)
- [ ] 9.4 Add crash cymbal on downbeat after fill
- [ ] 9.5 Implement fill intensity based on transition type

## 10. Build-Up Generation
- [ ] 10.1 Detect build-up sections (pre-chorus, pre-drop)
- [ ] 10.2 Generate snare roll build-up (16th notes, crescendo)
- [ ] 10.3 Add bass density increase (quarter → eighth → sixteenth)
- [ ] 10.4 Implement cymbal swell (simulated via velocity curve)
- [ ] 10.5 Add filter sweep samples (riser, build effect)

## 11. Breakdown Generation
- [ ] 11.1 Detect breakdown sections
- [ ] 11.2 Generate kick-only breakdown (drop snare/hats)
- [ ] 11.3 Add silence breakdown (full stop for 1-2 beats)
- [ ] 11.4 Implement gradual breakdown (progressive voice removal)
- [ ] 11.5 Add re-entry after breakdown (crash + full arrangement)

## 12. Energy Level Mapping
- [ ] 12.1 Define energy levels per section type (intro=low, chorus=high)
- [ ] 12.2 Implement energy curve across song (build and release)
- [ ] 12.3 Map energy to instrumentation (low=chords only, high=full band)
- [ ] 12.4 Map energy to dynamics (velocity ranges)
- [ ] 12.5 Map energy to rhythmic density (note subdivision)

## 13. Rebracketing Marker Detection
- [ ] 13.1 Parse lyrics for rebracketing markers
- [ ] 13.2 Identify marker timing (which section, which bar/beat)
- [ ] 13.3 Classify marker type (spatial, temporal, ontological, etc.)
- [ ] 13.4 Determine marker emphasis intensity

## 14. Coordinated Rebracketing Emphasis
- [ ] 14.1 Ensure melody emphasizes marker (pitch, duration)
- [ ] 14.2 Ensure bass emphasizes marker (low register, pedal tone)
- [ ] 14.3 Ensure drums emphasize marker (crash, fill, break)
- [ ] 14.4 Ensure chords emphasize marker (chord change, harmonic tone)
- [ ] 14.5 Add sample/effect at marker (impact sound)

## 15. Sample and Loop Integration
- [ ] 15.1 Create sample library structure (pads, loops, effects, vocal chops)
- [ ] 15.2 Implement sample selection based on style and energy
- [ ] 15.3 Place ambient pads in intro/outro
- [ ] 15.4 Place risers/impacts at transitions
- [ ] 15.5 Place rhythmic loops in high-energy sections

## 16. Sample Triggering and Automation
- [ ] 16.1 Define sample trigger points (song position in bars/beats)
- [ ] 16.2 Add volume automation (fade in/out, swell)
- [ ] 16.3 Implement filter automation (for risers, sweeps)
- [ ] 16.4 Add tempo-sync for rhythmic loops
- [ ] 16.5 Implement key-sync for tonal samples (pads)

## 17. Multi-Track MIDI Export
- [ ] 17.1 Create multi-track MIDI file (Type 1)
- [ ] 17.2 Assign Track 1: Chord progression (piano or pad patch)
- [ ] 17.3 Assign Track 2: Melody + lyrics (voice patch + lyric meta-events)
- [ ] 17.4 Assign Track 3: Bass line (bass patch)
- [ ] 17.5 Assign Track 4: Drums (channel 10, GM drums)
- [ ] 17.6 Assign Track 5+: Samples/loops (if MIDI-triggerable)

## 18. MIDI Metadata and Tempo
- [ ] 18.1 Add tempo meta-events to MIDI file
- [ ] 18.2 Add time signature meta-events
- [ ] 18.3 Add key signature meta-events
- [ ] 18.4 Add track name meta-events
- [ ] 18.5 Add copyright/comment meta-events (optional)

## 19. Layer Integration Scoring
- [ ] 19.1 Score chord-melody fit (harmonic alignment)
- [ ] 19.2 Score bass-drum synchronization (kick-bass pocket)
- [ ] 19.3 Score melody-bass intervals (no clashes)
- [ ] 19.4 Score overall layer coherence (all parts work together)

## 20. Structural Coherence Scoring
- [ ] 20.1 Score section sequence logic (intro → verse → chorus makes sense)
- [ ] 20.2 Score transition smoothness (fills, builds appropriate)
- [ ] 20.3 Score energy flow (natural build and release)
- [ ] 20.4 Score repetition vs variation (verse 1 ≈ verse 2, but not identical)

## 21. Sectional Contrast Scoring
- [ ] 21.1 Score verse vs chorus contrast (energy, instrumentation, melody)
- [ ] 21.2 Score bridge contrast (different enough from verse/chorus)
- [ ] 21.3 Score intro/outro appropriateness (sparse, resolves)

## 22. Rebracketing Emphasis Scoring
- [ ] 22.1 Score marker detection (all markers found)
- [ ] 22.2 Score cross-layer coordination (all layers emphasize)
- [ ] 22.3 Score emphasis intensity (appropriate for marker type)
- [ ] 22.4 Score post-marker resolution (return to normal after emphasis)

## 23. Dynamics and Energy Scoring
- [ ] 23.1 Score energy curve (build/release feels natural)
- [ ] 23.2 Score dynamic range (velocity variation across sections)
- [ ] 23.3 Score instrumentation appropriateness (intro sparse, chorus full)

## 24. Performability Scoring
- [ ] 24.1 Score individual part performability (bass, drums, melody each humanly possible)
- [ ] 24.2 Score ensemble coordination (parts don't require impossible synchronization)
- [ ] 24.3 Score MIDI export quality (clean, no errors)

## 25. LangGraph Agent Integration
- [ ] 25.1 Define agent prompt templates (White, Violet, Black)
- [ ] 25.2 Implement agent invocation after arrangement generation
- [ ] 25.3 Parse agent feedback for actionable suggestions
- [ ] 25.4 Implement parameter adjustment based on feedback
- [ ] 25.5 Add agent approval loop (iterate until agents satisfied)

## 26. White Agent (Conceptual Critic)
- [ ] 26.1 White Agent evaluates ontological correctness
- [ ] 26.2 White Agent checks rebracketing marker emphasis
- [ ] 26.3 White Agent assesses conceptual coherence
- [ ] 26.4 White Agent suggests improvements

## 27. Violet Agent (Musical Critic)
- [ ] 27.1 Violet Agent evaluates harmonic quality
- [ ] 27.2 Violet Agent checks voice leading and melody fit
- [ ] 27.3 Violet Agent assesses rhythmic pocket and groove
- [ ] 27.4 Violet Agent suggests musical improvements

## 28. Black Agent (Creative Provocateur)
- [ ] 28.1 Black Agent suggests unconventional choices
- [ ] 28.2 Black Agent proposes structural disruptions
- [ ] 28.3 Black Agent challenges conventional wisdom
- [ ] 28.4 Black Agent pushes creative boundaries

## 29. Brute-Force Arrangement Search
- [ ] 29.1 Generate multiple candidate arrangements (N complete songs)
- [ ] 29.2 Score each candidate on all criteria
- [ ] 29.3 Return top-K arrangements with score breakdowns
- [ ] 29.4 Include agent feedback for each top candidate

## 30. Variation and Refinement
- [ ] 30.1 Implement arrangement variation (change one parameter, regenerate)
- [ ] 30.2 Add manual override (user can specify section parameters)
- [ ] 30.3 Implement partial regeneration (regenerate just chorus, keep verse)
- [ ] 30.4 Add A/B comparison (generate two versions, pick best)

## 31. Song Structure Templates
- [ ] 31.1 Implement "Verse-Chorus" template
- [ ] 31.2 Implement "Verse-Prechorus-Chorus" template
- [ ] 31.3 Implement "AABA" template
- [ ] 31.4 Implement "Through-Composed" template
- [ ] 31.5 Implement custom structure (user-defined section sequence)

## 32. Intro and Outro Generation
- [ ] 32.1 Generate sparse intro (chords only, or chords + light drums)
- [ ] 32.2 Implement intro build-up (gradually add instruments)
- [ ] 32.3 Generate resolving outro (simplify to tonic, fade)
- [ ] 32.4 Implement cold ending (abrupt stop on final chord)
- [ ] 32.5 Add fade-out simulation (gradual velocity decrease)

## 33. Chromatic Mode Integration
- [ ] 33.1 Accept chromatic mode parameter (Red, Orange, Yellow, Green, Cyan, Blue, Indigo, Violet)
- [ ] 33.2 Map chromatic mode to chord voicing style
- [ ] 33.3 Map chromatic mode to melody character
- [ ] 33.4 Map chromatic mode to bass/drum groove feel
- [ ] 33.5 Ensure all layers reflect chromatic mode aesthetic

## 34. Testing & Validation
- [ ] 34.1 Write unit tests for song structure parsing
- [ ] 34.2 Test sequential generation (chords → melody → bass → drums)
- [ ] 34.3 Validate transition generation (fills, builds, breakdowns)
- [ ] 34.4 Test rebracketing marker coordination across layers
- [ ] 34.5 Verify multi-track MIDI export
- [ ] 34.6 Human review: listen to full arrangements, assess quality

## 35. Documentation & Examples
- [ ] 35.1 Document song structure template syntax
- [ ] 35.2 Add arrangement pipeline usage examples
- [ ] 35.3 Document scoring criteria and weighting
- [ ] 35.4 Create example: generate full arrangement for Rainbow Table concept
- [ ] 35.5 Document agent integration and feedback loop
