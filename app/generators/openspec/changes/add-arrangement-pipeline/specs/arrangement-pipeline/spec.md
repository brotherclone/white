# Arrangement Pipeline

## ADDED Requirements

### Requirement: Song Structure Definition
The system SHALL define song structures as sequences of section types.

#### Scenario: Verse-Chorus structure
- **WHEN** structure="verse-chorus" is specified
- **THEN** sections are: Intro → Verse → Chorus → Verse → Chorus → Bridge → Chorus → Outro

#### Scenario: AABA structure
- **WHEN** structure="AABA" is specified
- **THEN** sections are: A (theme) → A (repeat) → B (bridge) → A (return)

#### Scenario: Custom structure
- **WHEN** custom sections ["intro", "verse", "verse", "outro"] are specified
- **THEN** exactly those sections in that order are used

### Requirement: Section Type Characteristics
The system SHALL apply different characteristics to different section types.

#### Scenario: Intro characteristics
- **WHEN** generating intro
- **THEN** sparse instrumentation (chords only or chords + light drums), lower energy

#### Scenario: Verse characteristics
- **WHEN** generating verse
- **THEN** narrative focus, simpler patterns, lower tessitura melody

#### Scenario: Chorus characteristics
- **WHEN** generating chorus
- **THEN** full arrangement, higher energy, melodic hook, crashes

#### Scenario: Bridge characteristics
- **WHEN** generating bridge
- **THEN** contrast with verse/chorus (different chords, different groove)

### Requirement: Sequential Generator Orchestration
The system SHALL orchestrate generators in sequence: chords → melody → bass → drums.

#### Scenario: Generate chords first
- **WHEN** arrangement pipeline runs
- **THEN** chord progressions are generated for all sections before melody

#### Scenario: Generate melody over chords
- **WHEN** chords are complete
- **THEN** melody is generated using chord progressions as harmonic foundation

#### Scenario: Generate bass after melody
- **WHEN** melody is complete
- **THEN** bass is generated avoiding clashes with melody

#### Scenario: Generate drums last
- **WHEN** bass is complete
- **THEN** drums are generated synchronized with bass

### Requirement: Sectional Chord Progression Generation
The system SHALL generate appropriate chord progressions per section type.

#### Scenario: Verse chord progression
- **WHEN** generating verse chords
- **THEN** simpler, supportive progressions (e.g., I-vi-IV-V)

#### Scenario: Chorus chord progression
- **WHEN** generating chorus chords
- **THEN** same as verse OR more intense (add 7ths, extensions)

#### Scenario: Bridge chord progression
- **WHEN** generating bridge chords
- **THEN** contrasting progression (modulate or change functions)

### Requirement: Sectional Melody Generation
The system SHALL generate melodies appropriate to section type.

#### Scenario: Verse melody tessitura
- **WHEN** generating verse melody
- **THEN** lower tessitura (C4-G4), narrative rhythm

#### Scenario: Chorus melody tessitura
- **WHEN** generating chorus melody
- **THEN** higher tessitura (E4-C5), memorable hook

#### Scenario: Chorus melodic climax
- **WHEN** generating chorus
- **THEN** highest melodic note occurs in chorus (not verse)

### Requirement: Sectional Bass Generation
The system SHALL generate bass lines appropriate to section energy.

#### Scenario: Intro bass sparsity
- **WHEN** generating intro bass
- **THEN** whole notes or half notes (very sparse)

#### Scenario: Verse bass simplicity
- **WHEN** generating verse bass
- **THEN** simple root-fifth patterns

#### Scenario: Chorus bass activity
- **WHEN** generating chorus bass
- **THEN** busier patterns (walking bass, more chord tones, syncopation)

### Requirement: Sectional Drum Generation
The system SHALL generate drum patterns appropriate to section energy.

#### Scenario: Intro drums minimal
- **WHEN** generating intro drums
- **THEN** no drums OR minimal kick only

#### Scenario: Verse drums basic
- **WHEN** generating verse drums
- **THEN** basic backbeat (kick 1,3 / snare 2,4), no crashes

#### Scenario: Chorus drums full
- **WHEN** generating chorus drums
- **THEN** full kit, crash cymbal on beat 1, fills

### Requirement: Transition Detection
The system SHALL detect section boundaries and mark them for transitions.

#### Scenario: Verse to chorus transition
- **WHEN** verse ends and chorus begins
- **THEN** transition point is marked for fill/build generation

#### Scenario: Chorus to bridge transition
- **WHEN** chorus ends and bridge begins
- **THEN** dramatic transition point is marked

### Requirement: Fill Generation at Transitions
The system SHALL generate drum fills before major section changes.

#### Scenario: Fill before chorus
- **WHEN** transitioning verse → chorus
- **THEN** snare fill (1-2 beats) before chorus downbeat

#### Scenario: Fill before bridge
- **WHEN** transitioning chorus → bridge
- **THEN** tom fill (1 bar) before bridge downbeat

#### Scenario: Crash after fill
- **WHEN** fill completes
- **THEN** crash cymbal plays on downbeat of new section

### Requirement: Build-Up Generation
The system SHALL generate build-ups leading to high-energy sections.

#### Scenario: Snare roll build-up
- **WHEN** approaching chorus or drop
- **THEN** snare 16th note roll with crescendo over 1-2 bars

#### Scenario: Bass density increase
- **WHEN** building to chorus
- **THEN** bass rhythm increases (quarter → eighth → sixteenth)

#### Scenario: Cymbal swell
- **WHEN** building to climax
- **THEN** crash cymbal velocity curve simulates swell

### Requirement: Breakdown Generation
The system SHALL generate breakdowns for dramatic contrast.

#### Scenario: Kick-only breakdown
- **WHEN** breakdown section is marked
- **THEN** drums reduce to kick only (drop snare/hats)

#### Scenario: Silence breakdown
- **WHEN** maximum dramatic effect
- **THEN** all instruments drop out for 1-2 beats

#### Scenario: Re-entry after breakdown
- **WHEN** breakdown ends
- **THEN** full arrangement re-enters with crash

### Requirement: Energy Level Mapping
The system SHALL map energy levels to instrumentation and dynamics.

#### Scenario: Low energy intro
- **WHEN** intro has low energy level
- **THEN** only chords play, velocities 50-70

#### Scenario: Medium energy verse
- **WHEN** verse has medium energy level
- **THEN** chords + bass + basic drums, velocities 60-80

#### Scenario: High energy chorus
- **WHEN** chorus has high energy level
- **THEN** full arrangement, velocities 90-110

### Requirement: Rebracketing Marker Detection in Arrangement
The system SHALL detect rebracketing markers and coordinate emphasis across all layers.

#### Scenario: Detect marker timing
- **WHEN** lyrics contain marker "[(is not)]" in verse bar 6
- **THEN** marker timing is recorded (section=verse, bar=6)

#### Scenario: Classify marker type
- **WHEN** marker is classified as "ontological"
- **THEN** emphasis intensity is set to "high"

### Requirement: Coordinated Marker Emphasis
The system SHALL ensure all layers emphasize rebracketing markers simultaneously.

#### Scenario: Melody emphasizes marker
- **WHEN** marker occurs
- **THEN** melody has peak pitch and extended duration at marker

#### Scenario: Bass emphasizes marker
- **WHEN** marker occurs
- **THEN** bass drops to low register or sustains pedal tone

#### Scenario: Drums emphasize marker
- **WHEN** marker occurs
- **THEN** drums play crash + fill before marker

#### Scenario: Chords emphasize marker
- **WHEN** marker occurs
- **THEN** chord changes or lands on harmonic chord tone

#### Scenario: Coordinated timing
- **WHEN** all layers emphasize
- **THEN** emphasis occurs at same beat (synchronized)

### Requirement: Sample Integration
The system SHALL integrate samples and loops into arrangement.

#### Scenario: Ambient pad in intro
- **WHEN** intro is generated
- **THEN** ambient pad sample is placed, sustained throughout intro

#### Scenario: Riser before chorus
- **WHEN** transitioning to chorus
- **THEN** riser sample (filter sweep) plays 1-2 bars before chorus

#### Scenario: Impact at marker
- **WHEN** rebracketing marker occurs
- **THEN** impact sound effect plays at marker beat

### Requirement: Sample Triggering
The system SHALL trigger samples at specific song positions.

#### Scenario: Trigger point specification
- **WHEN** sample is placed
- **THEN** song position is specified (section, bar, beat)

#### Scenario: Volume automation
- **WHEN** sample plays
- **THEN** volume fades in/out or swells as specified

#### Scenario: Tempo synchronization
- **WHEN** rhythmic loop is placed
- **THEN** loop tempo matches song tempo

### Requirement: Multi-Track MIDI Export
The system SHALL export complete arrangement as multi-track MIDI file.

#### Scenario: Track 1 - Chords
- **WHEN** exporting arrangement
- **THEN** Track 1 contains chord progression with piano patch (MIDI patch 0)

#### Scenario: Track 2 - Melody + Lyrics
- **WHEN** exporting arrangement
- **THEN** Track 2 contains melody with voice patch and MIDI lyric meta-events

#### Scenario: Track 3 - Bass
- **WHEN** exporting arrangement
- **THEN** Track 3 contains bass line with fingered bass patch (MIDI patch 33)

#### Scenario: Track 4 - Drums
- **WHEN** exporting arrangement
- **THEN** Track 4 contains drums on channel 10 (General MIDI)

### Requirement: MIDI Metadata
The system SHALL include metadata in exported MIDI files.

#### Scenario: Tempo meta-event
- **WHEN** exporting MIDI
- **THEN** tempo meta-event (e.g., 120 BPM) is at start of file

#### Scenario: Time signature meta-event
- **WHEN** exporting MIDI
- **THEN** time signature meta-event (e.g., 4/4) is at start

#### Scenario: Key signature meta-event
- **WHEN** exporting MIDI
- **THEN** key signature meta-event (e.g., F Major) is at start

#### Scenario: Track name meta-events
- **WHEN** exporting MIDI
- **THEN** each track has track name meta-event ("Chords", "Melody", etc.)

### Requirement: Layer Integration Scoring
The system SHALL score how well layers work together.

#### Scenario: Chord-melody fit
- **WHEN** scoring layer integration
- **THEN** melody notes that are chord tones score higher

#### Scenario: Bass-drum synchronization
- **WHEN** scoring layer integration
- **THEN** kick aligning with bass roots scores higher

#### Scenario: Melody-bass intervals
- **WHEN** scoring layer integration
- **THEN** no harsh intervals (m2, M7) score higher

### Requirement: Structural Coherence Scoring
The system SHALL score arrangement structure quality.

#### Scenario: Logical section sequence
- **WHEN** scoring structure
- **THEN** intro → verse → chorus sequence scores higher than random order

#### Scenario: Smooth transitions
- **WHEN** scoring structure
- **THEN** fills and builds at transitions score higher

#### Scenario: Energy flow
- **WHEN** scoring structure
- **THEN** natural build and release (not flat energy) scores higher

### Requirement: Sectional Contrast Scoring
The system SHALL score contrast between sections.

#### Scenario: Verse vs chorus contrast
- **WHEN** scoring contrast
- **THEN** higher chorus energy vs verse scores higher

#### Scenario: Bridge contrast
- **WHEN** scoring contrast
- **THEN** different chords/groove in bridge scores higher

### Requirement: Rebracketing Emphasis Scoring
The system SHALL score how well markers are emphasized.

#### Scenario: All layers emphasize
- **WHEN** scoring marker emphasis
- **THEN** melody + bass + drums + chords all emphasizing scores 1.0

#### Scenario: Partial emphasis
- **WHEN** scoring marker emphasis
- **THEN** only some layers emphasizing scores proportionally lower

#### Scenario: No emphasis
- **WHEN** scoring marker emphasis
- **THEN** marker not emphasized scores 0.0

### Requirement: Dynamics and Energy Scoring
The system SHALL score energy curve and dynamics.

#### Scenario: Natural energy curve
- **WHEN** scoring dynamics
- **THEN** energy building to chorus, releasing in outro scores higher

#### Scenario: Dynamic range
- **WHEN** scoring dynamics
- **THEN** velocity variation across sections (not flat) scores higher

### Requirement: Performability Scoring
The system SHALL score whether arrangement is humanly performable.

#### Scenario: Individual parts performable
- **WHEN** scoring performability
- **THEN** each part (bass, drums, melody) must be playable by humans

#### Scenario: Ensemble coordination
- **WHEN** scoring performability
- **THEN** parts don't require impossible synchronization

#### Scenario: MIDI export clean
- **WHEN** scoring performability
- **THEN** MIDI file exports without errors

### Requirement: LangGraph Agent Integration
The system SHALL invoke LangGraph agents for qualitative critique.

#### Scenario: Invoke White Agent
- **WHEN** arrangement is complete
- **THEN** White Agent evaluates ontological correctness

#### Scenario: Invoke Violet Agent
- **WHEN** arrangement is complete
- **THEN** Violet Agent evaluates musical quality

#### Scenario: Invoke Black Agent
- **WHEN** arrangement is complete
- **THEN** Black Agent suggests creative alternatives

### Requirement: White Agent Evaluation
The system SHALL use White Agent for conceptual critique.

#### Scenario: Check rebracketing emphasis
- **WHEN** White Agent runs
- **THEN** feedback includes assessment of marker emphasis quality

#### Scenario: Check ontological coherence
- **WHEN** White Agent runs
- **THEN** feedback includes assessment of concept-to-music alignment

#### Scenario: Suggest improvements
- **WHEN** White Agent finds issues
- **THEN** feedback includes specific suggestions for improvement

### Requirement: Violet Agent Evaluation
The system SHALL use Violet Agent for musical critique.

#### Scenario: Check harmonic quality
- **WHEN** Violet Agent runs
- **THEN** feedback includes assessment of chord progressions and voice leading

#### Scenario: Check melody fit
- **WHEN** Violet Agent runs
- **THEN** feedback includes assessment of melody-harmony alignment

#### Scenario: Check rhythmic pocket
- **WHEN** Violet Agent runs
- **THEN** feedback includes assessment of bass-drum synchronization

### Requirement: Black Agent Evaluation
The system SHALL use Black Agent for creative provocation.

#### Scenario: Suggest unconventional choices
- **WHEN** Black Agent runs
- **THEN** feedback includes unexpected structural alternatives

#### Scenario: Challenge conventions
- **WHEN** Black Agent runs
- **THEN** feedback questions conventional verse-chorus-bridge structure

#### Scenario: Push boundaries
- **WHEN** Black Agent runs
- **THEN** feedback suggests extreme or avant-garde approaches

### Requirement: Agent Feedback Loop
The system SHALL iterate based on agent feedback until approval.

#### Scenario: Parse agent feedback
- **WHEN** agent provides feedback
- **THEN** actionable suggestions are extracted

#### Scenario: Adjust parameters
- **WHEN** agent suggests changes
- **THEN** generator parameters are adjusted accordingly

#### Scenario: Regenerate with changes
- **WHEN** parameters adjusted
- **THEN** arrangement is regenerated

#### Scenario: Iterate until approval
- **WHEN** agents still not satisfied
- **THEN** loop continues (max iterations to prevent infinite loop)

### Requirement: Brute-Force Arrangement Search
The system SHALL generate multiple candidate arrangements and select best.

#### Scenario: Generate N candidates
- **WHEN** num_candidates=50 is specified
- **THEN** 50 complete arrangements are generated

#### Scenario: Score all candidates
- **WHEN** candidates are generated
- **THEN** each is scored on all criteria (structure, layers, markers, dynamics, performability)

#### Scenario: Return top K arrangements
- **WHEN** top_k=5 is specified
- **THEN** 5 highest-scoring arrangements are returned with scores and agent feedback

### Requirement: Arrangement Variation
The system SHALL support generating variations of arrangements.

#### Scenario: Vary one parameter
- **WHEN** generating variation
- **THEN** one parameter (e.g., chorus chord progression) changes, rest stays same

#### Scenario: Partial regeneration
- **WHEN** user specifies
- **THEN** only one section (e.g., bridge) is regenerated, others preserved

#### Scenario: A/B comparison
- **WHEN** generating two versions
- **THEN** both are exported for comparison

### Requirement: Manual Override
The system SHALL allow manual specification of section parameters.

#### Scenario: Override verse chord progression
- **WHEN** user provides specific chords for verse
- **THEN** those chords are used instead of generated ones

#### Scenario: Override drum pattern
- **WHEN** user provides specific groove for chorus
- **THEN** that groove is used instead of generated one

### Requirement: Intro Generation
The system SHALL generate appropriate introductions.

#### Scenario: Sparse intro
- **WHEN** generating intro
- **THEN** chords only, no drums, low energy

#### Scenario: Intro build-up
- **WHEN** intro leads to verse
- **THEN** instruments gradually added (chords → bass → drums)

### Requirement: Outro Generation
The system SHALL generate appropriate endings.

#### Scenario: Fade-out outro
- **WHEN** generating fade-out
- **THEN** velocities gradually decrease over outro bars

#### Scenario: Cold ending
- **WHEN** generating cold ending
- **THEN** arrangement stops abruptly on final chord

#### Scenario: Resolving outro
- **WHEN** generating resolving outro
- **THEN** chords resolve to tonic, bass plays root, drums simplify

### Requirement: Chromatic Mode Integration
The system SHALL apply chromatic mode aesthetic across all layers.

#### Scenario: Red mode arrangement
- **WHEN** chromatic_mode="Red" is specified
- **THEN** aggressive chords, intense bass, heavy drums

#### Scenario: Violet mode arrangement
- **WHEN** chromatic_mode="Violet" is specified
- **THEN** complex harmonies, introspective melody, sparse drums

#### Scenario: Yellow mode arrangement
- **WHEN** chromatic_mode="Yellow" is specified
- **THEN** bright harmonies, uplifting melody, light drums

### Requirement: Song Structure Templates
The system SHALL provide predefined song structure templates.

#### Scenario: Verse-Chorus template
- **WHEN** template="verse-chorus" is selected
- **THEN** structure is Intro → V → C → V → C → Bridge → C → Outro

#### Scenario: Verse-Prechorus-Chorus template
- **WHEN** template="verse-prechorus-chorus" is selected
- **THEN** structure is Intro → V → PC → C → V → PC → C → Bridge → C → Outro

#### Scenario: AABA template
- **WHEN** template="AABA" is selected
- **THEN** structure is A → A → B → A

#### Scenario: Through-Composed template
- **WHEN** template="through-composed" is selected
- **THEN** each section is unique, no repetition

### Requirement: Pattern Reuse Across Sections
The system SHALL reuse patterns appropriately across similar sections.

#### Scenario: Verse 1 = Verse 2
- **WHEN** generating Verse 2
- **THEN** Verse 1 patterns are reused (possibly with minor variation)

#### Scenario: Chorus repetition
- **WHEN** chorus repeats
- **THEN** same melody and chords, possibly busier drums on second chorus

### Requirement: Configurable Scoring Weights
The system SHALL allow custom weighting of arrangement scoring criteria.

#### Scenario: Emphasize structure
- **WHEN** weights={"structure": 0.4, "layers": 0.3, "markers": 0.2, "dynamics": 0.1}
- **THEN** arrangements with better structure score highest

#### Scenario: Emphasize marker emphasis
- **WHEN** weights={"markers": 0.5, "structure": 0.2, "layers": 0.2, "dynamics": 0.1}
- **THEN** arrangements emphasizing rebracketing strongly score highest

### Requirement: Full Arrangement Export
The system SHALL export complete, playable MIDI arrangement.

#### Scenario: Export multi-track MIDI
- **WHEN** arrangement is complete
- **THEN** single MIDI file with all tracks (chords, melody, bass, drums) is created

#### Scenario: Human-performable output
- **WHEN** MIDI is exported
- **THEN** real musicians can load MIDI into DAW and perform parts

#### Scenario: Ready for recording
- **WHEN** MIDI is exported
- **THEN** file is ready for studio recording session with real instruments
