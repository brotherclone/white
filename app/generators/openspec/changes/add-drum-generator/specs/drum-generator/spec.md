# Drum Pattern Generator

## ADDED Requirements

### Requirement: Drum Kit Voice Definition
The system SHALL define drum kit voices with MIDI note mappings (General MIDI standard).

#### Scenario: Map kick drum
- **WHEN** generating kick drum hit
- **THEN** MIDI note 36 (Acoustic Bass Drum) is used

#### Scenario: Map snare drum
- **WHEN** generating snare hit
- **THEN** MIDI note 38 (Acoustic Snare) is used

#### Scenario: Map hi-hat closed
- **WHEN** generating closed hi-hat
- **THEN** MIDI note 42 (Closed Hi-Hat) is used

#### Scenario: Map crash cymbal
- **WHEN** generating crash cymbal
- **THEN** MIDI note 49 (Crash Cymbal 1) is used

### Requirement: Basic Backbeat Generation
The system SHALL generate simple backbeat patterns (kick on 1 and 3, snare on 2 and 4).

#### Scenario: Generate rock backbeat
- **WHEN** generating basic rock pattern in 4/4
- **THEN** kick plays on beats 1 and 3, snare on beats 2 and 4

#### Scenario: Add hi-hat subdivision
- **WHEN** generating with hi-hat
- **THEN** hi-hat plays eighth notes throughout

### Requirement: Groove Template Selection
The system SHALL provide style-specific groove templates.

#### Scenario: Rock groove
- **WHEN** style="rock" is specified
- **THEN** kick on 1 and 3, snare on 2 and 4, straight 8th note hi-hats

#### Scenario: Funk groove
- **WHEN** style="funk" is specified
- **THEN** syncopated kicks, ghost note snares on 16ths, 16th note hi-hats with accents

#### Scenario: Jazz groove
- **WHEN** style="jazz" is specified
- **THEN** sparse syncopated kicks, ride cymbal swing pattern, ghost snares

#### Scenario: Electronic groove
- **WHEN** style="electronic" is specified
- **THEN** four-on-the-floor kick, snare on 3 (or 2 and 4), quantized hi-hats

### Requirement: Kick-Bass Synchronization
The system SHALL align kick drum with bass line root notes.

#### Scenario: Kick on bass root
- **WHEN** bass plays root note on beat 1
- **THEN** kick drum also plays on beat 1

#### Scenario: Kick matches bass anticipation
- **WHEN** bass anticipates on "and of 4"
- **THEN** kick also plays on "and of 4"

#### Scenario: No kick-bass conflict
- **WHEN** bass and kick play simultaneously
- **THEN** they reinforce each other (not clash rhythmically)

### Requirement: Dynamic Variation
The system SHALL generate varied velocities for expressive dynamics.

#### Scenario: Ghost notes
- **WHEN** generating ghost notes (textural hits)
- **THEN** velocity is 20-40

#### Scenario: Normal hits
- **WHEN** generating normal groove hits
- **THEN** velocity is 60-90

#### Scenario: Accents
- **WHEN** generating accented hits
- **THEN** velocity is 100-127

#### Scenario: Crescendo
- **WHEN** building intensity over bars
- **THEN** velocities gradually increase (e.g., 60 → 70 → 80 → 90)

### Requirement: Hi-Hat Articulation
The system SHALL differentiate between open and closed hi-hat.

#### Scenario: Closed hi-hat
- **WHEN** generating standard hi-hat pattern
- **THEN** MIDI note 42 (Closed Hi-Hat) is used

#### Scenario: Open hi-hat accent
- **WHEN** accenting offbeats
- **THEN** MIDI note 46 (Open Hi-Hat) is used on accent beats

#### Scenario: Hi-hat pedal close
- **WHEN** "chick" sound is needed
- **THEN** MIDI note 44 (Pedal Hi-Hat) is used

### Requirement: Fill Generation
The system SHALL generate drum fills at phrase boundaries.

#### Scenario: Fill at end of 8-bar phrase
- **WHEN** bar 8 of 8-bar phrase is reached
- **THEN** beats 3-4 contain snare/tom fill

#### Scenario: Snare roll fill
- **WHEN** generating simple fill
- **THEN** 16th note snare roll with crescendo (70 → 110 velocity)

#### Scenario: Tom fill
- **WHEN** generating dramatic fill
- **THEN** descending toms (high → mid → low) over 1-2 beats

#### Scenario: Return to groove after fill
- **WHEN** fill completes on beat 4
- **THEN** downbeat of next bar resumes normal groove (often with crash)

### Requirement: Crash Cymbal Placement
The system SHALL place crash cymbals at sectional emphasis points.

#### Scenario: Crash on chorus downbeat
- **WHEN** transitioning from verse to chorus
- **THEN** crash cymbal plays on beat 1 of chorus

#### Scenario: Crash after fill
- **WHEN** fill ends on beat 4
- **THEN** crash cymbal plays on downbeat (beat 1) of next section

#### Scenario: Crash with kick
- **WHEN** crash is placed
- **THEN** kick drum also plays simultaneously for emphasis

### Requirement: Rebracketing Marker Emphasis
The system SHALL emphasize rebracketing markers through rhythmic techniques.

#### Scenario: Crash at marker
- **WHEN** rebracketing marker occurs
- **THEN** crash cymbal plays on marker beat

#### Scenario: Fill leading to marker
- **WHEN** rebracketing marker is at beat 8
- **THEN** fill begins at beat 6 or 7, building to marker

#### Scenario: Rhythmic break before marker
- **WHEN** rebracketing marker represents disruption
- **THEN** drums drop out 1 beat before, re-enter at marker with crash

#### Scenario: Dynamic surge at marker
- **WHEN** marker requires emphasis
- **THEN** velocities increase 10-20 points around marker

### Requirement: Playability Constraints
The system SHALL enforce physical limitations of human drummers.

#### Scenario: Max 4 simultaneous notes
- **WHEN** generating polyphonic pattern
- **THEN** no more than 4 notes sound simultaneously (2 hands, 2 feet)

#### Scenario: No impossible limb combinations
- **WHEN** hi-hat and ride are both present
- **THEN** they do not play simultaneously (same right hand)

#### Scenario: Tempo-based subdivision limits
- **WHEN** tempo > 160 BPM
- **THEN** 32nd note subdivisions are avoided (too fast)

#### Scenario: Recovery after fill
- **WHEN** complex fill finishes
- **THEN** at least 1 beat before next complex pattern (hands reset)

### Requirement: Syncopation and Subdivision
The system SHALL generate rhythmic subdivisions appropriate to style.

#### Scenario: Straight 8th notes
- **WHEN** generating rock or pop groove
- **THEN** hi-hat/ride plays evenly spaced 8th notes

#### Scenario: 16th note subdivisions
- **WHEN** generating funk or R&B groove
- **THEN** hi-hat plays 16th notes with accents

#### Scenario: Swing feel
- **WHEN** generating jazz groove
- **THEN** ride cymbal plays swing 8ths (triplet-based)

#### Scenario: Syncopated anticipations
- **WHEN** generating funk kick
- **THEN** kicks on "and of 2" and "and of 4" (anticipations)

### Requirement: Groove Scoring
The system SHALL score drum patterns on groove quality.

#### Scenario: Tight pocket
- **WHEN** kick aligns with beats 1 and 3, snare on 2 and 4
- **THEN** groove score is high

#### Scenario: Style-appropriate syncopation
- **WHEN** funk pattern uses syncopated kicks
- **THEN** groove score increases

#### Scenario: Stiff/robotic feel
- **WHEN** all hits have identical velocity and timing
- **THEN** groove score decreases

### Requirement: Bass Synchronization Scoring
The system SHALL score kick-bass alignment.

#### Scenario: Perfect kick-bass alignment
- **WHEN** every bass root matches with kick drum
- **THEN** bass sync score is 1.0

#### Scenario: Partial alignment
- **WHEN** some but not all bass roots match kicks
- **THEN** bass sync score is proportional (e.g., 0.7 if 7 of 10 match)

#### Scenario: Kick-bass conflict
- **WHEN** kick and bass play different rhythms that clash
- **THEN** bass sync score decreases significantly

### Requirement: Variety Scoring
The system SHALL score drum patterns on appropriate variety.

#### Scenario: Multi-voice usage
- **WHEN** pattern uses kick, snare, hats, crashes, toms
- **THEN** variety score increases

#### Scenario: Dynamic range
- **WHEN** pattern includes ghost notes and accents
- **THEN** variety score increases

#### Scenario: Over-repetition
- **WHEN** exact same 1-bar loop repeats without variation
- **THEN** variety score decreases

### Requirement: Playability Scoring
The system SHALL score drum patterns on human performability.

#### Scenario: Physically possible
- **WHEN** max 4 simultaneous notes, no limb conflicts
- **THEN** playability score is high

#### Scenario: Impossible coordination
- **WHEN** pattern requires 5+ simultaneous limbs
- **THEN** playability score is 0.0 (reject pattern)

#### Scenario: Tempo feasible
- **WHEN** subdivision is achievable at given tempo
- **THEN** playability score is high

### Requirement: Fill Quality Scoring
The system SHALL score drum fills on musicality and placement.

#### Scenario: Fill at phrase boundary
- **WHEN** fill occurs at end of 4, 8, or 16-bar phrase
- **THEN** fill score increases

#### Scenario: Smooth transition after fill
- **WHEN** groove resumes cleanly after fill
- **THEN** fill score increases

#### Scenario: Excessive fills
- **WHEN** fills occur every 2 bars (too frequent)
- **THEN** fill score decreases

### Requirement: Dynamic Expression Scoring
The system SHALL score drum patterns on expressive dynamics.

#### Scenario: Velocity variation
- **WHEN** pattern includes ghost notes, normal hits, accents
- **THEN** dynamic expression score is high

#### Scenario: Crescendo/diminuendo
- **WHEN** pattern includes gradual velocity changes
- **THEN** dynamic expression score increases

#### Scenario: Flat dynamics
- **WHEN** all hits have same velocity (no variation)
- **THEN** dynamic expression score decreases

### Requirement: Ride Cymbal Patterns
The system SHALL generate ride cymbal patterns as alternative to hi-hat.

#### Scenario: Jazz ride pattern
- **WHEN** generating jazz groove
- **THEN** ride cymbal plays "ding-ding-a-ding" pattern (swing 8ths with triplet)

#### Scenario: Rock ride pattern
- **WHEN** generating rock groove (chorus or intense section)
- **THEN** ride cymbal plays straight 8ths or quarter notes

#### Scenario: Ride bell accent
- **WHEN** accenting ride pattern
- **THEN** MIDI note 53 (Ride Bell) is used on accents

### Requirement: Tom Patterns
The system SHALL generate tom drum patterns for fills and emphasis.

#### Scenario: Descending tom fill
- **WHEN** generating dramatic fill
- **THEN** high tom → mid tom → low tom (MIDI notes 50, 47, 45)

#### Scenario: Ascending tom fill
- **WHEN** generating build-up
- **THEN** low tom → mid tom → high tom

#### Scenario: Tom accents in groove
- **WHEN** adding variation to groove
- **THEN** occasional tom hits replace snare on beat 4

### Requirement: Percussion Additions
The system SHALL support additional percussion instruments (shakers, tambourine, cowbell).

#### Scenario: Tambourine on backbeat
- **WHEN** generating pop/rock with tambourine
- **THEN** MIDI note 54 (Tambourine) plays on beats 2 and 4

#### Scenario: Shaker subdivision
- **WHEN** generating with shaker
- **THEN** MIDI note 69 (Cabasa) or 70 (Maracas) plays 16th notes

#### Scenario: Cowbell accent
- **WHEN** generating funk with cowbell
- **THEN** MIDI note 56 (Cowbell) plays on syncopated beats

### Requirement: Sectional Variation
The system SHALL vary drum patterns between song sections.

#### Scenario: Verse simpler than chorus
- **WHEN** generating verse drum pattern
- **THEN** fewer voices, lower energy (e.g., kick/snare/hats only)

#### Scenario: Chorus intensity
- **WHEN** generating chorus
- **THEN** more voices (add crash, ride, toms), higher velocities

#### Scenario: Bridge contrast
- **WHEN** generating bridge
- **THEN** different groove template (e.g., half-time, different ride pattern)

### Requirement: Build-Up Patterns
The system SHALL generate build-up patterns leading to sectional changes.

#### Scenario: Snare roll build-up
- **WHEN** approaching chorus or drop
- **THEN** snare plays 16th note roll with crescendo over 1-2 bars

#### Scenario: Tom roll build-up
- **WHEN** approaching climax
- **THEN** toms play descending pattern with accelerating rhythm

#### Scenario: Crash cymbal swell
- **WHEN** building to major section change
- **THEN** crash cymbal sustains with crescendo (simulate cymbal swell)

### Requirement: Breakdown Patterns
The system SHALL generate breakdown patterns for sparse, minimal sections.

#### Scenario: Kick-only breakdown
- **WHEN** entering breakdown
- **THEN** only kick drum plays (snare/hats drop out)

#### Scenario: Silence breakdown
- **WHEN** maximum dramatic effect
- **THEN** all drums drop out for 1-2 bars

#### Scenario: Gradual breakdown
- **WHEN** transitioning to breakdown
- **THEN** voices drop out progressively (toms → snare → hats, leaving kick)

### Requirement: Humanization
The system SHALL add subtle timing and velocity variation to simulate human performance.

#### Scenario: Timing variation
- **WHEN** humanization is enabled
- **THEN** each hit is offset by ±5-15ms randomly

#### Scenario: Velocity variation
- **WHEN** humanization is enabled
- **THEN** each hit's velocity varies by ±5-10 randomly

#### Scenario: Natural acceleration
- **WHEN** simulating live drummer
- **THEN** slight tempo drift (±1-2 BPM) over long sections

### Requirement: Brute-Force Drum Search
The system SHALL generate many candidate drum patterns and return top-scoring options.

#### Scenario: Generate N candidates
- **WHEN** num_candidates=300 is specified
- **THEN** 300 drum patterns are generated for the same progression/bass

#### Scenario: Score all candidates
- **WHEN** candidates are generated
- **THEN** each is scored on groove, bass sync, variety, playability, fills, dynamics

#### Scenario: Return top K patterns
- **WHEN** top_k=10 is specified
- **THEN** the 10 highest-scoring patterns are returned with score breakdowns

### Requirement: MIDI Export with Drum Track
The system SHALL export drum patterns as MIDI files on channel 10 (General MIDI drums).

#### Scenario: Drum track on channel 10
- **WHEN** exporting drum pattern
- **THEN** all drum notes are placed on MIDI channel 10

#### Scenario: General MIDI note mapping
- **WHEN** exporting kick drum
- **THEN** MIDI note 36 is used (Acoustic Bass Drum per GM standard)

#### Scenario: Velocity export
- **WHEN** exporting with dynamics
- **THEN** MIDI velocities range from 20 (ghost) to 127 (accent)

#### Scenario: Multi-track sync
- **WHEN** exporting with other instruments
- **THEN** drum track synchronizes with bass, chords, melody tracks

### Requirement: Integration with Chord Progressions
The system SHALL accept chord progressions and emphasize harmonic changes.

#### Scenario: Crash on chord change
- **WHEN** major chord change occurs (e.g., IV → V → I)
- **THEN** crash cymbal plays on downbeat of new chord

#### Scenario: Fill before chord change
- **WHEN** significant harmonic movement approaching
- **THEN** fill plays 1-2 beats before chord change

#### Scenario: Harmonic rhythm alignment
- **WHEN** chord changes every 4 beats
- **THEN** drum pattern emphasizes 4-beat phrase structure

### Requirement: Integration with Melody
The system SHALL accept melody and leave space for vocal/melodic climax.

#### Scenario: Detect melody phrase endings
- **WHEN** melody phrase ends
- **THEN** drum fill may be placed at phrase ending

#### Scenario: Reduce crash during vocal climax
- **WHEN** melody reaches highest/most important note
- **THEN** crash cymbal is avoided (don't compete with melody)

#### Scenario: Support melody rhythm
- **WHEN** melody has syncopated rhythm
- **THEN** drums support (not clash with) melody rhythm

### Requirement: Configurable Scoring Weights
The system SHALL allow custom weighting of drum scoring criteria.

#### Scenario: Emphasize groove
- **WHEN** weights={"groove": 0.4, "bass_sync": 0.3, "variety": 0.2, "playability": 0.1}
- **THEN** patterns with tight pocket score highest

#### Scenario: Emphasize bass synchronization
- **WHEN** weights={"bass_sync": 0.5, "groove": 0.3, "fills": 0.2}
- **THEN** patterns locking with bass score highest

### Requirement: Pattern Repetition and Evolution
The system SHALL support repeating patterns with variation.

#### Scenario: Verse 1 = Verse 2 pattern
- **WHEN** generating verse 2
- **THEN** verse 1 drum pattern can be reused

#### Scenario: Variation on repeat
- **WHEN** pattern repeats
- **THEN** slight variation (add ghost notes, change fill) is optionally applied

#### Scenario: Chorus intensification
- **WHEN** chorus repeats
- **THEN** second chorus may be busier (add toms, extra crashes)

### Requirement: Time Signature Support
The system SHALL generate patterns in various time signatures.

#### Scenario: 4/4 time
- **WHEN** time_signature="4/4"
- **THEN** patterns emphasize beats 1 and 3 (kick), 2 and 4 (snare)

#### Scenario: 3/4 time (waltz)
- **WHEN** time_signature="3/4"
- **THEN** patterns emphasize beat 1 (kick), beat 2 (snare), beat 3 (hats)

#### Scenario: 6/8 time
- **WHEN** time_signature="6/8"
- **THEN** patterns emphasize beats 1 and 4 (kick), beat 4 (snare)

#### Scenario: 7/8 or 5/4 (odd meters)
- **WHEN** time_signature is odd meter
- **THEN** patterns group beats logically (e.g., 7/8 as 3+2+2 or 2+2+3)

### Requirement: Tempo-Dependent Generation
The system SHALL adapt patterns to tempo.

#### Scenario: Slow tempo (< 80 BPM)
- **WHEN** tempo is slow
- **THEN** 16th note subdivisions are used more (create motion)

#### Scenario: Fast tempo (> 160 BPM)
- **WHEN** tempo is fast
- **THEN** simpler subdivisions (8ths, quarters) to maintain playability

#### Scenario: Double-time feel
- **WHEN** tempo is moderate but energy needs increase
- **THEN** hi-hat plays double-time (16ths) while kick/snare stay half-time
