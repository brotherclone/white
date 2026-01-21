# Bass Line Generator

## ADDED Requirements

### Requirement: Chord Tone Extraction
The system SHALL extract chord tones (root, third, fifth, seventh, extensions) from chord symbols.

#### Scenario: Extract triad tones
- **WHEN** chord is "Cmaj"
- **THEN** chord tones are [C, E, G]

#### Scenario: Extract seventh chord tones
- **WHEN** chord is "Am7"
- **THEN** chord tones are [A, C, E, G]

#### Scenario: Extract extended chord tones
- **WHEN** chord is "Cmaj9"
- **THEN** chord tones are [C, E, G, B, D]

### Requirement: Root Note Placement
The system SHALL place bass root notes on chord change boundaries.

#### Scenario: Root on chord change
- **WHEN** chord changes from C to Am at beat 5
- **THEN** bass plays A (root of Am) at beat 5

#### Scenario: Root octave selection
- **WHEN** placing root note
- **THEN** octave is selected within playable bass range (E1-G3)

### Requirement: Chord Tone Bass Patterns
The system SHALL generate bass lines using chord tones (roots, fifths, thirds, sevenths).

#### Scenario: Root-fifth pattern
- **WHEN** generating with "root_fifth" pattern over Cmaj
- **THEN** bass alternates C and G

#### Scenario: Arpeggiated pattern
- **WHEN** generating with "arpeggio" pattern over Am7
- **THEN** bass cycles through [A, C, E, G] or subset

#### Scenario: Root emphasis
- **WHEN** generating any pattern
- **THEN** roots appear on strong beats more frequently than other chord tones

### Requirement: Walking Bass Generation
The system SHALL generate walking bass lines with stepwise chromatic motion between chords.

#### Scenario: Chromatic approach from below
- **WHEN** next chord is Cmaj and current bass is on G
- **THEN** bass walks G → A → B → C (stepwise ascent)

#### Scenario: Chromatic approach tone
- **WHEN** approaching C from below
- **THEN** B (chromatic leading tone) may be inserted on weak beat

#### Scenario: Diatonic passing tones
- **WHEN** walking from C to G in C major
- **THEN** passing tones D, E, F may be used

### Requirement: Groove Pattern Templates
The system SHALL provide style-specific groove templates (funk, rock, jazz, R&B).

#### Scenario: Funk groove
- **WHEN** generating with style="funk"
- **THEN** 16th note syncopations and ghost notes are included

#### Scenario: Rock groove
- **WHEN** generating with style="rock"
- **THEN** driving eighth note root-fifth patterns are used

#### Scenario: Jazz walking
- **WHEN** generating with style="jazz"
- **THEN** quarter note walking bass with chromatic approaches is used

### Requirement: Syncopation and Rhythmic Variation
The system SHALL generate syncopated rhythms appropriate to musical style.

#### Scenario: Anticipation
- **WHEN** chord changes on beat 1
- **THEN** bass may anticipate by playing root on "and of 4" (half-beat early)

#### Scenario: Ghost notes
- **WHEN** generating funk or R&B bass
- **THEN** muted/low-velocity ghost notes are added for rhythmic texture

#### Scenario: Rests for groove
- **WHEN** generating rhythmic bass
- **THEN** rests are strategically placed to create groove pocket

### Requirement: Melody Integration
The system SHALL avoid harsh intervals with generated melody.

#### Scenario: No minor 2nd clashes
- **WHEN** melody is on C5 and bass candidate is B1
- **THEN** bass adjusts to C2 or other chord tone to avoid minor 2nd

#### Scenario: No major 7th clashes
- **WHEN** melody is on B4 and bass candidate is C1
- **THEN** bass adjusts to avoid major 7th interval

#### Scenario: Counterpoint preference
- **WHEN** melody ascends
- **THEN** bass preferentially descends (contrary motion)

### Requirement: Playability Constraints
The system SHALL enforce bass instrument range and fingering limitations.

#### Scenario: Range enforcement
- **WHEN** generating for 4-string bass
- **THEN** all notes fall within E1-G3 range

#### Scenario: Position shift limits
- **WHEN** consecutive notes require position change
- **THEN** maximum 5-fret shift per beat is enforced

#### Scenario: Open string preference
- **WHEN** note is E, A, D, or G
- **THEN** open string is preferentially used (if rhythmically appropriate)

### Requirement: Rebracketing Marker Emphasis
The system SHALL emphasize rebracketing markers through bass techniques.

#### Scenario: Low register drop
- **WHEN** rebracketing marker occurs
- **THEN** bass drops to lowest comfortable register (E1-G1)

#### Scenario: Pedal tone through marker
- **WHEN** rebracketing marker spans multiple beats
- **THEN** bass sustains root note as pedal tone

#### Scenario: Rhythmic density shift
- **WHEN** approaching rebracketing marker
- **THEN** rhythmic subdivision increases (quarter → eighth → sixteenth)

#### Scenario: Chromatic tension before marker
- **WHEN** marker is at beat 8
- **THEN** chromatic approach or tritone substitution at beat 7

### Requirement: Harmonic Fit Scoring
The system SHALL score bass lines on harmonic alignment with chords.

#### Scenario: Chord tone on strong beat
- **WHEN** strong beat (1, 3) has chord tone
- **THEN** harmonic fit score increases

#### Scenario: Passing tone on weak beat
- **WHEN** weak beat has non-chord tone as passing tone
- **THEN** harmonic fit score is neutral (acceptable)

#### Scenario: Non-harmonic tone on strong beat
- **WHEN** strong beat has non-chord tone without resolution
- **THEN** harmonic fit score decreases

### Requirement: Rhythmic Coherence Scoring
The system SHALL score bass lines on rhythmic consistency and groove.

#### Scenario: Consistent subdivision
- **WHEN** bass uses same subdivision (e.g., all eighth notes) throughout section
- **THEN** rhythmic coherence score increases

#### Scenario: Style-appropriate syncopation
- **WHEN** funk bass uses anticipated roots and ghost notes
- **THEN** rhythmic coherence score increases

#### Scenario: Chaotic rhythm
- **WHEN** bass rhythm changes randomly without musical reason
- **THEN** rhythmic coherence score decreases

### Requirement: Playability Scoring
The system SHALL score bass lines on human performer feasibility.

#### Scenario: Within range
- **WHEN** all notes are E1-G3
- **THEN** playability score is high

#### Scenario: Feasible position shifts
- **WHEN** no position shift exceeds 5 frets per beat
- **THEN** playability score is high

#### Scenario: Impossible fingering
- **WHEN** rapid wide intervals require impossible hand positions
- **THEN** playability score decreases

### Requirement: Melody Support Scoring
The system SHALL score bass lines on how well they support the melody.

#### Scenario: No interval clashes
- **WHEN** all bass-melody intervals avoid m2 and M7
- **THEN** melody support score is high

#### Scenario: Good counterpoint
- **WHEN** bass and melody exhibit contrary motion
- **THEN** melody support score increases

#### Scenario: Harmonic reinforcement
- **WHEN** bass plays root while melody plays chord tone
- **THEN** melody support score increases

### Requirement: Groove Scoring
The system SHALL score bass lines on rhythmic feel and pocket.

#### Scenario: Syncopation in funk
- **WHEN** funk bass uses anticipations and ghost notes
- **THEN** groove score increases

#### Scenario: Driving eighths in rock
- **WHEN** rock bass maintains steady eighth note pulse
- **THEN** groove score increases

#### Scenario: Stiff/robotic feel
- **WHEN** bass lacks rhythmic variation or syncopation
- **THEN** groove score decreases

### Requirement: Walking Bass Smoothness
The system SHALL prefer stepwise motion in walking bass patterns.

#### Scenario: Stepwise motion
- **WHEN** walking bass moves by whole or half steps
- **THEN** smoothness score increases

#### Scenario: Chromatic fills
- **WHEN** walking bass uses chromatic approach tones
- **THEN** smoothness score increases

#### Scenario: Large leaps
- **WHEN** walking bass leaps more than a fifth
- **THEN** smoothness score decreases

### Requirement: Pedal Tone Detection
The system SHALL identify opportunities for pedal tones (sustained bass under changing harmony).

#### Scenario: Static root under chord changes
- **WHEN** chords change but share common root (e.g., C → Cmaj7 → C7)
- **THEN** bass sustains C as pedal tone

#### Scenario: Dominant pedal
- **WHEN** progression builds tension over several bars
- **THEN** bass may sustain dominant note (e.g., G in C major)

#### Scenario: Pedal release
- **WHEN** harmonic tension resolves
- **THEN** pedal tone releases to new root

### Requirement: Chromatic Approach Tones
The system SHALL generate chromatic leading tones approaching target notes.

#### Scenario: Half-step below target
- **WHEN** target root is C
- **THEN** B may be placed on weak beat before C

#### Scenario: Half-step above target
- **WHEN** target root is A
- **THEN** B♭ may approach from above

#### Scenario: Double chromatic (encircling)
- **WHEN** target is emphasized
- **THEN** both chromatic neighbors may surround target (C → B → C# → C)

### Requirement: Articulation and Dynamics
The system SHALL generate varied articulations for musical expression.

#### Scenario: Staccato notes
- **WHEN** generating percussive bass
- **THEN** short note durations (< 50% of beat) are marked

#### Scenario: Legato passages
- **WHEN** generating smooth bass
- **THEN** note durations overlap slightly (portamento)

#### Scenario: Ghost notes
- **WHEN** generating funk/R&B
- **THEN** low-velocity muted notes are marked as ghost notes

#### Scenario: Accents
- **WHEN** emphasizing downbeats or syncopations
- **THEN** high-velocity notes are marked as accents

### Requirement: Style-Specific Generation
The system SHALL adapt bass patterns to specified musical style.

#### Scenario: Rock style
- **WHEN** style="rock" is specified
- **THEN** root-fifth eighth note patterns with minimal syncopation are generated

#### Scenario: Funk style
- **WHEN** style="funk" is specified
- **THEN** 16th note syncopations with ghost notes and slap markers are generated

#### Scenario: Jazz style
- **WHEN** style="jazz" is specified
- **THEN** walking quarter notes with chromatic approaches are generated

#### Scenario: R&B style
- **WHEN** style="R&B" is specified
- **THEN** smooth, melodic bass with syncopated anticipations is generated

### Requirement: Harmonic Function Analysis
The system SHALL analyze chord progressions for functional harmonic movement.

#### Scenario: Detect cadences
- **WHEN** progression is IV → V → I
- **THEN** bass emphasizes strong root motion (down fifth, up fourth)

#### Scenario: Circle of fifths
- **WHEN** progression follows circle of fifths (vi → ii → V → I)
- **THEN** bass emphasizes descending fifth motion

#### Scenario: Weak function movement
- **WHEN** chords move without strong functional relationship
- **THEN** bass uses chromatic approaches or passing chords

### Requirement: Variety Scoring
The system SHALL score bass lines on appropriate repetition vs. variation.

#### Scenario: Balanced chord tone usage
- **WHEN** bass uses mix of roots, fifths, thirds (not just roots)
- **THEN** variety score increases

#### Scenario: Rhythmic variation
- **WHEN** bass pattern changes across phrases
- **THEN** variety score increases

#### Scenario: Over-repetition
- **WHEN** bass plays identical pattern for entire song
- **THEN** variety score decreases

### Requirement: Brute-Force Bass Search
The system SHALL generate many candidate bass lines and return top-scoring options.

#### Scenario: Generate N candidates
- **WHEN** num_candidates=500 is specified
- **THEN** 500 bass lines are generated for the same chord progression

#### Scenario: Score all candidates
- **WHEN** candidates are generated
- **THEN** each is scored on all criteria (harmonic fit, rhythm, playability, melody support, groove, variety)

#### Scenario: Return top K bass lines
- **WHEN** top_k=10 is specified
- **THEN** the 10 highest-scoring bass lines are returned with score breakdowns

### Requirement: MIDI Export with Bass Track
The system SHALL export bass lines as MIDI files with appropriate patches and articulations.

#### Scenario: Bass track creation
- **WHEN** exporting bass line
- **THEN** separate MIDI track is created for bass

#### Scenario: Bass patch selection
- **WHEN** exporting with style="fingered"
- **THEN** MIDI patch 33 (fingered bass) is used

#### Scenario: Velocity mapping
- **WHEN** exporting with ghost notes and accents
- **THEN** velocity values map appropriately (ghost=40-60, normal=80-100, accent=110-127)

#### Scenario: Articulation markers
- **WHEN** exporting slides or hammer-ons
- **THEN** pitch bend or MIDI control change events are added

### Requirement: Integration with Chord Progressions
The system SHALL accept `ChordProgressionGenerator` output as input.

#### Scenario: Parse chord progression
- **WHEN** receiving chord progression with timing
- **THEN** harmonic rhythm (chord change points) is extracted

#### Scenario: Synchronize timing
- **WHEN** chords last 4 beats each
- **THEN** bass patterns align with 4-beat boundaries

#### Scenario: Extract chord tones
- **WHEN** chord is Am7 at bar 2
- **THEN** bass knows available chord tones [A, C, E, G] for bar 2

### Requirement: Integration with Melody
The system SHALL accept `MelodyGenerator` output and avoid clashes.

#### Scenario: Analyze melody timing
- **WHEN** receiving melody with note timing
- **THEN** bass identifies simultaneous sounding moments

#### Scenario: Check intervals
- **WHEN** melody and bass sound simultaneously
- **THEN** interval quality is checked (avoid m2, M7)

#### Scenario: Adjust bass for melody
- **WHEN** clash is detected
- **THEN** bass note is shifted to different chord tone or octave

### Requirement: Configurable Scoring Weights
The system SHALL allow custom weighting of bass scoring criteria.

#### Scenario: Emphasize harmonic fit
- **WHEN** weights={"harmonic_fit": 0.4, "groove": 0.3, "playability": 0.2, "melody_support": 0.1}
- **THEN** bass lines fitting chords strongly are prioritized

#### Scenario: Emphasize groove
- **WHEN** weights={"groove": 0.5, "harmonic_fit": 0.2, "variety": 0.2, "playability": 0.1}
- **THEN** rhythmically interesting bass lines score higher

### Requirement: Pattern Repetition and Variation
The system SHALL support repeating bass patterns with controlled variation.

#### Scenario: Verse pattern repetition
- **WHEN** generating verse 1 and verse 2
- **THEN** same bass pattern can be reused with optional variation

#### Scenario: Chorus intensification
- **WHEN** generating chorus
- **THEN** rhythmic density or register can increase relative to verse

#### Scenario: Bridge contrast
- **WHEN** generating bridge
- **THEN** different pattern style is used for contrast

### Requirement: Build-Up and Breakdown
The system SHALL support gradual density changes for musical dynamics.

#### Scenario: Build-up pattern
- **WHEN** approaching chorus or climax
- **THEN** rhythmic subdivision increases (quarter → eighth → sixteenth)

#### Scenario: Breakdown pattern
- **WHEN** entering breakdown or intro
- **THEN** rhythmic density decreases (sixteenth → eighth → quarter)

#### Scenario: Sustained build
- **WHEN** build-up spans multiple bars
- **THEN** density increases gradually, not abruptly
