# Melody Generator

## ADDED Requirements

### Requirement: Syllable-Based Lyrics Parsing
The system SHALL parse lyrics into syllables with stress patterns for rhythm generation.

#### Scenario: Extract syllables from text
- **WHEN** lyrics "Beautiful ladies they're solving mysteries" are parsed
- **THEN** syllables ["Beau", "ti", "ful", "la", "dies", "they're", "sol", "ving", "mys", "ter", "ies"] are extracted

#### Scenario: Detect stress patterns
- **WHEN** syllables are analyzed for stress
- **THEN** stressed syllables are marked (e.g., "BEAU-ti-ful LA-dies")

#### Scenario: Identify phrase boundaries
- **WHEN** lyrics contain punctuation or line breaks
- **THEN** phrase boundaries are marked for musical phrasing

### Requirement: Rebracketing Marker Detection
The system SHALL identify rebracketing markers in lyrics for melodic emphasis.

#### Scenario: Detect bracketed markers
- **WHEN** lyrics contain "The room [(is not)] a container"
- **THEN** "is not" is identified as a rebracketing marker

#### Scenario: Mark emphasis points
- **WHEN** rebracketing markers are found
- **THEN** corresponding syllables are tagged for melodic prominence

### Requirement: Prosody-Based Rhythm Generation
The system SHALL generate rhythms from lyric prosody and stress patterns.

#### Scenario: Stressed syllables on strong beats
- **WHEN** generating rhythm from stressed/unstressed patterns
- **THEN** stressed syllables land on downbeats or strong beats

#### Scenario: Unstressed syllables as pickup
- **WHEN** a phrase begins with unstressed syllables
- **THEN** they are placed as anacrusis (pickup notes) before the bar

#### Scenario: Natural speech rhythm
- **WHEN** converting prosody to rhythm
- **THEN** the result mimics natural speech patterns

### Requirement: Melodic Contour Templates
The system SHALL generate melodic contours using common shape templates.

#### Scenario: Arch contour
- **WHEN** generating with arch template
- **THEN** melody ascends to a peak and descends

#### Scenario: Descending contour
- **WHEN** generating with descending template
- **THEN** melody starts high and gradually lowers

#### Scenario: Wave contour
- **WHEN** generating with wave template
- **THEN** melody alternates between highs and lows in a wave pattern

### Requirement: Harmonic Alignment
The system SHALL align melodies to underlying chord progressions.

#### Scenario: Chord tones on strong beats
- **WHEN** a strong beat coincides with a chord
- **THEN** the melody note is preferentially a chord tone

#### Scenario: Passing tones between chord tones
- **WHEN** melody moves between chord tones
- **THEN** non-chord tones are used as stepwise passing tones

#### Scenario: Neighbor tones for embellishment
- **WHEN** adding melodic embellishment
- **THEN** neighbor tones (upper or lower) decorate chord tones

### Requirement: Rebracketing Marker Melodic Emphasis
The system SHALL emphasize rebracketing markers through melodic techniques.

#### Scenario: Pitch elevation at markers
- **WHEN** a rebracketing marker occurs
- **THEN** the melody rises in pitch relative to surrounding notes

#### Scenario: Duration extension at markers
- **WHEN** a rebracketing marker occurs
- **THEN** note durations are lengthened for emphasis

#### Scenario: Harmonic emphasis
- **WHEN** a rebracketing marker occurs
- **THEN** the melody aligns with a chord tone for stability

#### Scenario: Contour disruption
- **WHEN** a rebracketing marker represents ontological shift
- **THEN** the melodic contour changes direction or introduces a leap

### Requirement: Vocal Range Constraints
The system SHALL enforce singable vocal ranges appropriate for human performers.

#### Scenario: Configurable range limits
- **WHEN** generating for soprano voice
- **THEN** melody stays within C4-C6 range

#### Scenario: Comfortable tessitura
- **WHEN** analyzing generated melody
- **THEN** most notes fall within a comfortable octave-and-fifth range

#### Scenario: Avoid extreme notes
- **WHEN** melody approaches range limits
- **THEN** it turns back rather than exceeding limits

### Requirement: Stepwise Motion Preference
The system SHALL prefer stepwise melodic motion over large leaps.

#### Scenario: Stepwise motion scores higher
- **WHEN** scoring melodic contours
- **THEN** melodies moving by steps (1-2 semitones) score higher than leaps

#### Scenario: Leap recovery
- **WHEN** a melodic leap occurs (interval > 4 semitones)
- **THEN** the next interval moves stepwise in the opposite direction

#### Scenario: Avoid awkward intervals
- **WHEN** generating intervals
- **THEN** augmented and diminished intervals are avoided or resolved

### Requirement: Breathing Points
The system SHALL insert breathing points at natural phrase boundaries.

#### Scenario: Rest at phrase end
- **WHEN** a phrase ends (punctuation, line break)
- **THEN** a rest is inserted for breathing

#### Scenario: Maximum phrase length
- **WHEN** a phrase exceeds typical breath capacity (e.g., 8 beats at moderate tempo)
- **THEN** a breathing point is inserted

### Requirement: Climax Placement
The system SHALL place the melodic climax (highest note) at a meaningful structural point.

#### Scenario: Climax near phrase end
- **WHEN** generating a phrase
- **THEN** the highest note occurs in the final third of the phrase

#### Scenario: Climax on important word
- **WHEN** lyrics contain emotionally important words
- **THEN** the climax aligns with those words

#### Scenario: Single climax per section
- **WHEN** generating a verse or chorus
- **THEN** there is one primary climax rather than multiple competing peaks

### Requirement: Verse-Chorus Melodic Contrast
The system SHALL create melodic contrast between verse and chorus sections.

#### Scenario: Chorus higher tessitura
- **WHEN** generating chorus melody
- **THEN** average pitch is higher than verse

#### Scenario: Chorus more rhythmic activity
- **WHEN** generating chorus
- **THEN** rhythmic density (notes per beat) is higher than verse

#### Scenario: Verse melodic restraint
- **WHEN** generating verse
- **THEN** melody uses narrower range and simpler rhythms

### Requirement: Call-and-Response Patterns
The system SHALL generate call-and-response melodic structures.

#### Scenario: Question-answer phrasing
- **WHEN** generating paired phrases
- **THEN** first phrase ends on non-tonic note (question), second on tonic (answer)

#### Scenario: Melodic echoing
- **WHEN** implementing call-and-response
- **THEN** the response mirrors or inverts the call's contour

### Requirement: Cadence Generation
The system SHALL generate melodic cadences that match phrase structure.

#### Scenario: Authentic cadence
- **WHEN** phrase ends on tonic chord
- **THEN** melody resolves to the tonic note

#### Scenario: Half cadence
- **WHEN** phrase ends on dominant chord
- **THEN** melody ends on a dominant chord tone (question feel)

#### Scenario: Deceptive cadence
- **WHEN** subverting expectations
- **THEN** melody resolves to unexpected note (ontological disruption)

### Requirement: Melodic Scoring Functions
The system SHALL score generated melodies on multiple dimensions.

#### Scenario: Contour smoothness
- **WHEN** scoring melodic smoothness
- **THEN** stepwise motion and gradual contours score higher

#### Scenario: Harmonic fit
- **WHEN** scoring harmonic alignment
- **THEN** chord tones on strong beats score higher

#### Scenario: Range appropriateness
- **WHEN** scoring vocal range
- **THEN** melodies within comfortable range score higher

#### Scenario: Variety
- **WHEN** scoring melodic interest
- **THEN** diverse rhythms and intervals score higher than repetition

### Requirement: Brute-Force Melody Search
The system SHALL generate many candidate melodies and return top-scoring options.

#### Scenario: Generate N candidates
- **WHEN** num_candidates=500 is specified
- **THEN** 500 melodies are generated for the same lyrics and chords

#### Scenario: Score all candidates
- **WHEN** candidates are generated
- **THEN** each is scored on all criteria

#### Scenario: Return top K melodies
- **WHEN** top_k=10 is specified
- **THEN** the 10 highest-scoring melodies are returned with score breakdowns

### Requirement: MIDI Export with Embedded Lyrics
The system SHALL export melodies as MIDI files with lyrics embedded.

#### Scenario: Melody as MIDI notes
- **WHEN** exporting melody
- **THEN** each note is written as a MIDI Note On/Off pair with correct timing

#### Scenario: Lyrics as MIDI meta-events
- **WHEN** exporting lyrics
- **THEN** MIDI Lyric meta-events are inserted at corresponding note times

#### Scenario: Multi-track export
- **WHEN** exporting with chord progression
- **THEN** MIDI file contains separate tracks for melody and chords

#### Scenario: Tempo and time signature
- **WHEN** exporting MIDI
- **THEN** tempo and time signature meta-events are included

### Requirement: Integration with Chord Progressions
The system SHALL accept chord progressions from ChordProgressionGenerator as input.

#### Scenario: Align melody timing to chords
- **WHEN** generating melody over chord progression
- **THEN** melody phrases align with chord change boundaries

#### Scenario: Use chord tones
- **WHEN** generating melody notes
- **THEN** chord tones from the current chord are preferred on strong beats

#### Scenario: Synchronize durations
- **WHEN** a chord lasts 4 beats
- **THEN** melody notes within that duration reference that chord's tonality

### Requirement: Configurable Scoring Weights
The system SHALL allow custom weighting of melodic scoring criteria.

#### Scenario: Emphasize harmonic fit
- **WHEN** weights={"harmonic_fit": 0.5, "smoothness": 0.2, "variety": 0.15, "range": 0.15}
- **THEN** melodies fitting chords strongly are prioritized

#### Scenario: Emphasize singability
- **WHEN** weights={"range": 0.4, "smoothness": 0.4, "variety": 0.2}
- **THEN** easy-to-sing melodies score higher

### Requirement: Prosody-Rhythm Validation
The system SHALL validate that generated rhythms match lyric prosody.

#### Scenario: No misplaced stresses
- **WHEN** validating rhythm-prosody alignment
- **THEN** stressed syllables do not land on weak beats

#### Scenario: Natural speech flow
- **WHEN** listening to generated rhythm
- **THEN** it mimics natural speech cadence

### Requirement: Melodic Motif Repetition
The system SHALL support repeating melodic motifs across phrases.

#### Scenario: Verse motif
- **WHEN** generating multiple verses
- **THEN** melodic motif from first verse can be repeated with variation

#### Scenario: Chorus hook
- **WHEN** generating chorus
- **THEN** a memorable melodic hook is emphasized and repeated
