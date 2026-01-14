# Chord Progression Generator

## Requirements

### Requirement: MIDI Corpus Database
The system SHALL parse a MIDI chord library into structured Polars DataFrames for efficient querying and analysis.

#### Scenario: Parse individual chords
- **WHEN** MIDI files in the chords directory are parsed
- **THEN** each chord is extracted with metadata (root, quality, function, MIDI notes, intervals)

#### Scenario: Parse chord progressions
- **WHEN** MIDI progression files are parsed
- **THEN** sequences of chords are extracted with timing and position information

#### Scenario: Store in Polars DataFrames
- **WHEN** chord data is processed
- **THEN** columnar DataFrames enable fast filtering by key, mode, function, quality

### Requirement: Chord Transition Graphs
The system SHALL build NetworkX directed graphs representing common chord transitions learned from the corpus.

#### Scenario: Build chord-to-chord graph
- **WHEN** analyzing progressions
- **THEN** a graph maps chord pairs (e.g., C_major → G_major) with transition probabilities

#### Scenario: Build function-to-function graph
- **WHEN** analyzing progressions with roman numeral functions
- **THEN** a graph maps function transitions (e.g., I → IV, IV → V) with probabilities

#### Scenario: Weight by frequency
- **WHEN** multiple progressions use the same transition
- **THEN** edge weights accumulate and normalize to probabilities

### Requirement: Random Chord Generation
The system SHALL generate random chord progressions from a specified key and mode.

#### Scenario: Generate in specific key
- **WHEN** generating a random progression in C Major
- **THEN** only chords in C Major are selected

#### Scenario: Filter by category
- **WHEN** generating with category="triad"
- **THEN** only triadic chords (not 7ths, 9ths, etc.) are selected

#### Scenario: Specified length
- **WHEN** generating with length=4
- **THEN** exactly 4 chords are returned

### Requirement: Graph-Guided Generation
The system SHALL generate progressions following learned transition probabilities from the corpus.

#### Scenario: Start from specific function
- **WHEN** generating with start_function="I"
- **THEN** the first chord is a I chord (tonic)

#### Scenario: Follow transition probabilities
- **WHEN** selecting each subsequent chord
- **THEN** the function transition graph provides weighted sampling of likely next chords

#### Scenario: Fallback to random
- **WHEN** the graph has no valid next transitions
- **THEN** a random chord from the key is selected

### Requirement: Multi-Criteria Scoring
The system SHALL score generated progressions on multiple musical dimensions.

#### Scenario: Melody scoring
- **WHEN** scoring melodic movement (highest notes)
- **THEN** stepwise motion scores higher than large leaps

#### Scenario: Voice leading scoring
- **WHEN** scoring voice leading quality
- **THEN** smaller total voice movement between chords scores higher

#### Scenario: Variety scoring
- **WHEN** scoring chord variety
- **THEN** more unique chords (less repetition) scores higher

#### Scenario: Graph probability scoring
- **WHEN** scoring against corpus statistics
- **THEN** transitions common in the corpus score higher

#### Scenario: Weighted combination
- **WHEN** computing total score
- **THEN** individual scores are combined with configurable weights

### Requirement: Brute-Force Search with Top-K Selection
The system SHALL generate many candidate progressions and return the highest-scoring ones.

#### Scenario: Generate N candidates
- **WHEN** num_candidates=1000 is specified
- **THEN** 1000 random or graph-guided progressions are generated

#### Scenario: Score all candidates
- **WHEN** candidates are generated
- **THEN** each is scored using the multi-criteria scoring function

#### Scenario: Return top K
- **WHEN** top_k=10 is specified
- **THEN** the 10 highest-scoring progressions are returned with scores and breakdowns

#### Scenario: Score breakdown transparency
- **WHEN** returning top-K results
- **THEN** each result includes individual scores (melody, voice_leading, variety, graph_prob)

### Requirement: Configurable Scoring Weights
The system SHALL allow custom weighting of scoring criteria to emphasize different musical priorities.

#### Scenario: Emphasize voice leading
- **WHEN** weights={"voice_leading": 0.5, "melody": 0.2, "variety": 0.1, "graph_probability": 0.2}
- **THEN** smooth voice leading is prioritized over other factors

#### Scenario: Emphasize corpus fidelity
- **WHEN** weights={"graph_probability": 0.6, ...}
- **THEN** progressions matching corpus patterns score higher

#### Scenario: Custom aesthetic
- **WHEN** custom weights are provided
- **THEN** scoring reflects the specified aesthetic priorities

### Requirement: Key and Mode Support
The system SHALL generate progressions in any major or minor key present in the corpus.

#### Scenario: Major keys
- **WHEN** generating in C Major
- **THEN** chords are filtered to mode_in_key="Major"

#### Scenario: Minor keys
- **WHEN** generating in A Minor
- **THEN** chords are filtered to mode_in_key="Minor"

#### Scenario: Available keys
- **WHEN** querying available keys
- **THEN** all keys present in the corpus are listed

### Requirement: MIDI Note Extraction
The system SHALL provide MIDI note lists for each chord in generated progressions.

#### Scenario: Convert to MIDI notes
- **WHEN** a progression is generated
- **THEN** each chord includes its MIDI note list for playback or export

#### Scenario: Chord voicing preserved
- **WHEN** chords are sampled from the database
- **THEN** the specific voicing (note arrangement) from the corpus is maintained

### Requirement: Function-Based Querying
The system SHALL support querying chords by roman numeral function within a key.

#### Scenario: Get all I chords in C Major
- **WHEN** querying function="I" in C Major
- **THEN** all tonic chords (C, Cmaj7, Cadd9, etc.) are returned

#### Scenario: Get dominants
- **WHEN** querying function="V" in G Major
- **THEN** all dominant chords (D, D7, Dsus4, etc.) are returned

#### Scenario: Sample by function
- **WHEN** sampling a random chord by function
- **THEN** one chord with that function is randomly selected

### Requirement: Progression Display
The system SHALL provide human-readable printing of generated progressions.

#### Scenario: Print progression
- **WHEN** printing a progression
- **THEN** each chord shows: position, function, chord name, note names

#### Scenario: Format consistency
- **WHEN** displaying multiple progressions
- **THEN** consistent formatting enables easy comparison

### Requirement: Corpus Statistics
The system SHALL compute and expose statistics about the chord library.

#### Scenario: Count total chords
- **WHEN** querying database size
- **THEN** total number of unique chords is returned

#### Scenario: List qualities
- **WHEN** querying chord qualities
- **THEN** distribution of major, minor, diminished, augmented, etc. is provided

#### Scenario: List functions
- **WHEN** querying functions
- **THEN** distribution of I, ii, iii, IV, V, vi, vii° is provided

### Requirement: Graph Persistence
The system SHALL save and load transition graphs for efficient reuse.

#### Scenario: Save graphs as pickle
- **WHEN** building graphs from corpus
- **THEN** graphs are serialized to .pkl files

#### Scenario: Load graphs on initialization
- **WHEN** ChordProgressionGenerator is instantiated
- **THEN** pre-built graphs are loaded from disk

### Requirement: Generation Mode Selection
The system SHALL support multiple generation strategies.

#### Scenario: Pure random generation
- **WHEN** use_graph=False
- **THEN** chords are sampled uniformly at random from the key

#### Scenario: Graph-guided generation
- **WHEN** use_graph=True
- **THEN** the function transition graph guides chord selection

#### Scenario: Hybrid approach
- **WHEN** brute-force generates with use_graph=True
- **THEN** graph-guided candidates are scored and ranked
