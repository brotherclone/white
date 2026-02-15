# MIDI Chord Library Prototype

A prototype system for transforming MIDI chord progressions into a queryable database using Polars/Parquet with NetworkX graph indices. Designed for brute-force + scoring generation of chord progressions, bass lines, and melodies.

## Overview

This prototype parses **2,746 MIDI files** containing:
- Individual chords (triads, extended chords, modal chords)
- Chord progressions across all 12 keys
- Major and minor modes

The data is structured for:
1. **Fast random sampling** (Polars/Parquet columnar storage)
2. **Music theory relationships** (NetworkX graphs for chord transitions)
3. **Brute-force generation** with **scoring functions**

## Data Structure

### Parquet Files

**`chords.parquet`** (1,594 chords)
- Columns: `chord_id`, `key_root`, `key_quality`, `mode_in_key`, `function`, `chord_name`, `quality`, `category`, `root_note`, `bass_note`, `midi_notes`, `intervals`, `note_names`, etc.
- Optimized for filtering and sampling by key, function, quality, or category

**`progressions.parquet`** (9,104 chord entries from 1,152 progressions)
- Flattened progression data with position indices
- Columns: `progression_id`, `progression_name`, `key_root`, `position`, `total_chords`, `root_note`, `quality`, `midi_notes`, `intervals`, `duration_ticks`, etc.

### NetworkX Graphs

**Chord Transition Graph** (`chord_transition_graph.pkl`)
- Nodes: Individual chords (e.g., `48_maj7` = C maj7)
- Edges: Transitions between chords with frequency weights
- 22 nodes, 247 edges

**Function Transition Graph** (`function_transition_graph.pkl`)
- Nodes: (Key, Function) pairs (e.g., `C_Major:I`, `C_Major:IV`)
- Edges: Probability-weighted transitions (I → IV, IV → V, etc.)
- 93 nodes, 588 edges
- **Most useful for music-theory-based generation**

## Usage

### 1. Build the Database

Parse all MIDI files and create the Parquet/graph database:

```bash
python -m app.generators.midi.prototype.build_database
```

Output:
- `data/chords.parquet`
- `data/progressions.parquet`
- `data/chord_transition_graph.pkl`
- `data/function_transition_graph.pkl`
- `data/stats.pkl`

### 2. Generate Chord Progressions

```python
from app.generators.midi.prototype.generator import ChordProgressionGenerator

# Load the generator
gen = ChordProgressionGenerator()

# Random progression
prog = gen.generate_progression_random('C', 'Major', length=4, category='triad')

# Graph-guided (follows learned progressions)
prog = gen.generate_progression_graph_guided('C', 'Major', length=8, start_function='I')

# Brute-force + scoring (recommended!)
results = gen.generate_progression_brute_force(
    key_root='C',
    mode='Major',
    length=4,
    num_candidates=1000,  # Generate 1000 random progressions
    top_k=10,             # Return top 10 by score
    use_graph=True        # Use graph-guided generation
)

# Results are scored and ranked
for score, progression, score_breakdown in results:
    print(f"Score: {score:.3f}")
    print(f"Breakdown: {score_breakdown}")
    gen.print_progression(progression)
```

### 3. Custom Scoring Weights

Adjust scoring to prefer different musical qualities:

```python
# Prefer voice leading and graph probability
custom_weights = {
    'melody': 0.1,           # Melodic motion in top voice
    'voice_leading': 0.4,    # Smooth voice transitions
    'variety': 0.1,          # Chord diversity
    'graph_probability': 0.4 # Match learned progressions
}

results = gen.generate_progression_brute_force(
    'G', 'Major',
    length=6,
    num_candidates=500,
    top_k=5,
    weights=custom_weights
)
```

### 4. Query the Database Directly

Use Polars for fast filtering:

```python
import polars as pl

# Load data
chords_df = pl.read_parquet('data/chords.parquet')

# Get all IV chords in C Major
iv_chords = chords_df.filter(
    (pl.col('key_root') == 'C') &
    (pl.col('mode_in_key') == 'Major') &
    (pl.col('function') == 'IV')
)

# Get all extended chords
extended = chords_df.filter(pl.col('category') == 'extended')

# Sample 10 random chords containing note E4 (MIDI 64)
e_chords = chords_df.filter(
    pl.col('midi_notes').list.contains(64)
).sample(10)

# Get chords with specific interval structure (major triad: 0, 4, 7)
major_triads = chords_df.filter(
    pl.col('intervals') == [0, 4, 7]
)
```

## Scoring Functions

The prototype includes four scoring metrics:

### 1. **Melody Score** (`score_progression_melody`)
- Evaluates melodic motion in the highest voice
- Rewards stepwise motion (1-2 semitones)
- Penalizes large leaps (>5 semitones)

### 2. **Voice Leading Score** (`score_progression_voice_leading`)
- Measures total voice movement between chords
- Rewards minimal motion (smooth voice leading)
- Score: 1.0 = excellent (avg 3 semitones), 0.0 = poor (>12 semitones)

### 3. **Variety Score** (`score_progression_variety`)
- Ratio of unique chords to total chords
- Penalizes excessive repetition

### 4. **Graph Probability Score** (`score_progression_graph_probability`)
- Based on transition frequencies from the corpus
- Higher score for common progressions (I-IV-V, ii-V-I, etc.)

## Architecture Benefits

### Why Parquet + Graph?

**Polars/Parquet:**
- Blazing fast vectorized operations
- Low memory footprint (columnar compression)
- Easy filtering by multiple criteria
- Perfect for brute-force sampling

**NetworkX Graphs:**
- Encodes music theory knowledge
- Probability-weighted transitions
- Natural for progression generation
- Great for finding chord substitutions

**Hybrid Approach:**
1. Use Polars to quickly sample candidate progressions
2. Use graphs to guide generation toward musical patterns
3. Score candidates using both statistical and musical metrics
4. Return top-ranked progressions

## Next Steps for Full Implementation

### Bass Line Generation
```python
# Sample bass notes from chord root/bass notes
def generate_bass_line(progression):
    bass_line = []
    for chord in progression:
        # Could use root, or walk to next chord's root
        bass_line.append(chord['bass_note'])
    return bass_line
```

### Melody Generation
```python
# Sample notes from chord tones + passing tones
def generate_melody(progression):
    melody = []
    for chord in progression:
        # Sample from chord tones
        note = random.choice(chord['midi_notes'])
        melody.append(note)
    return melody
```

### Voice Leading Constraints
```python
# Add to graph edges: voice leading distance
# Filter candidates by voice leading quality before scoring
```

### Export to MIDI
```python
import mido

def export_to_midi(progression, tempo=120):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    for chord in progression:
        for note in chord['midi_notes']:
            track.append(mido.Message('note_on', note=note, velocity=64, time=0))
        track.append(mido.Message('note_off', note=note, velocity=0, time=480))

    return mid
```

## Performance

- **Database build:** ~10 seconds for 2,746 MIDI files
- **Generation:** ~0.5s for 1000 candidates (brute force)
- **Query:** <1ms for most Polars filters
- **Storage:** ~500KB total (Parquet + graphs)

## Example Output

```
--- Rank 1: Score = 0.738 ---
Score breakdown: {
    'melody': 1.0,
    'voice_leading': 1.17,
    'variety': 0.75,
    'graph_probability': 0.09
}
  1. I     | Cmaj7sus2       | Notes: ['C3', 'G3', 'D4', 'B4']
  2. I     | Cmaj7sus2       | Notes: ['C3', 'G3', 'D4', 'B4']
  3. I     | Cmaj6Add9       | Notes: ['C3', 'G3', 'D4', 'E4', 'A4']
  4. VII   | Gmaj6Add9;B     | Notes: ['B2', 'G3', 'D4', 'E4', 'A4']
```

## Files

```
app/generators/midi/prototype/
├── __init__.py
├── midi_parser.py          # Parse MIDI files to extract chord data
├── build_database.py       # Build Parquet + graph database
├── generator.py            # Chord progression generator
├── README.md              # This file
└── data/
    ├── chords.parquet
    ├── progressions.parquet
    ├── chord_transition_graph.pkl
    ├── function_transition_graph.pkl
    └── stats.pkl
```

## Production Pipeline

This prototype is used by the **Chord Generation Pipeline** (`app/generators/midi/chord_pipeline.py`), which adds:
- Reads song proposals from shrinkwrapped threads (key, BPM, concept, color)
- ChromaticScorer integration (scores chord progressions for chromatic consistency)
- Composite scoring (music theory + chromatic fitness)
- MIDI file output + YAML review interface for human labeling

```bash
python -m app.generators.midi.chord_pipeline \
    --thread shrinkwrapped/white-the-breathing-machine-learns-to-sing \
    --song "song_proposal_Black (0x221f20)_sequential_dissolution_v2.yml"
```

See `app/generators/midi/chord_pipeline.py` for full documentation.

## Dependencies

- `polars` - Fast DataFrame library
- `networkx` - Graph algorithms
- `mido` - MIDI file I/O
- `numpy` - Numerical operations (if needed)

Install via uv:
```bash
uv pip install polars networkx mido numpy
```
