# Generators OpenSpec Documentation

This directory contains complete OpenSpec documentation for the White Album generators system - the "brute force" approach to music generation that honors both musical craftsmanship and machine intelligence.

## Philosophy

**NOT another AI music generator.** This system uses:
- Music theory as scaffolding (not prison)
- Brute-force search with compositional scoring
- ML models to **guide** (not replace) generation
- MIDI scaffolds for real human musicians
- "Suck on that, Spotify" quality standards

## Project Structure

```
app/generators/openspec/
├── README.md (this file)
├── project.md (overall project context and conventions)
├── specs/ (current capabilities)
│   └── chord-progression-generator/
│       └── spec.md
└── changes/ (planned additions)
    ├── add-melody-generator/
    │   ├── proposal.md
    │   ├── tasks.md
    │   └── specs/melody-generator/spec.md
    ├── add-bass-generator/
    │   ├── proposal.md
    │   ├── tasks.md
    │   └── specs/bass-generator/spec.md
    ├── add-drum-generator/
    │   ├── proposal.md
    │   ├── tasks.md
    │   └── specs/drum-generator/spec.md
    ├── add-arrangement-pipeline/
    │   ├── proposal.md
    │   ├── tasks.md
    │   └── specs/arrangement-pipeline/spec.md
    └── add-ml-model-integration/
        ├── proposal.md
        ├── tasks.md
        └── specs/ml-integration/spec.md
```

## Current Status

### ✅ Complete (Working Code)
- **Chord Progression Generator** (`/app/generators/midi/prototype/generator.py`)
  - Polars DataFrames + NetworkX transition graphs
  - Graph-guided generation
  - Multi-criteria scoring (melody, voice leading, variety, graph probability)
  - Brute-force search with top-K selection
  - MIDI export

### 📋 Planned (OpenSpec Documented)

#### 1. Melody Generator (`add-melody-generator/`)
**Why**: Melody carries conceptual narrative and rebracketing markers

**Key Features**:
- Syllable-based lyrics parsing with stress patterns
- Prosody-based rhythm generation
- Melodic contour templates (arch, descending, wave)
- Harmonic alignment with chord progressions
- Rebracketing marker melodic emphasis (pitch elevation, duration extension)
- Vocal range constraints and singability
- MIDI export with embedded lyrics

**Requirements**: 14 main requirements, 292 scenarios
**Tasks**: 12 sections, 63 implementation tasks

#### 2. Bass Line Generator (`add-bass-generator/`)
**Why**: Bass provides rhythmic/harmonic foundation, locks with drums for groove

**Key Features**:
- Chord tone extraction and harmonic alignment
- Walking bass patterns with chromatic approaches
- Groove pattern templates (funk, rock, jazz, R&B)
- Melody integration and clash avoidance
- Rebracketing marker emphasis (low register, pedal tones, rhythmic shifts)
- Bass-drum synchronization (kick alignment)
- Playability constraints (range, fingering, position shifts)

**Requirements**: 28 main requirements, extensive scenarios
**Tasks**: 20 sections, 100 implementation tasks

#### 3. Drum Pattern Generator (`add-drum-generator/`)
**Why**: Drums define groove, tempo, energy, and rhythmic emphasis

**Key Features**:
- Multi-voice drum patterns (kick, snare, hi-hat, toms, cymbals)
- Groove templates (rock, funk, jazz, electronic, hip-hop, breakbeat)
- Kick-bass synchronization for rhythmic pocket
- Fill generation at transitions
- Build-ups and breakdowns
- Rebracketing marker emphasis (crashes, fills, breaks, dynamic surges)
- Humanization (timing/velocity variation)
- General MIDI drum map export

**Requirements**: 30+ main requirements
**Tasks**: 29 sections, 145 implementation tasks

#### 4. Arrangement Pipeline (`add-arrangement-pipeline/`)
**Why**: Orchestrates all generators into complete song structure

**Key Features**:
- Song structure templates (verse-chorus, AABA, through-composed)
- Sectional variation (verse lighter, chorus heavier)
- Transition generation (fills, builds, breakdowns)
- Energy mapping across song structure
- Rebracketing marker coordination across all layers
- Sample/loop integration
- Multi-track MIDI export
- LangGraph agent critique (White, Violet, Black agents)

**Requirements**: 38 main requirements
**Tasks**: 35 sections, 175 implementation tasks

#### 5. ML Model Integration (`add-ml-model-integration/`)
**Why**: Close the feedback loop between training and generation

**Key Features**:
- Rebracketing classifier integration (validate ontological correctness)
- Chromatic style model guidance (musical character per mode)
- Temporal sequence model (energy curve prediction)
- Multi-modal embedding integration (concept → parameters)
- Synthetic data generation (generators → training data)
- Human-in-the-loop feedback (musicians critique → model refinement)
- ML as guide, not generator (theory remains primary)

**Requirements**: 40+ main requirements
**Tasks**: 38 sections, 190 implementation tasks

## Generation Pipeline Flow

```
User Input: Concept + Lyrics + Chromatic Mode
    ↓
┌───────────────────────────────────┐
│  1. ML Model Guidance (Optional)  │
│  - Predict parameters from concept│
│  - Suggest chromatic character    │
│  - Predict energy curve           │
└───────────────┬───────────────────┘
                ↓
┌───────────────────────────────────┐
│  2. Chord Progression Generation  │ ✅ COMPLETE
│  - Graph-guided or random         │
│  - Multi-criteria scoring         │
│  - Brute-force search (N=1000)    │
│  - Return top-K (K=10)            │
└───────────────┬───────────────────┘
                ↓
┌───────────────────────────────────┐
│  3. Melody Generation             │ 📋 Planned
│  - Parse lyrics to syllables      │
│  - Prosody → rhythm               │
│  - Generate melodic contours      │
│  - Align to harmony               │
│  - Emphasize rebracketing markers │
└───────────────┬───────────────────┘
                ↓
┌───────────────────────────────────┐
│  4. Bass Line Generation          │ 📋 Planned
│  - Extract chord tones            │
│  - Generate walking/groove bass   │
│  - Avoid melody clashes           │
│  - Emphasize markers (low reg)    │
└───────────────┬───────────────────┘
                ↓
┌───────────────────────────────────┐
│  5. Drum Pattern Generation       │ 📋 Planned
│  - Select groove template         │
│  - Synchronize with bass          │
│  - Generate fills/builds          │
│  - Emphasize markers (crashes)    │
└───────────────┬───────────────────┘
                ↓
┌───────────────────────────────────┐
│  6. Arrangement Orchestration     │ 📋 Planned
│  - Apply song structure           │
│  - Coordinate all layers          │
│  - Add samples/loops              │
│  - Generate transitions           │
└───────────────┬───────────────────┘
                ↓
┌───────────────────────────────────┐
│  7. ML Validation (Optional)      │ 📋 Planned
│  - Rebracketing classifier check  │
│  - Style embedding scoring        │
│  - Regenerate if validation fails │
└───────────────┬───────────────────┘
                ↓
┌───────────────────────────────────┐
│  8. Agent Critique                │ 📋 Planned
│  - White: Ontological correctness │
│  - Violet: Musical quality        │
│  - Black: Creative alternatives   │
└───────────────┬───────────────────┘
                ↓
┌───────────────────────────────────┐
│  9. Multi-Track MIDI Export       │
│  - Track 1: Chords                │
│  - Track 2: Melody + Lyrics       │
│  - Track 3: Bass                  │
│  - Track 4: Drums                 │
│  - Track 5+: Samples              │
└───────────────┬───────────────────┘
                ↓
┌───────────────────────────────────┐
│  10. Human Performance            │
│  - Musicians load MIDI            │
│  - Perform with real instruments  │
│  - Record audio                   │
│  - Mix and master                 │
└───────────────┬───────────────────┘
                ↓
┌───────────────────────────────────┐
│  11. Feedback Loop                │ 📋 Planned
│  - Recordings → training data     │
│  - Retrain ML models              │
│  - Improved guidance next cycle   │
└───────────────────────────────────┘
```

## Key Design Principles

### 1. Theory-Informed, Not Theory-Bound
- Use music theory as scaffolding, not prison
- Allow ontological needs to break traditional rules
- Rebracketing may demand unconventional progressions

### 2. Brute-Force Search with Compositional Scoring
- Generate many candidates (hundreds/thousands)
- Score on multiple dimensions (melody, voice leading, variety, theory, ML)
- Select top-K for human review
- Let composers choose from options, not just one output

### 3. Incremental Layering
- Start simple: chord progressions
- Build up: melody, then bass, then drums
- Each layer informs the next
- All layers remain editable MIDI until performance

### 4. ML as Guide, Not Generator
- Trained models suggest chromatic modes for sections
- Style transfer informs melodic contours
- Rebracketing classifiers validate ontological correctness
- But generation happens through search + scoring

### 5. Human-in-the-Loop
- Agents (White, Violet, Black) critique outputs
- Musicians interpret and perform MIDI
- Composers select from top candidates
- Final creative decisions remain human

## Integration with Training Pipeline

The generators integrate bidirectionally with the training pipeline (`/training/`):

**Training → Generation**:
- Rebracketing classifier validates marker emphasis
- Chromatic style model suggests musical character
- Temporal sequence model guides progression pacing
- Multi-modal embeddings map concepts to parameters

**Generation → Training**:
- Generated MIDI → musician performance → recordings
- Recordings become new training data
- Synthetic examples for rare rebracketing types
- Controlled experiments (vary parameters, observe effect)

## Tech Stack

- **Python 3.13**
- **Polars** - DataFrames for chord database
- **NetworkX** - Transition graphs for chord progressions
- **mido** - MIDI file manipulation
- **PyTorch** - ML model inference (when integrated)
- **LangGraph** - Agent orchestration (when integrated)

## Data Sources

**Generation Data** (music theory):
- Chord library: `/Volumes/LucidNonsense/White/chords/`
- MIDI progressions parsed into transition graphs
- Function-based theory (roman numerals, voice leading)

**Training Data** (ML models):
- Rainbow Table: `/Volumes/LucidNonsense/White/staged_raw_material/`
- 8 albums, ~14k segments
- Multi-modal: text (concepts/lyrics), audio (WAV), MIDI

## Testing Strategy

### Unit Tests
- Database building (parsing, graph construction)
- Scoring functions (melody, voice leading, etc.)
- MIDI utilities (note manipulation, file I/O)

### Integration Tests
- Full generation pipeline (chords → MIDI file)
- Multi-layer generation (chords + melody + bass + drums)

### Qualitative Tests
- Agent evaluation ("Does this feel like temporal rebracketing?")
- Musician playability ("Can this actually be performed?")
- Composer selection ("Which of top-10 progressions is best?")

## Success Metrics

Unlike ML training (loss, accuracy, F1), generators are judged by:

1. **Musical Quality**
   - Voice leading smoothness
   - Harmonic interest
   - Rhythmic coherence

2. **Ontological Correctness**
   - Does it serve the rebracketing concept?
   - Does it match the intended chromatic mode?
   - Does it create the right ontological tension?

3. **Practical Usability**
   - Can musicians play it?
   - Can composers edit it?
   - Does it export cleanly to DAWs?

4. **Creative Value**
   - Does it inspire human musicians?
   - Does it suggest unexpected but valid choices?
   - Does it expand creative possibilities?

## Getting Started

### Current Working Example
```python
from app.generators.midi.prototype.generator import ChordProgressionGenerator

# Initialize (loads Polars DataFrames + NetworkX graphs)
generator = ChordProgressionGenerator(data_dir="chords/")

# Generate chord progressions
progressions = generator.generate_progression_brute_force(
    key_root="C",
    mode="Major",
    length=8,
    num_candidates=1000,
    top_k=10,
    use_graph=True
)

# Review top candidates with score breakdowns
for i, (prog, total_score, scores) in enumerate(progressions):
    print(f"\n=== Progression {i+1} (Score: {total_score:.3f}) ===")
    print(f"Melody: {scores['melody']:.3f}")
    print(f"Voice Leading: {scores['voice_leading']:.3f}")
    print(f"Variety: {scores['variety']:.3f}")
    print(f"Graph Probability: {scores['graph_probability']:.3f}")

    for chord in prog:
        print(f"  {chord['function']} - {chord['chord_name']}")
```

### Future Usage (when melody/bass/drums/arrangement complete)
```python
from app.generators.midi.arrangement.pipeline import ArrangementPipeline

# Initialize full pipeline
pipeline = ArrangementPipeline(
    chord_generator=chord_gen,
    melody_generator=melody_gen,
    bass_generator=bass_gen,
    drum_generator=drum_gen
)

# Generate complete arrangement
arrangement = pipeline.generate(
    concept="The room is not a container",
    lyrics=lyrics,
    chromatic_mode="Violet",
    structure="verse-chorus",
    style="indie rock"
)

# Export multi-track MIDI
arrangement.export_midi("the_room_is_not_a_container.mid")

# Musicians load MIDI, perform, record → "Suck on that, Spotify"
```

## Documentation

- **`project.md`** - Overall project context, philosophy, architecture
- **`specs/`** - Current implemented capabilities
- **`changes/`** - Planned additions (proposals, tasks, specs)

Each change proposal includes:
- **proposal.md** - Why, what changes, impact, design considerations
- **tasks.md** - Implementation checklist (numbered tasks)
- **specs/*.md** - Detailed requirements in WHEN/THEN scenario format

## Related Documentation

- Training Pipeline: `/Volumes/LucidNonsense/White/training/openspec/`
- Project Diary: `/Volumes/LucidNonsense/White/white_album_project_diary.md`
- Rebracketing Taxonomy: `/Volumes/LucidNonsense/White/training/docs/REBRACKETING_TAXONOMY.md`

## Status Summary

| Component | Status | Requirements | Tasks | Notes |
|-----------|--------|--------------|-------|-------|
| Chord Progression | ✅ Complete | 17 | - | Working code in prototype/ |
| Melody Generation | 📋 Documented | 14 | 63 | Ready for implementation |
| Bass Generation | 📋 Documented | 28 | 100 | Ready for implementation |
| Drum Generation | 📋 Documented | 30+ | 145 | Ready for implementation |
| Arrangement Pipeline | 📋 Documented | 38 | 175 | Ready for implementation |
| ML Integration | 📋 Documented | 40+ | 190 | Requires training models complete |

**Total Documented**: 5 change proposals, 167 main requirements, 673 implementation tasks

## Next Steps

1. **Implement Melody Generator** - Start with lyrics parsing and prosody-based rhythm
2. **Implement Bass Generator** - Chord tone extraction and groove patterns
3. **Implement Drum Generator** - Multi-voice patterns and fills
4. **Implement Arrangement Pipeline** - Orchestrate all generators
5. **Integrate ML Models** - Connect with training pipeline when Phase 2+ models trained

## Vision

The end goal: A system that combines the best of human musicianship and machine intelligence to create music that serves ontological concepts. Not replacing musicians, not generating synthetic garbage, but creating **MIDI scaffolds** that inspire and guide **real human performances** that are recorded, mixed, and mastered to professional standards.

**Suck on that, Spotify.** 🎵
