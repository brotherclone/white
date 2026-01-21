# Generators Project Context

## Purpose
Generate White Album musical content using a hybrid approach: music theory + brute-force search + ML guidance. The pipeline transforms ontological concepts into performable music through progressive layering: chords → melody/lyrics → bass → drums → samples → human performance.

## Tech Stack
- Python 3.13
- Polars (DataFrames)
- NetworkX (transition graphs)
- mido (MIDI manipulation)
- LangGraph agents (creative direction)
- PyTorch models (eventual ML guidance)

## Project Philosophy

### Not Just Another AI Music Generator
This system is fundamentally different from typical AI music generation:

**What We're NOT Doing:**
- Training end-to-end neural nets to "create music"
- Generating synthetic audio directly
- Replacing human musicianship
- Making generic "AI-generated" tracks

**What We ARE Doing:**
- Using ML to **understand** ontological patterns and rebracketing
- Using music theory + search to **generate** musical structures
- Creating MIDI scaffolding that **guides** human musicians
- Building toward **real recordings** of **real instruments**
- Making "suck on that, Spotify" quality music

### The Hybrid Approach

```
Phase 1: Understanding (ML Training)
  ↓
  ├─ Multi-modal classifiers learn rebracketing patterns
  ├─ Style transfer models understand chromatic modes
  └─ Analysis tools reveal ontological structure

Phase 2: Generation (Music Theory + Search)
  ↓
  ├─ Graph-guided chord progressions (completed ✓)
  ├─ Melody generators with rebracketing markers
  ├─ Bass line generators following harmonic structure
  ├─ Drum pattern generators with ontological rhythm
  └─ Sample/loop selection and arrangement

Phase 3: Human Performance
  ↓
  ├─ MIDI scaffolds guide musicians
  ├─ Real instruments, real people
  └─ Final mixing and mastering

Phase 4: Feedback Loop
  ↓
  └─ Recordings become training data for next iteration
```

### Key Principles

1. **Theory-Informed, Not Theory-Bound**
   - Use music theory as scaffolding, not prison
   - Allow ontological needs to break traditional rules
   - Rebracketing may demand unconventional progressions

2. **Brute-Force Search with Compositional Scoring**
   - Generate many candidates (hundreds/thousands)
   - Score on multiple dimensions (melody, voice leading, variety, theory)
   - Select top-K for human review
   - Let composers choose from options, not just one output

3. **Incremental Layering**
   - Start simple: chord progressions
   - Build up: melody, then bass, then drums
   - Each layer informs the next
   - All layers remain editable MIDI until performance

4. **ML as Guide, Not Generator**
   - Trained models suggest chromatic modes for sections
   - Style transfer informs melodic contours
   - Rebracketing classifiers validate ontological correctness
   - But generation happens through search + scoring

5. **Human-in-the-Loop**
   - Agents (White, Violet, etc.) critique outputs
   - Musicians interpret and perform MIDI
   - Composers select from top candidates
   - Final creative decisions remain human

## Code Architecture

### Current Structure
```
app/generators/
├── midi/
│   └── prototype/
│       ├── build_database.py     # Parse MIDI corpus → Polars + NetworkX
│       ├── midi_parser.py        # MIDI file parsing utilities
│       ├── generator.py          # Chord progression generation (✓ Complete)
│       └── main.py               # Example usage
└── openspec/                     # This directory
    ├── project.md                # This file
    ├── specs/                    # Current capabilities
    └── changes/                  # Planned additions
```

### Future Structure
```
app/generators/
├── midi/
│   ├── chords/                   # Chord progression (✓ Complete)
│   ├── melody/                   # Melody generation (planned)
│   ├── bass/                     # Bass line generation (planned)
│   ├── drums/                    # Drum pattern generation (planned)
│   └── arrangement/              # Full song arrangement (planned)
├── samples/                      # Sample selection and manipulation
├── agents/                       # LangGraph creative agents
└── integration/                  # ML model integration
```

## Domain Context

See:
- `/Volumes/LucidNonsense/White/white_album_project_diary.md` - Overall project
- `/Volumes/LucidNonsense/White/training/docs/REBRACKETING_TAXONOMY.md` - 8 rebracketing types
- `/Volumes/LucidNonsense/White/training/openspec/` - ML training pipeline

## Important Constraints

- **MIDI as interchange format** - All generation outputs MIDI for human editing
- **No direct audio synthesis** - Leave that to real instruments and musicians
- **Ontological correctness** - Music must serve conceptual rebracketing, not just "sound good"
- **Performability** - Human musicians must be able to play the parts
- **Editability** - All outputs remain human-editable before performance

## Data Sources

### Training Data (for ML)
- Staged raw material: `/Volumes/LucidNonsense/White/staged_raw_material/`
- Rainbow Table albums (8 albums, ~14k segments)
- Multimodal: text (concepts/lyrics), audio (WAV), MIDI

### Generation Data (for Music Theory)
- Chord library: `/Volumes/LucidNonsense/White/chords/`
- MIDI progressions parsed into transition graphs
- Function-based theory (roman numerals, voice leading)

## External Dependencies

- Polars for DataFrame operations
- NetworkX for graph-based generation
- mido for MIDI file I/O
- LangGraph for agent orchestration (later)
- PyTorch trained models for guidance (later)

## Testing Strategy

Generators have different testing needs than ML models:

**Unit Tests:**
- Database building (parsing, graph construction)
- Scoring functions (melody, voice leading, etc.)
- MIDI utilities (note manipulation, file I/O)

**Integration Tests:**
- Full generation pipeline (chords → MIDI file)
- Multi-layer generation (chords + melody + bass)

**Qualitative Tests:**
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

## Integration with Training

Eventually, generators and ML training inform each other:

**Training → Generation:**
- Rebracketing classifier suggests which type to emphasize
- Chromatic style model suggests melodic contours
- Temporal sequence model suggests progression pacing

**Generation → Training:**
- Generated music becomes new training data
- Synthetic examples for rare rebracketing types
- Controlled experiments (vary one parameter, observe effect)

But this integration is **Phase 4** - first we build solid generators.

## Current Status

**Completed:**
- ✅ Chord progression database (Polars + NetworkX)
- ✅ Graph-guided generation
- ✅ Multi-criteria scoring (melody, voice leading, variety, probability)
- ✅ Brute-force search with top-K selection

**In Progress:**
- 🔄 OpenSpec documentation (this!)

**Planned:**
- 📋 Melody generator
- 📋 Lyrics integration with melody
- 📋 Bass line generator
- 📋 Drum pattern generator
- 📋 Full arrangement pipeline
- 📋 Agent-based creative direction
- 📋 ML model integration

## Getting Started

See the example files:
- `app/generators/midi/prototype/main.py` - Basic usage examples
- `app/generators/midi/prototype/generator.py` - Full API documentation
- (Coming soon) OpenSpec changes for next phases
