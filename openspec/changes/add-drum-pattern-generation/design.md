## Context

The production pipeline has approved chord progressions with section labels (verse, chorus, bridge, intro, outro). Song proposals contain BPM, time signature, genre tags, and mood descriptors. The ChromaticScorer can evaluate any MIDI against a chromatic concept.

Drums differ from chords and strums in three key ways:
1. **Multi-voice**: kick, snare, hi-hat, toms, and cymbals are independent rhythmic lines played simultaneously
2. **Velocity-sensitive**: ghost notes (soft), normal hits, and accents define the feel more than note placement alone
3. **Genre-driven**: the same time signature produces radically different patterns across genres (ambient vs. post-punk vs. glitch)

## Goals / Non-Goals

**Goals:**
- Template-based drum pattern generation with multi-voice support
- Velocity dynamics (accent, normal, ghost) baked into templates
- Genre family mapping from song proposal genre tags
- Section-aware candidate generation (verse gets different patterns than chorus)
- ChromaticScorer composite scoring, same as chord pipeline
- Same review/promote YAML workflow

**Non-Goals:**
- Procedural/algorithmic rhythm generation (Euclidean, etc.)
- Humanization or swing
- Transition fills between sections
- Multi-bar patterns
- Audio preview generation

## Decisions

### Decision: GM Percussion MIDI mapping

All drum MIDI uses General MIDI channel 10 percussion mapping:

| Voice     | MIDI Note | Description          |
|-----------|-----------|----------------------|
| Kick      | 36        | Bass Drum 1          |
| Snare     | 38        | Acoustic Snare       |
| Rimshot   | 37        | Side Stick           |
| HH Closed | 42        | Closed Hi-Hat        |
| HH Open   | 46        | Open Hi-Hat          |
| HH Pedal  | 44        | Pedal Hi-Hat         |
| Crash     | 49        | Crash Cymbal 1       |
| Ride      | 51        | Ride Cymbal 1        |
| Tom High  | 50        | High Tom             |
| Tom Mid   | 47        | Low-Mid Tom          |
| Tom Low   | 45        | Low Tom              |
| Clap      | 39        | Hand Clap            |

This is the universally supported standard. Any DAW or synth plugin will interpret these correctly.

### Decision: Velocity levels

Three velocity tiers encoded directly in templates:

| Level  | Velocity | Usage                           |
|--------|----------|---------------------------------|
| Accent | 120      | Downbeats, emphasis hits        |
| Normal | 90       | Standard hits                   |
| Ghost  | 45       | Ghost notes, texture, feel      |

This provides enough dynamic range for patterns to feel musical without overcomplicating the template format. The user can further adjust in their DAW.

### Decision: Template data structure

Each drum template is a Python dict:

```python
{
    "name": "rock_basic",
    "genre_family": "rock",
    "energy": "medium",       # low, medium, high
    "description": "Basic rock beat — kick on 1&3, snare on 2&4, closed hats on eighths",
    "time_sig": (4, 4),
    "voices": {
        "kick":      [(0, "accent"), (2, "normal")],
        "snare":     [(1, "accent"), (3, "accent")],
        "hh_closed": [(0, "normal"), (0.5, "ghost"), (1, "normal"), (1.5, "ghost"),
                      (2, "normal"), (2.5, "ghost"), (3, "normal"), (3.5, "ghost")],
    }
}
```

Each voice entry is a list of `(beat_position, velocity_level)` tuples. Beat positions are floats relative to the bar (0 = beat 1, 0.5 = eighth note after beat 1, etc.).

This format is:
- Human-readable and auditable
- Easy to add new patterns
- Directly convertible to MIDI events
- Extensible (can add swing offset, probability, etc. later without breaking existing templates)

### Decision: Genre family mapping

Song proposals have genre tags like `["glitch ambient", "post-classical", "microsound"]`. These map to genre families that determine which templates are applicable:

| Genre Family | Genre Tags (contains)                              | Character                    |
|--------------|-----------------------------------------------------|------------------------------|
| ambient      | ambient, drone, atmospheric, soundscape              | Sparse, textural, space      |
| electronic   | electronic, synth, IDM, glitch, microsound           | Precise, programmatic        |
| krautrock    | krautrock, motorik, kosmische, neu!, can              | Motorik pulse, metronomic    |
| rock         | rock, post-punk, punk, garage, alternative           | Driving, backbeat            |
| classical    | classical, post-classical, orchestral, chamber        | Minimal percussion, tympani  |
| experimental | experimental, noise, industrial, avant-garde          | Unconventional, textural     |
| folk         | folk, acoustic, singer-songwriter                     | Brushes, simple patterns     |
| jazz         | jazz, swing, bebop, fusion                            | Ride-focused, syncopated     |

A song proposal's genres are scanned for keywords. Multiple families can match (e.g., "glitch ambient" matches both `ambient` and `electronic`). When multiple families match, templates from all matching families are included as candidates.

If no genre tags match any family, the generator falls back to `electronic` (the most neutral/versatile family for this project).

### Decision: Section-to-energy mapping

Different song sections typically have different energy levels. The generator uses a default mapping:

| Section  | Default Energy | Rationale                        |
|----------|---------------|----------------------------------|
| intro    | low           | Build into the song              |
| verse    | medium        | Foundation, not peak energy      |
| chorus   | high          | Peak energy, driving             |
| bridge   | low           | Contrast, breakdown              |
| outro    | medium        | Wind down but stay present       |

The user can override energy levels via CLI flags. Templates are tagged with an energy level, and the generator matches section energy to template energy.

### Decision: Candidate generation strategy

For each song section (from approved chords):
1. Determine the section's energy level
2. Find all templates matching: time signature + genre family + energy level
3. Also include templates one energy level adjacent (e.g., if section is "medium", include some "low" and "high" templates for variety)
4. Generate MIDI for each matching template × the section's bar count (from approved chord length)
5. Score all candidates with ChromaticScorer
6. Present top-k per section in review.yml

Expected candidate count: ~3 genre families × ~3 energy-matched templates × ~3 sections = ~27 candidates. Manageable for human review.

### Decision: ChromaticScorer integration

Drum patterns are scored the same way as chord candidates:
- Convert drum MIDI to bytes
- Compute `prepare_concept()` once from song proposal concept text
- Run `score()` on each candidate
- Composite = 30% energy-appropriateness + 70% chromatic match

The theory component is replaced by an **energy appropriateness** score:
- 1.0 if template energy exactly matches section target
- 0.5 if one level away
- 0.0 if two levels away

This is simpler than chord theory scoring because music theory metrics (voice leading, melody) don't apply to percussion.

### Decision: Output directory structure

```
production/black_sequential_dissolution/
├── chords/
│   ├── approved/
│   │   ├── verse.mid
│   │   ├── chorus.mid
│   │   └── bridge.mid
│   └── review.yml
├── drums/                              # NEW
│   ├── candidates/
│   │   ├── verse_ambient_sparse_01.mid
│   │   ├── verse_electronic_pulse_01.mid
│   │   ├── chorus_electronic_driving_01.mid
│   │   ├── chorus_rock_basic_01.mid
│   │   ├── bridge_ambient_minimal_01.mid
│   │   └── ...
│   ├── approved/
│   │   ├── verse_drums.mid
│   │   ├── chorus_drums.mid
│   │   └── bridge_drums.mid
│   └── review.yml
├── strums/                             # independent
└── assembly/                           # future
```

### Decision: Reuse promote_chords.py

The existing `promote_chords.py` is generic — it reads any `review.yml` with candidates and promotes approved entries. No changes needed for drums.

## Risks / Trade-offs

- **Template count may be thin for some genre families** — v1 ships with a modest set (~5-8 per family). More templates can be added incrementally based on the user's needs.
- **ChromaticScorer trained on melodic/harmonic content** — Drum-only MIDI may score differently than the training data. The scorer's confidence output will flag uncertainty. Empirical testing needed.
- **One-bar patterns may feel repetitive** — For v1 this is acceptable; the user arranges patterns across bars in the DAW. Multi-bar patterns are a natural future enhancement.
- **No fill generation** — Transition fills (e.g., snare roll into chorus) are deferred. The user can compose fills manually in the DAW using the approved pattern as a starting point.

## Open Questions

None — all key decisions resolved during proposal review.
