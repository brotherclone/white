## Context

The chord pipeline currently writes one chord per bar as a whole note (`bar_ticks = ticks_per_beat * 4`). Approved chords are harmonic selections — the user has already decided these are the right chords for the verse, chorus, bridge, etc. Now those chords need rhythmic treatment.

## Goals / Non-Goals

**Goals:**
- Take approved chord MIDI and generate rhythmic variations
- Support common rhythm patterns: whole, half, quarter, eighth, syncopated, arpeggiated
- Respect time signature (7/8 patterns differ from 4/4)
- Same generate → review → approve workflow as chords
- CLI invocation pointing at an approved chords directory

**Non-Goals:**
- Re-scoring with ChromaticScorer (harmony is unchanged)
- Velocity curves or humanization
- Generating new chord voicings (input chords are fixed)

## Decisions

### Decision: Rhythm patterns as templates

Each strum pattern is a template that defines note onsets and durations relative to the bar. Templates are time-signature-aware.

**4/4 examples (bar = 4 beats):**

| Pattern | Onsets (beats) | Durations | Description |
|---------|---------------|-----------|-------------|
| whole | [0] | [4] | Current behavior — one hit per bar |
| half | [0, 2] | [2, 2] | Two hits per bar |
| quarter | [0, 1, 2, 3] | [1, 1, 1, 1] | Four hits per bar |
| eighth | [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5] | [0.5 each] | Eight hits per bar |
| push | [0, 1.5, 2, 3.5] | [1.5, 0.5, 1.5, 0.5] | Syncopated — emphasis on 1 and 3, pushes on 2-and and 4-and |
| arp-up | [0, 0.25, 0.5, 0.75, ...] | per-note | Arpeggiate chord tones low→high, one per subdivision |
| arp-down | same but high→low | per-note | Arpeggiate chord tones high→low |

**7/8 examples (bar = 3.5 beats = 7 eighth notes):**

| Pattern | Onsets (eighth notes) | Description |
|---------|----------------------|-------------|
| whole | [0] | One hit, full bar |
| grouped-322 | [0, 3, 5] | 3+2+2 grouping (common 7/8 feel) |
| grouped-223 | [0, 2, 4] | 2+2+3 grouping |
| eighth | [0, 1, 2, 3, 4, 5, 6] | All eighth notes |

Templates are defined as Python data structures, not generated. This keeps behavior predictable and auditable.

**Alternatives considered:**
- Probabilistic rhythm generation (random onsets with constraints) — rejected because unpredictable and hard to review; the user wants known patterns applied to their chords
- ML-based rhythm generation — overkill; rhythm patterns are a solved problem in music theory

### Decision: Arpeggio handling

For arpeggiated patterns, notes from the chord are played sequentially instead of simultaneously. The pattern distributes chord tones across the subdivision:

- If chord has 4 notes and pattern has 8 slots: repeat the arpeggio cycle
- If chord has 5 notes and pattern has 4 slots: truncate to first 4 notes per cycle
- Direction: up (low→high), down (high→low)

### Decision: Candidates per pattern

For each approved chord file, the generator produces one candidate per applicable rhythm pattern. With ~6 patterns per time signature, a song with 3 approved chords (verse, chorus, bridge) would produce ~18 candidates total — a manageable review set.

The user may also want to hear the same pattern applied to the full progression (all approved chords strummed the same way). The generator should support both:
- **Per-chord mode** (default): each approved chord × each pattern = separate MIDI files
- **Progression mode**: all approved chords in sequence × each pattern = one MIDI file per pattern showing the full song flow

### Decision: Output directory structure

```
production/black_sequential_dissolution/
├── chords/
│   ├── approved/
│   │   ├── intro.mid
│   │   ├── verse.mid
│   │   └── bridge.mid
│   └── review.yml
├── strums/                          # NEW
│   ├── candidates/
│   │   ├── intro_half.mid
│   │   ├── intro_quarter.mid
│   │   ├── intro_eighth.mid
│   │   ├── verse_half.mid
│   │   ├── verse_quarter.mid
│   │   ├── ...
│   │   ├── progression_half.mid     # full sequence
│   │   ├── progression_quarter.mid
│   │   └── ...
│   ├── approved/
│   │   ├── intro_quarter.mid
│   │   └── verse_driving.mid
│   └── review.yml
├── drums/                           # future
└── assembly/                        # future
```

### Decision: Reuse existing review/promote workflow

The strum review file uses the same YAML schema as chord review. The existing `promote_chords.py` can be reused as-is (it operates on any review.yml with candidates/approved directories). No changes needed.

## Risks / Trade-offs

- **Pattern proliferation** — Many patterns × many approved chords could generate a lot of candidates. Mitigated by keeping pattern count small (~6) and supporting progression mode to reduce review burden.
- **Arpeggiation quality** — Simple up/down arpeggiation may sound mechanical. Acceptable for MIDI sketches; the user will refine in Logic Pro.
- **Odd time signatures** — 7/8, 5/4 etc. need specific pattern templates. The prototype will ship with 4/4 and 7/8 patterns; others can be added as needed.
