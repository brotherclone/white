# Change: Add Drum Pattern Generation Phase

## Why

The chord pipeline produces approved chord progressions labeled by song section (verse, chorus, bridge, intro, outro). But chords alone are a harmonic skeleton — drums establish the rhythmic pulse and energy that define how a song *feels*. A verse with a sparse kick-hat pattern feels intimate; the same chords with a driving four-on-the-floor feel anthemic.

Drums are the next layer in the production pipeline because they set the rhythmic foundation that bass and melody lock to. This phase follows the same generate-review-promote pattern as chords: template-based generation, ChromaticScorer ranking, human approval via YAML.

## What Changes

- New **drum pattern generator** that reads approved chords from a song's production directory, maps song sections to genre-appropriate drum templates, scores candidates with ChromaticScorer, and presents top candidates for human labeling
- Drum templates are **multi-voice** (kick, snare, hi-hat, toms, cymbals) with **velocity dynamics** (accents, normal hits, ghost notes) — essential for drums sounding musical
- Templates are organized by **genre family** derived from the song proposal's genre tags (e.g., "glitch ambient" maps to sparse/textural patterns, "post-punk" maps to driving patterns)
- Reuses the existing review YAML schema and promote workflow
- One song at a time, same as chords

## Impact

- Affected specs: `production-review` (reuses existing review/promote pattern — no modifications needed)
- New specs: `drum-generation`
- Affected code:
  - New `app/generators/midi/drum_pipeline.py` — main orchestrator
  - New `app/generators/midi/drum_patterns.py` — template library
  - `training/chromatic_scorer.py` — used as-is for scoring
  - `app/generators/midi/promote_chords.py` — reused as-is for promotion (it's generic)
- No changes to chord pipeline or strum pipeline

## Architecture Overview

```
Approved Chords (from chord pipeline)
    │
    └── verse.mid, chorus.mid, bridge.mid (with labels from review.yml)
         │  Song proposal: BPM, time sig, genres, moods
         │
         ▼
    Drum Pattern Generation
         │  1. Read approved chords to determine song sections
         │  2. Map song proposal genres → drum genre families
         │  3. For each section, select applicable drum templates
         │  4. Generate MIDI on channel 10 (GM percussion)
         │  5. Score with ChromaticScorer (composite ranking)
         │  6. Write top candidates + review.yml
         │
         ▼
    Review File (YAML)
         │  Human listens to drum MIDI renders
         │  Labels: verse-drums, chorus-drums, bridge-fill, etc.
         │  Status: approved / rejected
         │
         ▼
    Approved Drums
         │  Ready for bass generation phase (future)
         │
         ▼
    [Future: Bass → Melody+Lyrics → Assembly → Mix]
```

## Non-Goals (this iteration)

- Swing/groove quantization (humanization) — defer to DAW
- Fill generation between sections (transition fills) — future enhancement
- Programmatic variation within a pattern (subtle randomization per bar) — defer
- Multi-bar patterns (patterns longer than 1 bar) — keep it simple for v1
- Audio rendering of drum MIDI — user renders in DAW
- Strum integration — drums and strums are independent layers built on approved chords
