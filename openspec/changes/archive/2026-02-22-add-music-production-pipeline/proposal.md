# Change: Add Music Production Pipeline (Chord Phase)

## Why

The White Concept Agent generates rich chromatic concepts and the ChromaticScorer (Phase 10) can evaluate MIDI against those concepts. But there is no system to *compose music* from a shrinkwrapped concept. The gap between "here is a concept for a RED song" and "here are chord progressions to record in Logic Pro" requires a generation + scoring + human review pipeline.

This first iteration focuses on **chord generation** — the foundational layer that all subsequent music (drums, bass, melody) builds upon. A working chord pipeline proves the pattern (generate → score → human gate → approve) before scaling to additional instruments.

## What Changes

- New **chord generation pipeline** that reads a song proposal from a shrinkwrapped thread, generates chord progression candidates via the existing Markov chain prototype, scores them with ChromaticScorer, and presents top candidates for human labeling
- New **production review interface** — YAML-based review files where the human labels candidates (verse-candidate, chorus-candidate, bridge-candidate, etc.) and approves/rejects them
- Connects the existing chord prototype (`app/generators/midi/prototype/`) with the ChromaticScorer (`training/chromatic_scorer.py`) into an end-to-end workflow
- One song proposal at a time — the user picks a color/song from a shrinkwrapped thread

## Impact

- Affected specs: none (all new capabilities)
- New specs: `chord-generation`, `production-review`
- Affected code:
  - `app/generators/midi/prototype/generator.py` — extend with ChromaticScorer integration
  - `training/chromatic_scorer.py` — used as-is (already built)
  - New orchestrator script to tie generation → scoring → review
  - New YAML schema for review files
- Supersedes: `training/openspec/changes/add-evolutionary-music-generator/` (rough proposal, never implemented — scope has changed significantly)

## Architecture Overview

```
Shrinkwrapped Thread
    │
    └── Song Proposal (one color, e.g. Black from "Breathing Machine")
         │  key: F# minor, bpm: 84, time_sig: 7/8, concept: "..."
         │
         ▼
    Chord Generation
         │  1. ChordProgressionGenerator reads key/mode from proposal
         │  2. Generates N candidates (graph-guided Markov + brute-force)
         │  3. Music theory scoring (voice leading, variety, graph probability)
         │  4. ChromaticScorer scoring (how well does this match the chromatic concept?)
         │  5. Composite ranking → top 5-10 candidates
         │
         ▼
    Review File (YAML)
         │  Human listens to MIDI renders
         │  Labels: verse-candidate, chorus-candidate, bridge-candidate
         │  Status: approved / rejected / needs-revision
         │
         ▼
    Approved Chords
         │  Ready for drum generation phase (future)
         │
         ▼
    [Future: Drums → Bass → Melody+Lyrics → Assembly → Mix]
```

## Non-Goals (this iteration)

- Drum pattern generation
- Bass line generation
- Melody or lyric generation
- ACE Studios integration
- Logic Pro recipe export
- Web UI for review (YAML files are sufficient)
- Automated MIDI-to-audio rendering (user renders in DAW or we use FluidSynth later)
