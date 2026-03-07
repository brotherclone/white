# Proposal: redesign-melody-templates

## Problem

The current melody template library (31 templates, `melody_patterns.py`) produces
output that sounds theatrical rather than naturalistic — closer to musical theatre
than Nebraska-era Springsteen or The National. Two root causes:

1. **Templates are pitch-dense**: most patterns fill every or nearly every beat,
   leaving no room for lyrics. A real vocal phrase needs long notes, held vowels,
   and rests. The current templates were scored to sound like melodies in a DAW;
   they were not designed around syllable delivery.

2. **No `use_case` distinction**: some of the busier patterns are genuinely good as
   lead instrument lines (guitar hooks, synth solos). Right now the pipeline has no
   way to route them there — everything is treated as a potential vocal melody. This
   wastes good material and applies it badly.

Secondary issue: 31 templates across all time signatures and energy levels is too few.
Sameness is already audible after one song.

## Proposed Solution

### 1. Add `use_case` field to `MelodyPattern`

```python
use_case: str  # "vocal" | "lead"
```

- `"vocal"` — sparse, syllable-friendly, long durations, ≥30% rests by beat count
- `"lead"` — can be dense, ornamental, used as instrument track (guitar, synth, etc.)

The melody pipeline SHALL filter to `use_case="vocal"` when generating singer parts.
Lead-use patterns remain in the library for future instrument track generation.

### 2. Redesign vocal templates around syllable delivery

Vocal templates SHALL be designed with:
- Note durations of ≥ 1 beat on stressed syllables (half notes, dotted quarters)
- Explicit rests as structural elements (not just gaps between onsets)
- ≤ 6 notes per bar in 4/4 (max syllable density for natural speech rhythm)
- Long held notes on phrase endings (at least 1.5 beats)
- Inspiration from sparse vocal styles: Nebraska, The National (Boxer/Sleep Well Beast),
  Lankum, Phosphorescent

### 3. Expand template count

Target: ≥ 50 templates total
- ≥ 30 vocal templates (4/4)
- ≥ 8 vocal templates (other time sigs: 3/4, 6/8, 7/8)
- ≥ 12 lead templates (retained/reclassified from current library)

New vocal template archetypes needed:
- **Declarative** — one long phrase, syllable-per-beat, resolves on a held note
- **Call-and-rest** — phrase, breath, phrase, silence (Springsteen Nebraska style)
- **Haiku** — three short phrases with silence between (The National style)
- **Incantatory** — repeated short motif with minimal interval movement
- **Drone-and-step** — held root with one descent/ascent (Lankum style)
- **Conversational** — irregular phrase lengths, natural speech rhythm

### 4. Update singability scoring to reward vocal-appropriate density

Current singability scoring penalises large intervals and rewards rests, but does not
penalise note density. Add:
- Penalise > 6 onsets/bar in 4/4 for vocal templates
- Reward held notes (duration ≥ 1.5 beats)
- Reward explicit rest events in the MIDI output

## What Is Not Changing

- Template structure (intervals, rhythm, durations fields)
- Singer range system and `clamp_to_singer_range`
- Composite scoring weights (30% theory / 70% chromatic)
- Pipeline CLI interface
- Promotion workflow

## Scope

`melody_patterns.py` only. No changes to pipeline orchestration, scoring weights,
or output format beyond the `use_case` filter in template selection.
