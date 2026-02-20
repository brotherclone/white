# Design: Collapse Chord Primitive Phases

## Context

The chord → HR → strum pipeline was designed as three independent refinement passes, each with
its own candidate pool, review YAML, and approved directory. In practice this creates a fan-out
problem: a single approved chord section can spawn multiple named HR variants, each of which
spawns multiple strum variants. Downstream phases (bass, melody, drums) must carry the full
ancestor chain in their filenames, making the assembly loop grid unreadable by Phase 9.

The core insight is that HR and strum are not independent creative decisions — they are
articulation choices that belong *with* the chord, not *after* it. A chord primitive should be
self-contained: voicings + rhythm + articulation, ready to drop into Logic as-is.

## Goals / Non-Goals

- **Goal**: One approved MIDI per section label, containing the complete chord primitive
- **Goal**: Scratch beats available during candidate review without becoming production artifacts
- **Goal**: Downstream phases (drums, bass, melody) reference section labels only — never variant names
- **Goal**: Assembly table is one row per instrument, one column per section
- **Non-Goal**: Preserving existing HR or strum approved directories (breaking change)
- **Non-Goal**: Allowing multiple approved variants per section label (use distinct labels instead)

## Decisions

### Decision: HR + Strum baked into candidate generation (not pre-promotion transform)

Each chord candidate produced by the pipeline is already a complete primitive: the Markov
progression is combined with a randomly-sampled HR distribution and a randomly-sampled strum
pattern during candidate generation. The user reviews and selects one complete primitive per
section.

**Alternative considered**: Apply HR + strum as a pre-promotion transform (user picks chord,
then specifies HR + strum at promote time via flags). Rejected because it requires a more
complex promotion CLI and a two-step mental model for a step that should feel atomic.

**Trade-off**: If the user likes a chord voicing but dislikes the HR/strum pairing, they cannot
swap them independently — they must regenerate candidates. Acceptable given that candidate
generation is fast and the seed is reproducible.

### Decision: Scratch beat uses the simplest genre-appropriate template

The scratch beat auto-generated alongside each chord candidate uses the lowest-energy template
from the genre family inferred from the song proposal. It is not scored, not reviewed, and not
promoted — its only purpose is to provide rhythmic context during chord primitive auditioning.

**Alternative considered**: A pure click track (kick on beat 1 only). Rejected because a
minimal genre-appropriate pattern gives better groove context for evaluating the chord primitive.

### Decision: One promoted file per section label — strict enforcement

Promotion rejects a second approved candidate with the same label. Users who want multiple loops
for a section must use distinct labels (`verse_a`, `verse_b`). This enforces the clean 1:1
mapping that makes the assembly table readable.

**Alternative considered**: Allow `verse_1.mid`, `verse_2.mid` (current behavior). Rejected
because this is exactly the source of the drum-pairing ambiguity the change is solving.

## Migration Plan

- Songs with `harmonic_rhythm/approved/` or `strums/approved/` directories will have those
  directories orphaned. They are not migrated automatically.
- Songs already past chord approval but not yet into drums/bass/melody can re-run chord
  generation with the new pipeline to get complete primitives; their old `chords/approved/`
  files remain valid as-is (they just won't have HR/strum baked in).
- `add-production-plan` must update its spec delta to remove the `harmonic_rhythm/approved/`
  bar-count fallback before implementation.

## Open Questions

- None — design confirmed with user.
