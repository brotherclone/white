# Change: Evolutionary Pattern Breeding

## Why

Every pattern in the library is hand-coded. That means the character of the
generated music is bounded by what a human could think to write as a template
at 9am before a session. The Refractor scores candidates after the fact, but
it can only pick winners from the pool it's given. If the pool doesn't contain
a pattern that sits between "motorik" and "ambient pulse" — something with the
kick of the former and the space of the latter — the Refractor can't find it,
because it doesn't exist.

Evolutionary crossover breaks that ceiling. Instead of writing templates, we
breed them: take two parent patterns, recombine their components, score the
offspring with the Refractor, and let the fittest propagate. Over a few
generations, the population converges on novel patterns that genuinely match
the target concept without any human having to pre-imagine them.

This isn't a replacement for the hand-coded library — the hand-coded patterns
are good seed material and produce interpretable, reproducible results. It's an
expansion: run breeding once per aesthetic target and add the best offspring to
the library as tagged templates. Or run it per-song as an optional generation
mode when the standard pool doesn't score well enough.

## What Changes

### New module: `app/generators/midi/patterns/pattern_evolution.py`

A standalone evolutionary engine that works on any of the three pattern types
(drum, bass, melody). The algorithm:

**1. Seed population**

Start with the existing library filtered by aesthetic tag if `aesthetic_hints`
is available (e.g. `density=sparse` → seed from sparse-tagged patterns only).
Population size: 20–40.

**2. Fitness function**

Score each candidate with the Refractor (`score_batch`), using the song's
concept embedding. Fitness = composite chromatic score. Optionally blend with
a theory score (pattern-level theory metrics already exist in each pipeline).

**3. Crossover**

Patterns are represented as structured grids. Crossover is component-aware,
not byte-level:

- **Drums:** each voice (kick, snare, hh_closed, hh_open, ...) is an
  independent binary/velocity grid. Crossover swaps whole voice rows between
  parents. A child drum pattern might have parent A's kick and hat grid but
  parent B's snare and ghost voices.

- **Bass:** templates define a sequence of `(chord_tone, duration, velocity)`
  events. Crossover splits the sequence at a bar boundary and takes bars 1–N
  from parent A, bars N+1–end from parent B. Voice-leading constraints are
  re-applied post-crossover to prevent illegal leaps.

- **Melody:** same bar-boundary split as bass. Interval sequences are spliced;
  the starting pitch anchor is re-derived from the chord after splicing.

**4. Mutation**

Small random perturbations applied after crossover with probability 0.15:
- Drums: flip one cell in one voice grid; randomise one velocity value ±10
- Bass: shift one note's chord tone to an adjacent chord tone (root→5th, etc.)
- Melody: shift one interval by ±1 semitone; shift one duration by one
  subdivision

**5. Selection**

Tournament selection (k=3): pick 3 at random, keep the highest-fitness one.
Elitism: top 2 of each generation always survive. Run 5–10 generations.

**6. Output**

Top N offspring (default 5) are serialised back to the pattern dataclass format,
tagged `evolved` plus any inherited tags from their highest-fitness parent, and
returned as additional candidates for the pipeline phase. They are scored
alongside hand-coded candidates and compete on equal footing.

Optionally: `--save-evolved` flag writes the best offspring as new named
templates into the pattern library file, so they accumulate across runs.

### Integration with pipeline phases

Each pipeline phase (drum, bass, melody) gains an `--evolve` flag:

```
python -m app.generators.midi.pipelines.drum_pipeline \
    --production-dir <path> \
    --evolve \
    --generations 8 \
    --population 30
```

When `--evolve` is passed, the pipeline:
1. Runs the standard candidate generation (existing behaviour)
2. Runs the evolutionary engine with the same concept embedding
3. Merges evolved candidates into the candidate pool
4. Scores the combined pool with the Refractor
5. Writes all candidates to `candidates/` with `evolved_` prefix for evolved ones

The review.yml will show evolved candidates alongside hand-coded ones, labelled
so the human reviewer knows which is which.

### `pattern_evolution.py` public API

```python
def breed_drum_patterns(
    concept_emb: np.ndarray,
    seed_patterns: list[DrumPattern],
    generations: int = 8,
    population_size: int = 30,
    top_n: int = 5,
) -> list[DrumPattern]: ...

def breed_bass_patterns(
    concept_emb: np.ndarray,
    chord_progression: list[dict],
    seed_patterns: list[BassPattern],
    generations: int = 8,
    population_size: int = 30,
    top_n: int = 5,
) -> list[BassPattern]: ...

def breed_melody_patterns(
    concept_emb: np.ndarray,
    chord_progression: list[dict],
    seed_patterns: list[MelodyPattern],
    generations: int = 8,
    population_size: int = 30,
    top_n: int = 5,
) -> list[MelodyPattern]: ...
```

### Library accumulation (optional, `--save-evolved`)

A separate utility:

```
python -m app.generators.midi.patterns.pattern_evolution \
    --save-evolved shrink_wrapped/.../production/green__last.../drums/candidates \
    --min-score 0.65 \
    --pattern-type drum
```

Reads `evolved_*.mid` files from a candidates directory, deserialises the
embedded metadata (stored as a MIDI text track), and appends patterns above
`min-score` to the pattern library with `evolved` tag. This is how novel
patterns discovered on one song propagate to future songs.

## Impact

- New files:
  - `app/generators/midi/patterns/pattern_evolution.py`
- Modified files:
  - `app/generators/midi/pipelines/drum_pipeline.py` — `--evolve` flag + merge
  - `app/generators/midi/pipelines/bass_pipeline.py` — same
  - `app/generators/midi/pipelines/melody_pipeline.py` — same
- Dependencies: numpy (already present), Refractor (already present in pipelines)
- No breaking changes; `--evolve` is opt-in
- Tests:
  - `tests/generators/midi/patterns/test_pattern_evolution.py`
    - Crossover produces valid pattern structure for each type
    - Mutation stays within musical constraints (chord tones, register)
    - Fitness ordering is preserved across generations
    - `breed_*` returns exactly `top_n` patterns
  - Integration: `--evolve` flag produces merged candidate pool with evolved prefix
