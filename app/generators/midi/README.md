# MIDI Generators

Music production pipeline tools. Each phase follows the same pattern: **generate → score → human review → promote**.

## Chord Generator (`chord_generator/`)

The underlying chord generation engine. Parses a corpus of 2,746 MIDI files into a fast columnar database (Polars/Parquet) and two Markov transition graphs (NetworkX), then generates candidate chord progressions using brute-force sampling + music theory scoring.

**Data files** (pre-built, committed):
- `chord_generator/data/chords.parquet` — 1,594 individual chords indexed by key, mode, function, category
- `chord_generator/data/progressions.parquet` — 9,104 chord entries from 1,152 progressions
- `chord_generator/data/chord_transition_graph.pkl` — 22 nodes, 247 edges (chord-level)
- `chord_generator/data/function_transition_graph.pkl` — 93 nodes, 588 edges (Roman-numeral function level, **primary generation graph**)

**Three generation modes** (exposed via `ChordProgressionGenerator`):

| Mode | Method | When to use |
|---|---|---|
| Random | `generate_progression_random()` | Exploration, variety |
| Graph-guided | `generate_progression_graph_guided()` | Theory-coherent progressions |
| Brute-force | `generate_progression_brute_force()` | Production use — scores 1,000+ candidates |

**Internal scoring** (theory-only, used inside brute-force):
- `melody` — rewards stepwise motion in the top voice
- `voice_leading` — rewards minimal total movement between chords
- `variety` — penalises repetition (unique chord ratio)
- `graph_probability` — rewards transitions common in the corpus

Default weights: melody=0.25, voice_leading=0.30, variety=0.15, graph_probability=0.30. These feed into `chord_pipeline.py`'s composite score as the **theory component** (30% theory / 70% Refractor chromatic).

**Rebuild the database** (only needed if the MIDI corpus changes):
```bash
python -m app.generators.midi.chord_generator.build_database
```

See `chord_generator/README.md` for full data schema and query examples.

---

## Chord Pipeline

Generates chord progression candidates from a song proposal using Markov chains, scores them with music theory metrics + Refractor, and writes top candidates for human review.

```bash
python -m app.generators.midi.pipelines.chord_pipeline \
    --thread shrink_wrapped/white-the-breathing-machine-learns-to-sing \
    --song "song_proposal_Black (0x221f20)_sequential_dissolution_v2.yml"
```

| Flag                 | Default                           | Description                      |
|----------------------|-----------------------------------|----------------------------------|
| `--thread`           | *(required)*                      | shrink_wrapped thread directory   |
| `--song`             | *(required)*                      | Song proposal YAML filename      |
| `--seed`             | 42                                | Random seed for reproducibility  |
| `--num-candidates`   | 200                               | Total candidates to generate     |
| `--top-k`            | 10                                | Candidates to keep for review    |
| `--length`           | 4                                 | Progression length in bars       |
| `--theory-weight`    | 0.3                               | Weight for music theory score    |
| `--chromatic-weight` | 0.7                               | Weight for Refractor match |
| `--onnx-path`        | `training/data/refractor.onnx` | Path to ONNX model               |

**Output:** `<thread>/production/<song_slug>/chords/{candidates/, review.yml}`

## Drum Pipeline

Generates drum pattern candidates for approved chord sections. Reads the chord `review.yml` to determine song sections (verse, chorus, bridge, etc.), maps genre tags to template families, and scores with energy appropriateness + Refractor.

```bash
python -m app.generators.midi.pipelines.drum_pipeline \
    --production-dir shrink_wrapped/.../production/black__sequential_dissolution_v2
```

| Flag                 | Default                           | Description                                                                           |
|----------------------|-----------------------------------|---------------------------------------------------------------------------------------|
| `--production-dir`   | *(required)*                      | Song production directory (must contain `chords/review.yml` with approved candidates) |
| `--thread`           | *(auto-detected)*                 | Thread directory (reads from chord review if omitted)                                 |
| `--song`             | *(auto-detected)*                 | Song proposal filename (reads from chord review if omitted)                           |
| `--seed`             | 42                                | Random seed                                                                           |
| `--top-k`            | 5                                 | Candidates per section                                                                |
| `--energy-weight`    | 0.3                               | Weight for energy appropriateness                                                     |
| `--chromatic-weight` | 0.7                               | Weight for Refractor match                                                      |
| `--energy-override`  | *(none)*                          | Override section energy: `verse=high,chorus=medium`                                   |
| `--genre-override`   | *(none)*                          | Force genre families: `krautrock,ambient`                                             |
| `--onnx-path`        | `training/data/refractor.onnx` | Path to ONNX model                                                                    |

**Genre families:** ambient, electronic, krautrock, rock, classical, experimental, folk, jazz

**Section energy defaults:** intro=low, verse=medium, chorus=high, bridge=low, outro=medium

**Output:** `<production-dir>/drums/{candidates/, review.yml}`


## Promote

Promotes approved candidates from any `review.yml` to the `approved/` directory. Works for chords, drums, and strums.

```bash
python -m app.generators.midi.production.promote_part \
    --review <path-to-review.yml>
```

Edit the `review.yml` first — set `status: approved` and `label: <your label>` on candidates you want to keep. The label becomes the filename in `approved/`.

## Production Plan

Generates `production_plan.yml` — the structural backbone that defines section sequence, bar counts, repeat counts, and vocals intent. This is the bridge between the per-phase loop pipeline and a final song manifest.

```bash
# Generate initial plan (run after approving chords)
python -m app.generators.midi.production.production_plan \
    --production-dir shrink_wrapped/.../production/black__sequential_dissolution_v2

# Refresh bar counts after re-running upstream phases (preserves human edits)
python -m app.generators.midi.production.production_plan \
    --production-dir ... --refresh

# Bootstrap a partial manifest from the completed plan
python -m app.generators.midi.production.production_plan \
    --production-dir ... --bootstrap-manifest
```

Edit `production_plan.yml` to set `repeat` counts, `vocals` flags, `sounds_like` references, and reorder sections. The drum pipeline reads it automatically to annotate candidates with `next_section` context.

**Output:** `<production-dir>/production_plan.yml`, optionally `manifest_bootstrap.yml`

## Production Directory Structure

```
production/<song_slug>/
├── production_plan.yml      # song structure: sections, bars, repeats, vocals
├── manifest_bootstrap.yml   # partial manifest (generated on demand)
├── chords/
│   ├── candidates/          # generated MIDI files
│   ├── approved/            # promoted: verse.mid, chorus.mid, etc.
│   └── review.yml
├── drums/
│   ├── candidates/
│   ├── approved/
│   └── review.yml           # includes next_section when plan exists
├── harmonic_rhythm/
│   ├── candidates/
│   ├── approved/
│   └── review.yml
├── strums/
│   ├── candidates/
│   ├── approved/
│   └── review.yml
├── bass/
│   ├── candidates/
│   ├── approved/
│   └── review.yml
└── melody/
    ├── candidates/
    ├── approved/
    └── review.yml
```

## Pipeline Order

1. **Chords** — harmonic foundation (requires song proposal)
2. **Drums** — rhythmic foundation (requires approved chords)
3. **Production Plan** — song structure document (generate after approving chords, edit before drums if possible)
4. **Harmonic Rhythm** — variable chord durations (requires approved chords + drums)
5. **Strums** — chord rhythm patterns (requires approved chords, uses harmonic rhythm if available)
6. **Bass** — bass lines (requires approved chords + drums)
7. **Melody** — vocal lines (requires approved chords + bass)
8. Assembly — combine all layers (future)
