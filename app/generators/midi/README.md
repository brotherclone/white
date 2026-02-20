# MIDI Generators

Music production pipeline tools. Each phase follows the same pattern: **generate → score → human review → promote**.

## Chord Pipeline

Generates chord progression candidates from a song proposal using Markov chains, scores them with music theory metrics + ChromaticScorer, and writes top candidates for human review.

```bash
python -m app.generators.midi.chord_pipeline \
    --thread shrinkwrapped/white-the-breathing-machine-learns-to-sing \
    --song "song_proposal_Black (0x221f20)_sequential_dissolution_v2.yml"
```

| Flag                 | Default                           | Description                      |
|----------------------|-----------------------------------|----------------------------------|
| `--thread`           | *(required)*                      | Shrinkwrapped thread directory   |
| `--song`             | *(required)*                      | Song proposal YAML filename      |
| `--seed`             | 42                                | Random seed for reproducibility  |
| `--num-candidates`   | 200                               | Total candidates to generate     |
| `--top-k`            | 10                                | Candidates to keep for review    |
| `--length`           | 4                                 | Progression length in bars       |
| `--theory-weight`    | 0.3                               | Weight for music theory score    |
| `--chromatic-weight` | 0.7                               | Weight for ChromaticScorer match |
| `--onnx-path`        | `training/data/fusion_model.onnx` | Path to ONNX model               |

**Output:** `<thread>/production/<song_slug>/chords/{candidates/, review.yml}`

## Drum Pipeline

Generates drum pattern candidates for approved chord sections. Reads the chord `review.yml` to determine song sections (verse, chorus, bridge, etc.), maps genre tags to template families, and scores with energy appropriateness + ChromaticScorer.

```bash
python -m app.generators.midi.drum_pipeline \
    --production-dir shrinkwrapped/.../production/black__sequential_dissolution_v2
```

| Flag                 | Default                           | Description                                                                           |
|----------------------|-----------------------------------|---------------------------------------------------------------------------------------|
| `--production-dir`   | *(required)*                      | Song production directory (must contain `chords/review.yml` with approved candidates) |
| `--thread`           | *(auto-detected)*                 | Thread directory (reads from chord review if omitted)                                 |
| `--song`             | *(auto-detected)*                 | Song proposal filename (reads from chord review if omitted)                           |
| `--seed`             | 42                                | Random seed                                                                           |
| `--top-k`            | 5                                 | Candidates per section                                                                |
| `--energy-weight`    | 0.3                               | Weight for energy appropriateness                                                     |
| `--chromatic-weight` | 0.7                               | Weight for ChromaticScorer match                                                      |
| `--energy-override`  | *(none)*                          | Override section energy: `verse=high,chorus=medium`                                   |
| `--genre-override`   | *(none)*                          | Force genre families: `krautrock,ambient`                                             |
| `--onnx-path`        | `training/data/fusion_model.onnx` | Path to ONNX model                                                                    |

**Genre families:** ambient, electronic, krautrock, rock, classical, experimental, folk, jazz

**Section energy defaults:** intro=low, verse=medium, chorus=high, bridge=low, outro=medium

**Output:** `<production-dir>/drums/{candidates/, review.yml}`


## Promote

Promotes approved candidates from any `review.yml` to the `approved/` directory. Works for chords, drums, and strums.

```bash
python -m app.generators.midi.promote_part \
    --review <path-to-review.yml>
```

Edit the `review.yml` first — set `status: approved` and `label: <your label>` on candidates you want to keep. The label becomes the filename in `approved/`.

## Production Plan

Generates `production_plan.yml` — the structural backbone that defines section sequence, bar counts, repeat counts, and vocals intent. This is the bridge between the per-phase loop pipeline and a final song manifest.

```bash
# Generate initial plan (run after approving chords)
python -m app.generators.midi.production_plan \
    --production-dir shrinkwrapped/.../production/black__sequential_dissolution_v2

# Refresh bar counts after re-running upstream phases (preserves human edits)
python -m app.generators.midi.production_plan \
    --production-dir ... --refresh

# Bootstrap a partial manifest from the completed plan
python -m app.generators.midi.production_plan \
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
