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

## Harmonic Rhythm Pipeline

Generates variable chord duration distributions on a half-bar grid. Aligns chord changes with approved drum accent positions and scores with ChromaticScorer temporal mode. Section length can expand or contract.

```bash
python -m app.generators.midi.harmonic_rhythm_pipeline \
    --production-dir shrinkwrapped/.../production/black__sequential_dissolution_v2
```

| Flag                 | Default                           | Description                                                                       |
|----------------------|-----------------------------------|-----------------------------------------------------------------------------------|
| `--production-dir`   | *(required)*                      | Song production directory (must contain `chords/approved/` and `drums/approved/`) |
| `--seed`             | 42                                | Random seed                                                                       |
| `--top-k`            | 20                                | Candidates per section                                                            |
| `--alignment-weight` | 0.3                               | Weight for drum accent alignment                                                  |
| `--chromatic-weight` | 0.7                               | Weight for ChromaticScorer temporal match                                         |
| `--onnx-path`        | `training/data/fusion_model.onnx` | Path to ONNX model                                                                |

**Duration grid:** 0.5-bar increments. Min 0.5 bars per chord, max total = N * 2.0 bars.

**Output:** `<production-dir>/harmonic_rhythm/{candidates/, review.yml}`

## Strum Pipeline

Applies rhythm patterns to approved chord MIDI files — same harmony, different time feel. If approved harmonic rhythm exists, chords get variable durations; otherwise defaults to 1 bar per chord. No ChromaticScorer scoring since harmony is unchanged.

```bash
python -m app.generators.midi.strum_pipeline \
    --production-dir shrinkwrapped/.../production/black__sequential_dissolution_v2
```

| Flag               | Default      | Description                                                                                      |
|--------------------|--------------|--------------------------------------------------------------------------------------------------|
| `--production-dir` | *(required)* | Song production directory (must contain `chords/approved/`)                                      |
| `--mode`           | `per-chord`  | `per-chord` (each chord x each pattern), `progression` (full sequence x each pattern), or `both` |
| `--patterns`       | *(all)*      | Comma-separated pattern names to include: `quarter,eighth,arp_up`                                |

**4/4 patterns:** whole, half, quarter, eighth, push, arp_up, arp_down

**7/8 patterns:** whole, grouped_322, grouped_223, eighth, arp_up, arp_down

**Output:** `<production-dir>/strums/{candidates/, review.yml}`

## Promote

Promotes approved candidates from any `review.yml` to the `approved/` directory. Works for chords, drums, and strums.

```bash
python -m app.generators.midi.promote_chords \
    --review <path-to-review.yml>
```

Edit the `review.yml` first — set `status: approved` and `label: <your label>` on candidates you want to keep. The label becomes the filename in `approved/`.

## Production Directory Structure

```
production/<song_slug>/
├── chords/
│   ├── candidates/          # generated MIDI files
│   ├── approved/            # promoted: verse.mid, chorus.mid, etc.
│   └── review.yml
├── drums/
│   ├── candidates/
│   ├── approved/
│   └── review.yml
├── harmonic_rhythm/
│   ├── candidates/
│   ├── approved/
│   └── review.yml
├── strums/
│   ├── candidates/
│   ├── approved/
│   └── review.yml
├── bass/                    # future
└── assembly/                # future
```

## Pipeline Order

1. **Chords** — harmonic foundation (requires song proposal)
2. **Drums** — rhythmic foundation (requires approved chords)
3. **Harmonic Rhythm** — variable chord durations (requires approved chords + drums)
4. **Strums** — chord rhythm patterns (requires approved chords, uses harmonic rhythm if available)
5. Bass — bass lines (future)
6. Melody + Lyrics — vocal lines (future)
7. Assembly — combine all layers (future)
