# Pipeline Orchestration Design Report

**Spike:** `spike-pipeline-orchestration-design`
**Date:** 2026-03-15
**Status:** Complete — informs follow-on `refactor-pipeline-orchestration` proposal

---

## 1. Data Flow Map

### 1.1 Phase Dependency Graph

```
Song YAML (yml/<slug>.yml)
     │
     ▼
init_production.py ──────────────► initial_proposal.yml
     │                               (sounds_like, color, concept,
     │                                singer, key, bpm, time_sig, genres, mood)
     │
     ▼
chord_pipeline.py ─── reads: Song YAML, manifest.yml, initial_proposal.yml
     │                writes: chords/candidates/*.mid
     │                        chords/review.yml
     │                        (fields: pipeline, song_proposal, thread, key,
     │                                 bpm, color, time_sig, singer,
     │                                 scoring_weights, candidates[])
     ▼
drum_pipeline.py ─── reads: chords/review.yml, production_plan.yml (next_section)
     │               writes: drums/candidates/*.mid
     │                        drums/review.yml
     │                        (fields: pipeline, bpm, color, time_sig,
     │                                 scoring_weights, candidates[])
     ▼
bass_pipeline.py ─── reads: chords/review.yml, drums/review.yml
     │               writes: bass/candidates/*.mid
     │                        bass/review.yml
     │                        (fields: pipeline, bpm, color, time_sig,
     │                                 scoring_weights, candidates[])
     ▼
melody_pipeline.py ─── reads: chords/review.yml, drums/review.yml
     │                 writes: melody/candidates/*.mid
     │                          melody/review.yml
     │                          (fields: pipeline, bpm, color, time_sig, singer,
     │                                   scoring_weights, candidates[])
     ▼
lyric_pipeline.py ─── reads: chords/review.yml → Song YAML (via production_plan),
     │                         initial_proposal.yml
     │                 writes: melody/lyrics_review.yml
     │                          (fields: pipeline, bpm, time_sig, color, seed,
     │                                   model, scoring_weights, candidates[])
     ▼
production_plan.py ─── reads: chords/review.yml (all phases indirectly)
     │                  writes: production_plan.yml
     │                           (fields: title, bpm, time_sig, key, color, genres,
     │                                    mood, concept, sounds_like, sections[])
     ▼
composition_proposal.py ─── reads: chords/review.yml, initial_proposal.yml, Song YAML
     │                       writes: composition_proposal.yml
     │                                (fields: proposed_by, generated, color_target,
     │                                          title, loop_inventory, sounds_like,
     │                                          proposed_sections[], rationale)
     ▼
assembly_manifest.py ─── reads: production_plan.yml, all review.ymls
                          writes: manifest_bootstrap.yml, assembly_manifest.yml
```

### 1.2 sounds_like Trace

| File / Function               | Role                                       | Reads From                                        | Writes To                        |
|-------------------------------|--------------------------------------------|---------------------------------------------------|----------------------------------|
| Song YAML (`yml/<slug>.yml`)  | Optional author hint                       | —                                                 | `sounds_like: []` (often absent) |
| `init_production.py`          | **Source of truth** — generates via Claude | Song YAML                                         | `initial_proposal.yml`           |
| `initial_proposal.yml`        | Persistent store (pre-pipeline)            | —                                                 | —                                |
| `chord_pipeline.py:543`       | Consumer                                   | `initial_proposal.yml` → `raw_proposal` fallback  | `artist_context` local var only  |
| `drum_pipeline.py`            | **Not consumed**                           | (chord_review has no sounds_like)                 | —                                |
| `bass_pipeline.py`            | **Not consumed**                           | (chord_review has no sounds_like)                 | —                                |
| `melody_pipeline.py`          | **Not consumed**                           | (chord_review has no sounds_like)                 | —                                |
| `lyric_pipeline.py:994`       | Consumer                                   | `initial_proposal.yml` → chord_review fallback    | `meta["sounds_like"]` local      |
| `composition_proposal.py:117` | Consumer + re-writer                       | `initial_proposal.yml` → `raw.get("sounds_like")` | `composition_proposal.yml`       |
| `production_plan.py:64`       | Passive carrier                            | `load_song_proposal()` (from proposal YAML)       | `production_plan.yml`            |
| `assembly_manifest.py:962`    | Passive carrier                            | `plan.sounds_like`                                | `manifest_bootstrap.yml`         |

**Problem**: drum, bass, and melody phases do not have access to `sounds_like` at scoring time. This is acceptable for chromatic scoring (which only needs `concept_emb`) but would block any future phase that needs artist context (e.g., a timbre-matching step).

### 1.3 concept and color Trace

| Phase                  | Key name     | Source                                 | Fallback                            |
|------------------------|--------------|----------------------------------------|-------------------------------------|
| `chord_pipeline`       | `concept`    | Song YAML direct                       | `manifest.yml` (artist intent)      |
| `chord_pipeline`       | `color_name` | Song YAML `rainbow_color.color_name`   | None                                |
| `drum_pipeline`        | `concept`    | chord_review (not present!)            | `f"{color_name} chromatic concept"` |
| `drum_pipeline`        | `color_name` | `chord_review["color"]`                | `"White"`                           |
| `bass_pipeline`        | `concept`    | chord_review (not present!)            | `f"{color_name} chromatic concept"` |
| `bass_pipeline`        | `color_name` | `chord_review["color"]`                | `"White"`                           |
| `melody_pipeline`      | `concept`    | chord_review (not present!)            | `f"{color_name} chromatic concept"` |
| `melody_pipeline`      | `color_name` | `chord_review["color"]`                | `"White"`                           |
| `lyric_pipeline`       | `concept`    | `production_plan.load_song_proposal()` | empty string → fallback             |
| `lyric_pipeline`       | `color`      | chord_review                           | `""`                                |
| `composition_proposal` | `concept`    | Song YAML (via chord_review path)      | `""`                                |

**Critical finding**: drum, bass, and melody phases use a degraded concept embedding. The real concept text (`"A song about..."`) is available in the Song YAML and in `manifest.yml`, but chord_review.yml does not carry it. These three phases fall back to `f"{color_name} chromatic concept"` (e.g., `"Red chromatic concept"`), which produces a meaningful but less specific embedding than the actual concept prose.

**Key naming inconsistency**: `chord_pipeline` uses `color_name`; all other phases use `color`. These are the same value. Phases downstream of chord_pipeline access it from chord_review.yml's `color` field and rename it to `color_name` in their local `song_info` dict.

### 1.4 Dependency Graph (ASCII)

```
Song YAML ──┬──────────────────────────────────────┐
            │                                       │
            ▼                                       │
    initial_proposal.yml ◄── init_production.py     │
            │                                       │
            ├──────────► chord_pipeline ◄───────────┘
            │                   │
            │            chords/review.yml ◄─── manifest.yml (concept fallback in chord only)
            │                   │
            │          ┌────────┼────────┐
            │          ▼        ▼        ▼
            │        drum     bass    melody
            │          │        │        │
            │    drums/     bass/    melody/
            │    review     review   review
            │          └────────┴────────┘
            │                   │
            └──► lyric_pipeline ◄┘ (reads chord_review → Song YAML via production_plan)
                        │
                  lyrics_review.yml
                        │
            production_plan.py ◄── (all approved reviews)
                        │
              production_plan.yml
                        │
            composition_proposal.py ◄── initial_proposal.yml
                        │
             composition_proposal.yml
                        │
              assembly_manifest.py
                        │
              manifest_bootstrap.yml
```

---

## 2. load_song_proposal Audit

### 2.1 Implementations Found

**Implementation A — `chord_pipeline.py:162`**

```python
def load_song_proposal(thread_dir: Path, song_filename: str) -> dict:
```

- **Files read**: `thread_dir / "yml" / song_filename` + `thread_dir / "manifest.yml"` (concept fallback)
- **Fields returned**: `key_root`, `mode`, `bpm`, `time_sig` (tuple e.g. `(4, 4)`), `concept`, `color_name`, `singer`, `song_filename`, `thread_dir`, `raw_proposal`
- **Callers**: `run_chord_pipeline()` only
- **Notable**: Only implementation that reads `manifest.yml`. Only one returning `raw_proposal`. Returns `time_sig` as a tuple.

**Implementation B — `production_plan.py:228`**

```python
def load_song_proposal(proposal_path: Path) -> dict:
```

- **Files read**: `proposal_path` directly (caller supplies full path)
- **Fields returned**: `title`, `bpm`, `time_sig` (string e.g. `"4/4"`), `key`, `color`, `genres`, `mood`, `concept`
- **Callers**: `lyric_pipeline._find_and_load_proposal()`, `composition_proposal.load_song_proposal_data()`, `production_plan` CLI directly
- **Notable**: No `singer`, no `raw_proposal`, no `thread_dir`. Returns `time_sig` as string. Returns `color` not `color_name`.

**Implementation C — `lyric_pipeline._find_and_load_proposal(production_dir)`**

```python
def _find_and_load_proposal(production_dir: Path) -> dict:
```

- **Files read**: `chords/review.yml` → extracts `thread` + `song_proposal` → constructs path → calls `production_plan.load_song_proposal()`
- **Fields returned**: everything from B + `singer` (from chord_review) + `sounds_like` (empty default, patched in caller)
- **Callers**: `run_lyric_pipeline()` only
- **Notable**: Indirect — navigates from production dir to thread dir using chord_review.yml as a pointer.

**Implementation D — `composition_proposal.load_song_proposal_data(production_dir)`**

```python
def load_song_proposal_data(production_dir: Path) -> dict:
```

- **Files read**: `chords/review.yml` → calls `production_plan.load_song_proposal()` + reads `initial_proposal.yml` for sounds_like
- **Fields returned**: everything from B + `singer` + `sounds_like`
- **Callers**: `generate_composition_proposal()` only

### 2.2 Field Divergence Matrix

| Field           | chord_pipeline (A) | production_plan (B) | lyric (C)             | composition (D)        |
|-----------------|--------------------|---------------------|-----------------------|------------------------|
| `color`         | ✗ (`color_name`)   | ✓                   | ✓                     | ✓                      |
| `color_name`    | ✓                  | ✗                   | ✗                     | ✗                      |
| `concept`       | ✓                  | ✓                   | ✓ (often `""`)        | ✓                      |
| `bpm`           | ✓                  | ✓                   | ✓                     | ✓                      |
| `time_sig`      | tuple `(4,4)`      | string `"4/4"`      | string `"4/4"`        | string `"4/4"`         |
| `key`           | `key_root`+`mode`  | `key` (combined)    | `key`                 | `key`                  |
| `singer`        | ✓                  | ✗                   | ✓ (from chord_review) | ✓ (from chord_review)  |
| `sounds_like`   | via separate call  | via proposal YAML   | patched after load    | ✓ via initial_proposal |
| `genres`        | ✗                  | ✓                   | ✓                     | ✓                      |
| `mood`          | ✗                  | ✓                   | ✓                     | ✓                      |
| `raw_proposal`  | ✓                  | ✗                   | ✗                     | ✗                      |
| `thread_dir`    | ✓                  | ✗                   | ✗                     | ✗                      |
| `song_filename` | ✓                  | ✗                   | ✗                     | ✗                      |
| `title`         | ✗                  | ✓                   | ✓                     | ✓                      |

### 2.3 Canonical Loader Feasibility

One canonical loader can satisfy all callers with a single interface change:

```python
def load_song_proposal(proposal_path: Path, thread_dir: Path | None = None) -> dict
```

Minimum required output fields: `title`, `bpm`, `time_sig` (string), `key`, `color`, `concept`, `genres`, `mood`, `singer`, `sounds_like`.

Divergences that must be preserved or resolved:
- **`time_sig` format**: chord_pipeline's callers downstream expect a tuple for time signature arithmetic. Easiest fix: always return string `"4/4"` and fix the one caller that does arithmetic.
- **`key` vs `key_root`+`mode`**: chord_pipeline parses into components for Markov graph construction. The canonical loader should return both the combined string and parsed components.
- **manifest.yml fallback**: Only chord_pipeline reads it. Since concept fallback is only needed in that phase, the canonical loader can accept an optional `manifest_path` argument or the caller can patch `concept` after the fact.

**Conclusion**: One canonical loader is feasible. The divergence is bounded and explicit.

---

## 3. Phase Review Schema Audit

### 3.1 Per-Phase Schemas

**`chords/review.yml`**
```yaml
pipeline: chord_pipeline
song_proposal: <filename>.yml        # pointer back to source
thread: /path/to/thread/dir          # pointer back to thread
key: "F# minor"
bpm: 96
color: Red
time_sig: "4/4"
singer: Gabriel
scoring_weights: {theory: 0.3, chromatic: 0.7}
candidates:
  - label: verse
    status: approved|pending|rejected
    midi_file: candidates/<label>.mid
    score: 0.842
    theory_score: 0.71
    chromatic_score: 0.89
    hr_distribution: {verse: 2, chorus: 4}   # bars per section
```

**`drums/review.yml`**
```yaml
pipeline: drum_pipeline
bpm: 96
color: Red
time_sig: "4/4"
scoring_weights: {energy: 0.3, chromatic: 0.7}
candidates:
  - label: verse
    status: approved|pending|rejected
    midi_file: candidates/<label>.mid
    score: 0.791
    energy_score: 0.68
    chromatic_score: 0.84
```

**`bass/review.yml`**
```yaml
pipeline: bass_pipeline
bpm: 96
color: Red
time_sig: "4/4"
scoring_weights: {theory: 0.3, chromatic: 0.7}
candidates:
  - label: verse
    status: approved|pending|rejected
    midi_file: candidates/<label>.mid
    score: 0.803
    theory_score: 0.74
    chromatic_score: 0.83
```

**`melody/review.yml`**
```yaml
pipeline: melody_pipeline
bpm: 96
color: Red
time_sig: "4/4"
singer: Gabriel
scoring_weights: {theory: 0.3, chromatic: 0.7}
candidates:
  - label: verse
    status: approved|pending|rejected
    midi_file: candidates/<label>.mid
    score: 0.811
    theory_score: 0.79
    chromatic_score: 0.83
```

**`melody/lyrics_review.yml`**
```yaml
pipeline: lyric_pipeline
bpm: 96
time_sig: "4/4"
color: Red
seed: 42
model: claude-sonnet-4-6
scoring_weights: {theory: 0.3, chromatic: 0.7}
candidates:
  - label: verse
    status: approved|pending|rejected
    file: candidates/<label>.txt
    score: 0.856
    theory_score: 0.81
    chromatic_score: 0.88
```

### 3.2 Shared vs. Phase-Specific Fields

**Shared across all reviews**: `pipeline`, `bpm`, `color`, `time_sig`, `scoring_weights`, `candidates[].label`, `candidates[].status`, `candidates[].score`

**Phase-specific**:
- `chords/review.yml`: `song_proposal`, `thread`, `key`, `singer`, `hr_distribution` (per candidate)
- `drums/review.yml`: `energy_score` (not `theory_score`)
- `melody/review.yml`: `singer`
- `lyrics_review.yml`: `seed`, `model`, `file` (not `midi_file`)

**`promote_part.py` reads from candidates**: `status`, `midi_file` or `file`, `label`. Does NOT read any song-level fields from the review — fully decoupled.

### 3.3 Shared Base Schema Assessment

A shared base schema would help tooling (e.g., a pipeline status dashboard) but risks over-constraining legitimate per-phase differences (`energy_score` vs `theory_score`, `file` vs `midi_file`).

**Recommendation**: Define a shared header schema (not a base class), document it in `openspec/specs/`, and validate it with a lightweight checker. Leave candidate-level fields free-form — they are phase-internal.

Shared header fields to standardize:
```yaml
pipeline: <string>
bpm: <int>
color: <string>
time_sig: <string>   # always "N/N" format
scoring_weights: <dict>
```

Fields that should move to `song_context.yml` and be dropped from review files:
- `song_proposal` (chords only)
- `thread` (chords only)
- `key` (chords only — needed by bass/melody too but currently lost)
- `singer` (chords + melody)

---

## 4. Production Context Design

### 4.1 Proposed `song_context.yml` Schema

`song_context.yml` would live at `production/<song_slug>/song_context.yml` and serve as the single source of truth for static song metadata throughout the pipeline.

```yaml
# Generated by init_production.py — do not edit manually
# Fields here are stable for the life of the production directory.

schema_version: "1"
generated: "2026-03-15T..."
proposed_by: claude

# Identity
title: "The Breathing Machine Learns to Sing"
song_slug: the-breathing-machine-learns-to-sing
song_proposal: "the-breathing-machine-learns-to-sing.yml"  # relative to thread yml/
thread: /absolute/path/to/thread/dir

# Chromatic
color: Red
concept: |
  A song about a machine discovering memory...

# Musical
key: "F# minor"
bpm: 96
time_sig: "4/4"
singer: Gabriel

# Creative context (Claude-generated)
sounds_like:
  - "Grouper"
  - "The Caretaker"
  - "Stars of the Lid"
genres:
  - "ambient"
  - "experimental"
mood: "meditative"

# Status tracking (updated by each phase)
phases:
  chords: pending    # pending | complete | skipped
  drums: pending
  bass: pending
  melody: pending
  lyrics: pending
  composition_proposal: pending
```

### 4.2 Backward Compatibility

Existing production dirs (violet, blue) can have `song_context.yml` bootstrapped from their existing files without re-running any MIDI phase:

**Migration path per existing dir:**
1. Read `chords/review.yml` → extract: `song_proposal`, `thread`, `key`, `bpm`, `color`, `time_sig`, `singer`
2. Resolve `song_proposal` path → read Song YAML → extract: `title`, `concept`, `genres`, `mood`
3. Read `initial_proposal.yml` if present → extract `sounds_like`; else `sounds_like: []`
4. Detect which phases are complete by checking for approved candidates in each review.yml
5. Write `song_context.yml`

This is fully non-destructive — existing review.ymls stay unchanged. A migration script (`migrate_production_dir.py`) can automate this.

### 4.3 Relationship to Existing Files

| File                       | Current Role                                 | After song_context.yml                                                                                    |
|----------------------------|----------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| `song_context.yml`         | (new)                                        | Canonical static metadata + phase status                                                                  |
| `initial_proposal.yml`     | `sounds_like` store                          | **Superseded** — data moves to `song_context.yml`                                                         |
| `chords/review.yml`        | Carries song_proposal + thread pointer       | Retain for candidate history; drop `song_proposal`/`thread` fields                                        |
| `production_plan.yml`      | Section arrangement + sounds_like + metadata | Retain; keep arrangement data; drop overlapping fields (`bpm`, `color`, `time_sig`, `key`, `sounds_like`) |
| `composition_proposal.yml` | Creative deliverable + loop inventory        | Retain fully as creative artifact                                                                         |

`initial_proposal.yml` can be deprecated once `song_context.yml` is in place. Existing readers (`load_initial_proposal`) can transparently fall back to `song_context.yml` during the transition.

---

## 5. Migration Strategy

### Phase 1 — Bootstrap (no breaking changes)

1. Extend `init_production.py` to write `song_context.yml` instead of (or alongside) `initial_proposal.yml`
2. Write `load_song_context(production_dir) -> dict` returning the full context dict
3. Write `migrate_production_dir.py` to generate `song_context.yml` for existing dirs
4. Update `load_initial_proposal()` to transparently read from `song_context.yml` when `initial_proposal.yml` is absent

At this point all phases can read `song_context.yml` without any pipeline changes. Zero regressions.

### Phase 2 — Consolidate load_song_proposal (one breaking change, contained)

1. Add `production_plan.load_song_proposal_unified(proposal_path, singer=None, sounds_like=None) -> dict` returning a standardized dict with all fields (normalized `time_sig` string, both `color` and `key` in canonical forms)
2. Migrate `chord_pipeline.load_song_proposal` to call the unified version (parse `key` into components internally, not in the loader)
3. Retire implementation C (`lyric_pipeline._find_and_load_proposal`) — replace with `load_song_context` + `load_song_proposal_unified`
4. Retire implementation D (`composition_proposal.load_song_proposal_data`) — same replacement

### Phase 3 — Propagate concept to drum/bass/melody (correctness fix)

The drum, bass, and melody phases currently fall back to `f"{color_name} chromatic concept"` because `concept` is not in `chord_review.yml`. With `song_context.yml` in place, these phases can load it directly. Update `_load_song_info()` in each of these three pipelines to:
1. Call `load_song_context(production_dir)`
2. Use `song_context["concept"]` for the concept embedding
3. Keep the fallback for production dirs that predate `song_context.yml`

### Phase 4 — Clean up review.yml redundancy (cosmetic, low priority)

Remove `song_proposal` and `thread` from `chords/review.yml` — these are now in `song_context.yml`. Update `_find_and_load_proposal` callers to read from `song_context.yml` instead. This is cosmetic — it only reduces duplication.

---

## 6. Recommended Refactor Scope

### What to Change (ordered by value/risk)

| Priority | Change | Value | Risk | Estimate |
|---|---|---|---|---|
| 1 | Write `song_context.yml` in `init_production.py` | Unblocks all phases | None | ~1h |
| 2 | Write `migrate_production_dir.py` | Existing dirs get context | Low | ~1h |
| 3 | Propagate `concept` to drum/bass/melody via `song_context.yml` | Correct scoring | Low | ~2h |
| 4 | Unified `load_song_proposal` | Eliminates 3 divergent implementations | Medium | ~3h |
| 5 | Deprecate `initial_proposal.yml` (alias to song_context.yml) | Removes duplication | Low | ~1h |
| 6 | Drop `song_proposal`/`thread` from chord_review | Cosmetic | Low | ~30m |

**Total estimate: ~8.5 hours (~2 sessions)**

### What to Leave Alone

- **Per-phase review YAML schemas** — each phase's candidate-level fields are genuinely different and the asymmetry is correct. Do not try to unify them.
- **`promote_part.py`** — already generic and correct; touches only candidate-level data.
- **`composition_proposal.yml`** — creative artifact with its own structure; keep it separate from `song_context.yml`.
- **`production_plan.yml`** — the arrangement/sections data has no overlap with static song metadata after `song_context.yml` takes over the shared fields.

### Decision Gate (per tasks.md 5.2)

The proposed refactor is bounded at **~2 sessions** and is highly incremental — each phase can be merged independently without breaking the pipeline. This is **not a big-bang rewrite**. Phase 1 (song_context.yml bootstrap) is a pure addition that delivers value immediately. Phases 2–4 are refinements that can be done opportunistically.

**Recommendation**: Open `refactor-pipeline-orchestration` as a single spec with the 6 tasks above listed in priority order. Implement and merge Phase 1 (tasks 1–2) first, then proceed through the rest sequentially, validating the full test suite between each task.

---

## Appendix: Key File Locations

| Component            | Path                                                     |
|----------------------|----------------------------------------------------------|
| init_production      | `app/generators/midi/production/init_production.py`      |
| chord_pipeline       | `app/generators/midi/pipelines/chord_pipeline.py`        |
| drum_pipeline        | `app/generators/midi/pipelines/drum_pipeline.py`         |
| bass_pipeline        | `app/generators/midi/pipelines/bass_pipeline.py`         |
| melody_pipeline      | `app/generators/midi/pipelines/melody_pipeline.py`       |
| lyric_pipeline       | `app/generators/midi/pipelines/lyric_pipeline.py`        |
| production_plan      | `app/generators/midi/production/production_plan.py`      |
| composition_proposal | `app/generators/midi/production/composition_proposal.py` |
| assembly_manifest    | `app/generators/midi/production/assembly_manifest.py`    |
| promote_part         | `app/generators/midi/production/promote_part.py`         |
| retrieve_samples     | `app/generators/midi/production/retrieve_samples.py`     |
