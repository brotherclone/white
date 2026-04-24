# Change: Refactor repo into UV workspace with per-domain packages

## Why

The codebase has grown to ~40 specs and a flat `app/` layout that no longer reflects domain
boundaries. Training, MIDI generation, composition orchestration, and agent ideation are
entangled in a single package with no enforced isolation. A UV workspace restructure gives each
domain its own installable package, explicit dependency graph, and co-located tests — while
keeping the single repo and Modal training unchanged.

## What Changes

- **BREAKING** Import paths change from `from app.generators.midi...` → `from white_generation...`
  (and analogously for all other packages)
- **BREAKING** Root `pyproject.toml` becomes a UV workspace manifest; the `white` package itself
  is no longer directly installable from root
- `app/` is dismantled incrementally; each sub-domain becomes `packages/<name>/`
- Tests move per-package into `packages/<name>/tests/`; root `tests/` shrinks to
  `tests/integration/` (cross-package scenarios only)
- 10 new packages created, each with its own `pyproject.toml` and `src/white_<name>/` layout
- `web/` (Next.js) becomes `packages/client/` — no import namespace change, it is JavaScript
- Two greenfield packages: `diary` (MDX song lifecycle) and `production` (Logic/audio world)
- `app/reference/` static data moves into `ideation` (agents consume it)
- `app/util/shrinkwrap_chain_artifacts.py` moves into `composition`
- Training stays on Modal; `training/` source moves into `packages/training/`

## Package Inventory

| Package | Importable name | Source today |
|---|---|---|
| `core` | `white_core` | `app/structures/` |
| `training` | `white_training` | `training/` |
| `extraction` | `white_extraction` | `app/extractors/`, `app/util/` (audio/MIDI utils) |
| `analysis` | `white_analysis` | `training/refractor.py` + scoring wrapper |
| `generation` | `white_generation` | `app/generators/midi/patterns/`, `pipelines/` |
| `composition` | `white_composition` | `app/generators/midi/production/`, shrinkwrap util |
| `ideation` | `white_ideation` | `app/agents/`, `app/reference/` |
| `diary` | `white_diary` | GREENFIELD — MDX song lifecycle |
| `api` | `white_api` | `app/tools/candidate_server.py` + production server |
| `client` | n/a (Next.js) | `web/` |

## Impact

- Affected specs: `pipeline-orchestration`, `composition-proposal`, `chain-artifacts`,
  `production-plan`, `song-completion-dashboard`, `candidate-browser`, `candidate-browser-web`,
  and all generation specs (chord, drum, bass, melody, quartet, lyric)
- Affected code: every Python import in `app/`, `training/`, `tests/`, `scripts/`
- New files: 10 `pyproject.toml` files, root workspace manifest, `pytest.ini` per package
- No change to Modal training infrastructure; no change to Next.js build
- `white.egg-info/` removed; replaced by per-package dist-info

## Constraints

- The composition pipeline (chord → drum → bass → melody → promote → plan) MUST remain
  functional throughout migration; no package migration shall leave it broken
- Each package is migrated in its own PR, merged to develop before the next begins
- No compatibility shims or re-export stubs; cut over cleanly
- `training/` data files (`.pt`, `.onnx`, parquet) remain in place; only source moves
