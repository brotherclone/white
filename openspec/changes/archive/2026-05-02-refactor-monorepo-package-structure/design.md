## Context

Single flat repo with active music production underway (90+ min of completed tracks). The
pipeline must never be fully broken mid-migration. UV workspace is already the lock-file
toolchain (`uv.lock` present at root). This change formalises that into a proper workspace.

## Goals / Non-Goals

- **Goals:** Enforce package boundaries; make each domain independently installable and testable;
  give imports self-documenting namespaces; enable per-package CI targets.
- **Non-Goals:** Splitting into separate repos; changing Modal training infra; changing Next.js
  build pipeline; adding new features (diary and production packages are scaffolded but left
  empty until later specs).

## Decisions

### Import namespace: `white_<name>` (flat underscore) not `white.<name>` (dot namespace)

`white.core`, `white.generation` etc. requires Python namespace packages (no `__init__.py` at
the `white/` level in each package, or PEP 420 `pkgutil`-style). This is fragile: a single
package that accidentally ships a `white/__init__.py` silently shadows the others.

`white_core`, `white_generation` etc. is a simple string prefix — each package is a fully
independent importable module, no namespace magic required. Slightly more verbose in import
statements but zero fragility.

**Decision: `white_<name>` flat namespace.**

### src layout

Each package uses `packages/<name>/src/white_<name>/` (PEP 517 src layout). This prevents
accidental imports of the un-installed source tree and is the UV workspace standard.

### Tests: per-package, with a root integration layer

```
packages/<name>/
└── tests/          # unit + integration tests for this package only
tests/
└── integration/    # cross-package end-to-end scenarios
```

Root `pytest.ini` discovers both. Per-package `pytest.ini` (or `[tool.pytest.ini_options]` in
that package's `pyproject.toml`) lets you run `uv run --package white-generation pytest` for
isolation.

### Migration order (dependency depth, shallowest first)

1. `core` — no internal deps; everything imports it. Move first so other packages can depend on
   the new importable.
2. `training` — mostly isolated from `app/`; imports `white_core` after step 1.
3. `extraction` — imports `white_core`; audio/MIDI utils imported by `analysis` and `generation`.
4. `analysis` — imports `white_core`, `white_training` (Refractor), `white_extraction`.
5. `generation` — imports `white_core`, `white_analysis` (scoring). Patterns + pipelines.
6. `composition` — imports `white_core`, `white_generation`. **Most careful** — active production.
7. `ideation` — imports `white_core`, `white_composition` (plan data), `white_analysis`.
8. `diary` — greenfield; imports `white_composition` event hooks. Scaffold only at this stage.
9. `api` — imports `white_composition`, `white_generation`. Candidate server + production server.
10. `client` — Next.js; no Python imports; moves `web/` → `packages/client/`.

### Root pyproject.toml becomes workspace manifest

```toml
[tool.uv.workspace]
members = [
    "packages/core",
    "packages/training",
    ...
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.4.2",
    "ruff>=0.14.3",
    "black>=25.9.0",
    "pre-commit>=4.3.0",
]
```

Shared dev tooling (ruff, black, pytest, pre-commit) lives at workspace root. Each package's
`pyproject.toml` declares only its runtime dependencies and cross-package deps as
`white-core = {workspace = true}`.

### Tricky file placements

- `app/reference/` — static reference data (biographical, books, etc.) used by agents.
  Goes into `packages/ideation/src/white_ideation/reference/`.
- `app/util/shrinkwrap_chain_artifacts.py` → `packages/composition/`.
- `app/util/generate_negative_constraints.py` → `packages/extraction/` (training-data utility).
- `app/generators/midi/style_reference.py` → `packages/generation/`.
- `run_white_agent.py` (root entry point) → update imports in-place; stays at root as a script.

### Existing `white.egg-info/` and editable install

Currently installed as `pip install -e .`. After migration, each package is installed via
`uv sync` (workspace). The `.egg-info/` directory is removed. `sitecustomize.py` at root may
need updating if it manipulates sys.path.

## Risks / Trade-offs

- **Mid-migration broken imports** → Mitigated by strict migration order and per-PR CI gate.
  Never remove the old location until all callers are updated.
- **`training/data/` large binaries** → `.pt`, `.onnx`, parquet files stay in place; only
  `.py` source moves. No git LFS changes.
- **`tests/` root conftest.py** → Must be migrated carefully; any shared fixtures that span
  packages stay in root `tests/conftest.py`.

## Migration Plan

For each package migration:
1. `mkdir -p packages/<name>/src/white_<name>` + write `pyproject.toml`
2. Move source files (keep old location until PR merges)
3. Update all `from app.<old>...` imports across the repo to `from white_<name>...`
4. Move corresponding tests to `packages/<name>/tests/`
5. Run `uv sync` + full pytest suite
6. Delete old source location
7. Open PR → merge → move to next package

**Rollback:** Each PR is reversible independently. The dependency order means rolling back
package N does not break packages 1..N-1 (they have no dependency on N).

## Data / Asset Placements (resolved)

- `app/generators/midi/chord_generator/data/` (1,594-chord JSON) → moves with source into
  `packages/generation/src/white_generation/chord_generator/data/`. Tightly coupled to the
  chord generator; treat as package data, not a standalone asset.
- `staged_raw_material/` — Python code (`__init__.py`, `cleanup_evp_intermediates.py`) moves to
  `packages/extraction/src/white_extraction/staged_raw_material/`. The raw audio directories
  (`01_01/`, `01_02/`, …) are binary data assets and stay at repo root (or move to
  `packages/extraction/data/staged_raw_material/` if we want co-location). Either way they are
  never imported as Python; only the script moves.
- `staged_raw_material` is currently listed as a package in the root `pyproject.toml`
  `[tool.setuptools]`; that entry is removed when the workspace manifest replaces it.

## Resolved During Implementation

- **`sitecustomize.py`** stays at root — Python auto-imports it; suppresses aifc/sunau warnings.
- **`uv sync` does not auto-install workspace members** — must run `uv pip install -e packages/<name>`
  after each package scaffold, or `uv sync --all-packages` from root. Add to developer docs.
- **`--import-mode=importlib`** added to `pytest.ini` — avoids namespace collision when both
  `tests/` and `packages/*/tests/` have `artifacts/`, `concepts/` etc. subdirectories.
- **Infranym tools** (`infranym_audio_encoder`, `infranym_text_tools`, `infranym_midi_tools`,
  all encoding algorithms) moved to `white_core/util/infranym/` and `white_core/util/encodings/`
  rather than `ideation` — they already imported from `white_core` and have no agent logic.
- **`white_facet_prompts.py`** moved to `white_core/reference/prompts/` — only imports
  `white_core` enums; it's reference data, not agent code.
- **`gaming_tools.py`** and **`image_tools.py`** moved to `white_core/util/` — pure utilities
  with no agent deps; were misplaced in `agents/tools/`.
- **Reference data** (`indie_publications`, `white_facet_metadata`, encoding data) moved to
  `white_core/reference/` — all already importing from `white_core`; belong in core.
- Should the `staged_raw_material/` audio directories stay at repo root or move to
  `packages/extraction/data/staged_raw_material/`? Both work; moving them co-locates the data
  with the extraction package but is a large directory move. *Decide at task 3.*
