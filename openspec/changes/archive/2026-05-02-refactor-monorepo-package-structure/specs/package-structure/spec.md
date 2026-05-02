## ADDED Requirements

### Requirement: UV Workspace Root

The root `pyproject.toml` SHALL define a UV workspace with members pointing to each package
under `packages/`. Shared dev tooling (ruff, black, pytest, pre-commit, hypothesis) SHALL be
declared as workspace-level dev dependencies. No package-level code SHALL remain at the root
after migration is complete.

#### Scenario: Workspace sync installs all packages
- **WHEN** a developer runs `uv sync` at the repo root
- **THEN** all ten packages are installed in editable mode and `python -c "from white_core import ..."` succeeds without any path manipulation

#### Scenario: Root has no importable app package
- **WHEN** migration is complete
- **THEN** `import app` raises `ModuleNotFoundError` (the old flat package is gone)

---

### Requirement: Per-Package Source Layout

Each package SHALL live under `packages/<name>/` with a src layout:

```
packages/<name>/
├── pyproject.toml
└── src/
    └── white_<name>/
        └── __init__.py
```

The importable name for each package SHALL be `white_<name>` (underscore-prefixed flat
namespace, no Python namespace package required).

#### Scenario: Generation package imports succeed
- **WHEN** `white-generation` is installed
- **THEN** `from white_generation.pipelines.chord_pipeline import run_chord_pipeline` resolves correctly

#### Scenario: Core types available to all packages
- **WHEN** any package declares `white-core = {workspace = true}` as a dependency
- **THEN** `from white_core.structures.enums import LyricRepeatType` resolves correctly

---

### Requirement: Per-Package Tests

Each package SHALL co-locate its unit and package-level integration tests under
`packages/<name>/tests/`. The root `tests/` directory SHALL contain ONLY cross-package
integration tests under `tests/integration/`. The root `pytest.ini` SHALL discover both
locations.

#### Scenario: Package-scoped test run
- **WHEN** a developer runs `uv run --package white-generation pytest` from the repo root
- **THEN** only tests under `packages/generation/tests/` are executed

#### Scenario: Full suite from root
- **WHEN** a developer runs `uv run pytest` from the repo root
- **THEN** tests from all packages and `tests/integration/` are discovered and run

---

### Requirement: Package Dependency Graph

Inter-package dependencies SHALL be declared explicitly via `{workspace = true}` entries and
MUST respect the migration order (core → training → extraction → analysis → generation →
composition → ideation → diary → api → client). No package shall import from a package that
depends on it (no cycles).

#### Scenario: Circular import prevented
- **WHEN** `white_composition` imports from `white_generation`
- **THEN** `white_generation` has no import of `white_composition` at any level (enforced by
  the declared dependency graph)

#### Scenario: Core has no internal deps
- **WHEN** inspecting `packages/core/pyproject.toml`
- **THEN** there are no `white-*` workspace dependencies listed (core has no internal deps)

---

### Requirement: Import Namespace Migration

All Python imports of the form `from app.<module>...` and `from training.<module>...` SHALL be
replaced with their `white_<name>` equivalents. No compatibility shims or re-export stubs SHALL
be introduced.

#### Scenario: Old import path removed
- **WHEN** migration is complete
- **THEN** `rg "from app\." --include="*.py"` returns no results outside of
  `openspec/`, `scripts/`, or documentation files

#### Scenario: Entry point scripts updated
- **WHEN** `run_white_agent.py` is executed after migration
- **THEN** it imports successfully using the new `white_ideation` namespace

---

### Requirement: Composition Pipeline Continuity

The composition pipeline (chord → drum → bass → melody → promote → plan) SHALL remain
executable after every individual package migration PR. No migration step SHALL leave
`pipeline_runner.py` unable to run a full pipeline pass.

#### Scenario: Pipeline functional after each PR merge
- **WHEN** any single package migration PR is merged to develop
- **THEN** `uv run python -m white_composition.pipeline_runner --help` exits 0 and the
  integration tests in `tests/integration/` pass

---

### Requirement: Greenfield Package Scaffolds

`packages/diary/` and `packages/production/` SHALL be scaffolded with their `pyproject.toml`
and empty `src/white_diary/` / `src/white_production/` directories during the diary and
composition migration steps respectively. No implementation is required at scaffold time.

#### Scenario: Diary package installable but empty
- **WHEN** `packages/diary/pyproject.toml` is created and `uv sync` is run
- **THEN** `import white_diary` succeeds and `dir(white_diary)` returns an empty or near-empty
  module (scaffold only)
