## 0. Root Scaffold (prerequisite for all packages) ✅ DONE (commit 7107d0f)

- [x] 0.1 Investigate `sitecustomize.py` — determine if it can be removed or must stay
- [x] 0.2 Convert root `pyproject.toml` to UV workspace manifest (add `[tool.uv.workspace]`,
      move shared dev deps to workspace level, remove `[tool.setuptools]` packages list)
- [x] 0.3 Create `packages/` directory; confirm `uv sync` still resolves cleanly
- [x] 0.4 Update root `pytest.ini` to discover both `packages/*/tests/` and `tests/integration/`
- [x] 0.5 Move existing cross-package tests from `tests/` into `tests/integration/` as appropriate;
      delete remaining root test files that will migrate with their package
- [x] 0.6 Verify full test suite passes from root after workspace conversion

## 1. Package: core (`app/structures/` → `white_core`) ✅ DONE (commit 7107d0f)

- [x] 1.1 Scaffold `packages/core/pyproject.toml` + `src/white_core/__init__.py`
- [x] 1.2 Move `app/structures/` → `packages/core/src/white_core/`
- [x] 1.3 Update all `from app.structures...` imports across repo to `from white_core...`
- [x] 1.4 Move `tests/structures/` → `packages/core/tests/`
- [x] 1.5 Run `uv sync` + full pytest; fix failures
- [x] 1.6 Delete `app/structures/`
- [x] 1.7 Open PR: `feat: migrate core package (white_core)`

## 2. Package: training (`training/` → `white_training`) ✅ DONE (commit ed2cf94)

- [x] 2.1 Scaffold `packages/training/pyproject.toml` (declares `white-core = {workspace=true}`)
      + `src/white_training/__init__.py`
- [x] 2.2 Move `training/*.py`, `training/models/`, `training/core/`, `training/validation/`,
      `training/visualization/`, `training/tools/` into `packages/training/src/white_training/`
      (leave `training/data/`, `training/notebooks/`, `training/docs/` in place — data assets)
- [x] 2.3 Update `from training...` imports to `from white_training...`
- [x] 2.4 Move `training/tests/` → `packages/training/tests/`
- [x] 2.5 Run `uv sync` + full pytest; fix failures
- [x] 2.6 Delete moved source files from `training/` (data/docs/notebooks stay)
- [x] 2.7 Open PR: `feat: migrate training package (white_training)`

## 3. Package: extraction (`app/extractors/` + audio utils → `white_extraction`)

- [ ] 3.1 Scaffold `packages/extraction/pyproject.toml` (deps: `white-core`) + `src/white_extraction/`
- [ ] 3.2 Move `app/extractors/` → `packages/extraction/src/white_extraction/extractors/`
- [ ] 3.3 Move audio/MIDI utilities from `app/util/` (timestamp_audio_extractor,
      midi_segment_utils, etc.) → `packages/extraction/src/white_extraction/util/`
- [ ] 3.4 Move `app/util/generate_negative_constraints.py` with extraction (training-data util)
- [ ] 3.5 Move `staged_raw_material/__init__.py` and `cleanup_evp_intermediates.py` →
      `packages/extraction/src/white_extraction/staged_raw_material/`; decide whether audio
      data dirs (`01_01/` etc.) stay at root or move to `packages/extraction/data/staged_raw_material/`
- [ ] 3.6 Remove `staged_raw_material` from root `pyproject.toml` `[tool.setuptools]` (workspace
      manifest replaces it)
- [ ] 3.7 Update all `from app.extractors...`, relevant `from app.util...`, and
      `from staged_raw_material...` imports
- [ ] 3.8 Move `tests/extractors/` and relevant `tests/util/` → `packages/extraction/tests/`
- [ ] 3.9 Run `uv sync` + full pytest; fix failures
- [ ] 3.10 Delete moved source from `app/` and `staged_raw_material/` (Python files only)
- [ ] 3.11 Open PR: `feat: migrate extraction package (white_extraction)`

## 4. Package: analysis (`training/refractor.py` + scorer → `white_analysis`)

- [ ] 4.1 Scaffold `packages/analysis/pyproject.toml` (deps: `white-core`, `white-training`,
      `white-extraction`) + `src/white_analysis/`
- [ ] 4.2 Move `training/refractor.py` → `packages/analysis/src/white_analysis/refractor.py`
- [ ] 4.3 Update all `from training.refractor...` imports to `from white_analysis...`
- [ ] 4.4 Move relevant tests → `packages/analysis/tests/`
- [ ] 4.5 Run `uv sync` + full pytest; fix failures
- [ ] 4.6 Delete moved source
- [ ] 4.7 Open PR: `feat: migrate analysis package (white_analysis)`

## 5. Package: generation (patterns + pipelines → `white_generation`)

- [ ] 5.1 Scaffold `packages/generation/pyproject.toml` (deps: `white-core`, `white-analysis`)
      + `src/white_generation/`
- [ ] 5.2 Move `app/generators/midi/patterns/` → `packages/generation/src/white_generation/patterns/`
- [ ] 5.3 Move `app/generators/midi/pipelines/` → `packages/generation/src/white_generation/pipelines/`
- [ ] 5.4 Move `app/generators/midi/chord_generator/` (incl. data/) → `packages/generation/src/white_generation/chord_generator/`
- [ ] 5.5 Move `app/generators/midi/style_reference.py` → `packages/generation/src/white_generation/`
- [ ] 5.6 Update all `from app.generators.midi...` imports to `from white_generation...`
- [ ] 5.7 Move `tests/generators/midi/` (pattern + pipeline tests, not production tests) → `packages/generation/tests/`
- [ ] 5.8 Run `uv sync` + full pytest; fix failures
- [ ] 5.9 Delete moved source from `app/`
- [ ] 5.10 Open PR: `feat: migrate generation package (white_generation)`

## 6. Package: composition (production orchestration → `white_composition`) ⚠️ MOST CAREFUL

- [ ] 6.1 Scaffold `packages/composition/pyproject.toml` (deps: `white-core`, `white-generation`)
      + `src/white_composition/`
- [ ] 6.2 Scaffold `packages/diary/pyproject.toml` + empty `src/white_diary/__init__.py` (greenfield)
- [ ] 6.3 Scaffold `packages/production/pyproject.toml` + empty `src/white_production/__init__.py` (greenfield)
- [ ] 6.4 Move `app/generators/midi/production/` → `packages/composition/src/white_composition/`
- [ ] 6.5 Move `app/util/shrinkwrap_chain_artifacts.py` → `packages/composition/src/white_composition/`
- [ ] 6.6 Update all imports; verify `pipeline_runner` CLI works end-to-end
- [ ] 6.7 Move production-related tests → `packages/composition/tests/`
- [ ] 6.8 Run `uv sync` + full pipeline smoke test + pytest; fix failures
- [ ] 6.9 Delete moved source from `app/`
- [ ] 6.10 Open PR: `feat: migrate composition package (white_composition) + diary/production scaffolds`

## 7. Package: ideation (agents + reference → `white_ideation`)

- [ ] 7.1 Scaffold `packages/ideation/pyproject.toml` (deps: `white-core`, `white-composition`,
      `white-analysis`) + `src/white_ideation/`
- [ ] 7.2 Move `app/agents/` → `packages/ideation/src/white_ideation/agents/`
- [ ] 7.3 Move `app/reference/` → `packages/ideation/src/white_ideation/reference/`
- [ ] 7.4 Update all `from app.agents...` and `from app.reference...` imports
- [ ] 7.5 Move `tests/agents/` → `packages/ideation/tests/`
- [ ] 7.6 Update `run_white_agent.py` entry point at root to use `from white_ideation...`
- [ ] 7.7 Run `uv sync` + full pytest; fix failures
- [ ] 7.8 Delete moved source from `app/`
- [ ] 7.9 Open PR: `feat: migrate ideation package (white_ideation)`

## 8. Package: api (candidate server → `white_api`)

- [ ] 8.1 Scaffold `packages/api/pyproject.toml` (deps: `white-core`, `white-composition`,
      `white-generation`) + `src/white_api/`
- [ ] 8.2 Move `app/tools/candidate_server.py`, `candidate_browser.py`, `song_dashboard.py`
      → `packages/api/src/white_api/`
- [ ] 8.3 Update all imports
- [ ] 8.4 Move `tests/tools/` → `packages/api/tests/`
- [ ] 8.5 Run `uv sync` + full pytest; fix failures
- [ ] 8.6 Delete moved source from `app/tools/`; remove now-empty `app/` directory
- [ ] 8.7 Confirm `rg "from app\." --include="*.py"` returns zero results (migration complete)
- [ ] 8.8 Open PR: `feat: migrate api package (white_api) — app/ fully removed`

## 9. Package: client (`web/` → `packages/client/`)

- [ ] 9.1 Move `web/` → `packages/client/`
- [ ] 9.2 Update any root-level scripts or CI references that point to `web/`
- [ ] 9.3 Confirm Next.js build (`cd packages/client && npm run build`) passes
- [ ] 9.4 Open PR: `feat: migrate client package (packages/client)`

## 10. Cleanup and docs

- [ ] 10.1 Remove `white.egg-info/` (old editable install artifact)
- [ ] 10.2 Update `CLAUDE.md` import block to show new `white_*` namespaces
- [ ] 10.3 Update `openspec/project.md` with new tech stack and package structure
- [ ] 10.4 Update `ruff` `known-first-party` list to `white_core`, `white_generation`, etc.
- [ ] 10.5 Run `openspec validate refactor-monorepo-package-structure --strict`; archive change
- [ ] 10.6 Open PR: `chore: post-migration cleanup and docs`
