# Change: Consolidate to a single Python 3.12 virtual environment

## Why

The project maintains two virtual environments (`.venv` and `.venv312`) solely because
`transformers` is unpinned in `pyproject.toml` and resolves to 5.x, which breaks
`torch.load()` compatibility needed by the Refractor scorer. `.venv312` was created as a
workaround with `transformers 4.x` pinned manually. Every pipeline command in
`SONG_GENERATION_PROCESS.md` says "use `.venv312/bin/python`", there is a pytest plugin
that re-runs `@pytest.mark.venv312` tests under a subprocess, and error messages
throughout the codebase direct users to the alternate venv. The root cause is a one-line
fix in `pyproject.toml`. Once `transformers>=4,<5` is pinned, a single `.venv` serves all
purposes and all the machinery around `.venv312` can be deleted.

## What Changes

- Pin `transformers>=4.47,<5` in `pyproject.toml` (4.47+ for numpy 2.x compatibility)
- Tighten `requires-python` from `>=3.11` to `>=3.12` (project is committed to 3.12;
  `.python-version` already pins it)
- Delete `.venv312` and rebuild `.venv` via `uv sync`
- Remove the `venv312` pytest marker, the conftest re-runner plugin, and all
  `@pytest.mark.venv312` decorators
- Replace all `.venv312/bin/python` invocations in `SONG_GENERATION_PROCESS.md` with
  `.venv/bin/python`
- Clean up `venv312` references in error messages (`refractor.py`,
  `artist_catalog.py`, `song_evaluator.py`) and in openspec specs
- Remove the `.venv` / transformers 5.x entry from the known bugs table in
  `SONG_GENERATION_PROCESS.md`

## Impact

- Affected specs: `codebase-hygiene`
- Affected code:
  - `pyproject.toml`
  - `tests/conftest.py`
  - `pytest.ini`
  - `tests/generators/midi/test_chord_pipeline.py` (remove `@pytest.mark.venv312`)
  - `training/refractor.py` (remove venv312 runtime warning)
  - `app/generators/artist_catalog.py` (update error messages)
  - `app/generators/midi/production/song_evaluator.py` (update error messages)
  - `app/generators/SONG_GENERATION_PROCESS.md` (pervasive `.venv312` → `.venv`)
  - `openspec/specs/artist-style-catalog/spec.md`
  - `openspec/specs/lyric-feedback/spec.md`
