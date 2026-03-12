## 1. pyproject.toml — pin transformers and tighten Python floor

- [x] 1.1 Change `"transformers"` to `"transformers>=4.47,<5"` in dependencies
- [x] 1.2 Change `requires-python = ">=3.11"` to `requires-python = ">=3.12"`

## 2. Rebuild .venv

- [x] 2.1 Run `uv sync` — let uv resolve the new constraints
- [x] 2.2 Review `uv.lock` diff — confirm transformers resolves to 4.x, no unexpected
      regressions in other packages

## 3. Verify consolidated .venv before deleting .venv312

- [x] 3.1 Run `python -m pytest tests/` under the new `.venv` — all tests must pass
- [x] 3.2 Run `python -c "from training.refractor import Refractor; r = Refractor(); print('ok')"` —
      must import without errors
- [x] 3.3 Smoke-test one live `score()` call to confirm Refractor produces valid scores
      (use any approved MIDI candidate from `chain_artifacts/`)

## 4. Remove venv312 test infrastructure

- [x] 4.1 Remove the entire `# --- venv312 re-run support ---` block from
      `tests/conftest.py` (lines 36–90: `_find_venv312_python`, `pytest_collection_modifyitems`,
      and the `pytest_configure` addinivalue_line for the venv312 marker)
- [x] 4.2 Remove `venv312: mark tests that must run under .venv312 python` from
      `pytest.ini` markers section
- [x] 4.3 Remove `@pytest.mark.venv312` decorator from
      `tests/generators/midi/test_chord_pipeline.py`

## 5. Clean up venv312 references in source

- [x] 5.1 Remove the venv312 runtime warning block from `training/refractor.py`
      (the section that checks `sys.executable` and warns if not under `.venv312`)
- [x] 5.2 Update error messages in `app/generators/artist_catalog.py` that direct
      users to `.venv312` — replace with `.venv` or remove the venv-specific hint
- [x] 5.3 Update error messages in `app/generators/midi/production/song_evaluator.py`
      that direct users to `.venv312` — same treatment

## 6. Update documentation

- [x] 6.1 Replace all `.venv312/bin/python` occurrences in
      `app/generators/SONG_GENERATION_PROCESS.md` with `.venv/bin/python`
- [x] 6.2 Remove the `.venv uses transformers 5.x` known bugs table row from
      `SONG_GENERATION_PROCESS.md`
- [x] 6.3 Update the Singer mapping note in `SONG_GENERATION_PROCESS.md` if it still
      mentions venv split context
- [x] 6.4 Update `openspec/specs/artist-style-catalog/spec.md` — replace `.venv312`
      reference with `.venv`
- [x] 6.5 Update `openspec/specs/lyric-feedback/spec.md` — replace `.venv312`
      reference with `.venv`

## 7. Delete .venv312

- [x] 7.1 `rm -rf .venv312` — only after tasks 3.1–3.3 pass
- [x] 7.2 Run `python -m pytest tests/` one final time under `.venv` to confirm nothing
      relied on the venv312 subprocess path
