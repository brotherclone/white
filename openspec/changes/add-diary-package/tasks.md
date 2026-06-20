## 1. Diary package — model and store
- [ ] 1.1 Add `pyyaml` to `packages/diary/pyproject.toml` dependencies
- [ ] 1.2 Create `packages/diary/src/white_diary/diary_entry.py` — `DiaryEntry` Pydantic model
- [ ] 1.3 Create `packages/diary/src/white_diary/store.py` — `write_entry`, `load_entry`, `list_entries`, `delete_entry`
- [ ] 1.4 Export all public symbols from `packages/diary/src/white_diary/__init__.py`
- [ ] 1.5 `uv pip install -e packages/diary`

## 2. Tests — white_diary
- [ ] 2.1 Create `packages/diary/tests/__init__.py`
- [ ] 2.2 Create `packages/diary/tests/test_diary_store.py`:
  - write + load round-trip
  - list returns entries sorted ascending by `created_at`
  - list returns `[]` when no `diary/` dir exists
  - delete removes entry; subsequent load raises `FileNotFoundError`
  - load missing entry raises `FileNotFoundError`

## 3. API route
- [ ] 3.1 Add `white-diary` to `packages/api/pyproject.toml` dependencies
- [ ] 3.2 Create `packages/api/src/white_api/routes/diary.py` — `make_diary_router(get_shrink_wrapped_dir)`
  - resolve `song_slug → production_dir` by globbing `shrink_wrapped_dir/*/production/<song_slug>`
  - delegate all reads/writes to `white_diary` store functions
- [ ] 3.3 Register diary router in `candidate_server.py` alongside collaborators/work_orders

## 4. Tests — HTTP routes
- [ ] 4.1 Create `packages/api/tests/test_diary_routes.py`:
  - POST creates entry, returns 201 with generated id
  - GET list returns entries in `created_at` order
  - GET single returns entry; returns 404 for unknown id
  - PUT replaces entry
  - DELETE returns 204; subsequent GET returns 404
  - POST to unknown song_slug returns 404

## 5. Validation
- [ ] 5.1 `pytest packages/diary` — all store tests pass
- [ ] 5.2 `pytest packages/api/tests/test_diary_routes.py` — all route tests pass
- [ ] 5.3 `pytest` from repo root — baseline failure count unchanged
