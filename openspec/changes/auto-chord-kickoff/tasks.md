# Tasks: auto-chord-kickoff

Ordered delivery. Each task is independently verifiable.

---

## Task 1 — Add `_invoke_chord_pipeline_safe` to `WhiteAgent`

Add a private method that wraps `run_chord_pipeline()` for safe in-process use:
- Imports `run_chord_pipeline` from `app.generators.midi.pipelines.chord_pipeline`
- Accepts `thread_dir: str`, `song_filename: str`, and passes through defaults
  (`seed=42`, `num_candidates=200`, `top_k=10`)
- Catches `SystemExit` and all `Exception` subclasses, logs as WARNING, never re-raises

**Validation:** unit test with a mocked `run_chord_pipeline` that raises `SystemExit(1)` —
assert no exception escapes and WARNING was logged.

---

## Task 2 — Thread and filename resolution in `finalize_song_proposal`

After `self.save_all_proposals(state)` succeeds and `state.run_finished = True`:
- Resolve `thread_dir = Path(self._artifact_base_path()) / state.thread_id`
- Build `song_filename = f"song_proposal_{final.rainbow_color}_{final.iteration_id}.yml"`
  where `final = state.song_proposals.iterations[-1]`
- Guard: skip (log WARNING) if iterations is empty or MOCK_MODE is active
- Call `self._invoke_chord_pipeline_safe(str(thread_dir), song_filename)` when
  `self._auto_chord_generation` is True

**Validation:** unit test that patches `_invoke_chord_pipeline_safe` and asserts it
is called with the expected args when `auto_chord_generation=True`.

---

## Task 3 — Wire `auto_chord_generation` flag through `start_workflow`

- Add `auto_chord_generation: bool = False` parameter to `start_workflow()`
- Fall back to `os.getenv("AUTO_CHORD_GENERATION", "false").lower() == "true"`
  when the parameter is not explicitly `True`
- Store resolved value as `self._auto_chord_generation` before `workflow.invoke()`
  (so `finalize_song_proposal` can read it via `self`)

**Validation:** unit test that `AUTO_CHORD_GENERATION=true` env var triggers the
pipeline and that `auto_chord_generation=False` (explicit) suppresses it even with
the env var set.

**Note:** `self._auto_chord_generation` is a plain Python attribute set at runtime,
not a Pydantic field — `WhiteAgent` inherits from `BaseModel` so this needs
`model_config = ConfigDict(arbitrary_types_allowed=True)` or use `__dict__` assignment
if `WhiteAgent` already allows it. Confirm before implementing.

---

## Task 4 — Tests

New file: `tests/agents/test_white_agent_chord_kickoff.py`

Tests to include:
1. `_invoke_chord_pipeline_safe` swallows `SystemExit` → WARNING logged, no raise
2. `_invoke_chord_pipeline_safe` swallows generic `Exception` → WARNING logged
3. `_invoke_chord_pipeline_safe` success path → INFO logged with output path
4. `finalize_song_proposal` with flag=True calls safe wrapper with correct paths
5. `finalize_song_proposal` with flag=False does NOT call safe wrapper
6. `finalize_song_proposal` with `MOCK_MODE=true` does NOT call safe wrapper
7. `finalize_song_proposal` with empty iterations does NOT call safe wrapper
8. `start_workflow` with `AUTO_CHORD_GENERATION=true` env var triggers pipeline
9. `run_finished` remains True when chord gen raises

All tests use mocked `run_chord_pipeline`; no actual MIDI generation required.

---

## Dependencies

- No changes to `chord_pipeline.py`
- No changes to other agents
- Tasks 1 and 3 can be implemented in parallel; Task 2 depends on both
- Task 4 can be written alongside Tasks 1–3
