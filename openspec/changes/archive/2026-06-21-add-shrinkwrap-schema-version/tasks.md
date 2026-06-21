## 1. Core constant and version alignment

- [x] 1.1 Add `SHRINKWRAP_SCHEMA_VERSION = "2.0.0"` constant to
  `shrinkwrap_chain_artifacts.py`
- [x] 1.2 Import and use the constant in `write_manifest()` so every new
  `manifest.yml` includes `schema_version: "2.0.0"` as the first field
- [x] 1.3 Import and use the constant in `write_index()` so every new
  `index.yml` includes `schema_version: "2.0.0"` as the first field
- [x] 1.4 Import and use the constant in `scaffold_song_productions()` so every
  new `manifest_bootstrap.yml` includes `schema_version: "2.0.0"` as the first field
- [x] 1.5 Update `init_production.py` to use the imported constant instead of
  the hardcoded `"1"` string (import from `shrinkwrap_chain_artifacts` or move
  constant to `white_core`)

## 2. Migration function and CLI flag

- [x] 2.1 Add `migrate_manifests(output_dir: Path, dry_run: bool = False)`
  function to `shrinkwrap_chain_artifacts.py` with two passes:

  **Pass 1 — schema_version backfill** (Class A and B both need this):
  - Walk `output_dir/*/manifest.yml` — if `schema_version` absent, insert as
    first field and rewrite
  - Walk `output_dir/*/production/*/manifest_bootstrap.yml` — same treatment
  - Walk `output_dir/index.yml` — same treatment
  - Walk `output_dir/*/production/*/song_context.yml` — update `"1"` → `"2.0.0"`

  **Pass 2 — missing artifact synthesis** (Class B threads only):
  - For each thread dir that has `production/` but **no `manifest.yml`** at its
    root: call `_synthesize_thread_manifest_stub()` to write a stub
  - For each production sub-dir that has **no `manifest_bootstrap.yml`**: first
    try `scaffold_song_productions()` if thread has a `yml/` dir; if no `yml/`
    dir exists, call `_synthesize_bootstrap_stub()` using the slug

- [x] 2.2 Add `_synthesize_thread_manifest_stub(thread_dir: Path, thread_id: str | None) -> dict`
  — derives title by un-slugifying the dir name (replace hyphens with spaces,
  title-case), writes `manifest.yml` with `schema_version: "2.0.0"`, `stub: true`,
  derived `title`, and `null` for `bpm`, `key`, `concept`, `mood`, `genres`,
  `agent_name`, `timestamp`, `thread_id`

- [x] 2.3 Add `_synthesize_bootstrap_stub(prod_dir: Path) -> dict`
  — parses `{color}__{title_slug}_v{n}` double-underscore convention from the
  production dir name to extract `rainbow_color` (the prefix before `__`) and
  `title` (un-slugified remainder before `_v{n}`); falls back to the whole slug
  as title if no double-underscore found; writes `manifest_bootstrap.yml` with
  `schema_version: "2.0.0"`, `stub: true`, derived `title` and `rainbow_color`,
  and `null` for `bpm`, `key`, `singer`

- [x] 2.4 Add `--migrate` flag to the `shrinkwrap_chain_artifacts.py` CLI that
  calls `migrate_manifests()` then exits
- [x] 2.5 Add `--dry-run` support to `migrate_manifests()` (print all planned
  changes, write nothing)

## 3. Remove production_decisions

- [x] 3.1 Delete `packages/composition/src/white_composition/production_decisions.py`
- [x] 3.2 Delete `packages/composition/tests/test_production_decisions.py`
- [x] 3.3 In `pipeline_runner.py`: remove `"decisions"` from `PHASE_ORDER`;
  remove `"decisions": None` from `PHASE_REVIEW_FILES`; remove the
  `if phase == "decisions": ...` block from `_build_command()`; remove the
  `decisions_exists` status print from `print_status()`
- [x] 3.4 In `candidate_server.py` `scan_songs()`: remove
  `"has_decisions": (prod_dir / "production_decisions.yml").exists()` from
  the returned dict
- [x] 3.5 In `song_dashboard.py`: remove `decisions_present` from any scan or
  display logic
- [x] 3.6 In `packages/client/lib/types.ts`: remove `has_decisions: boolean`
  from `SongEntry`
- [x] 3.7 In `packages/client/app/page.tsx`:
  - Remove the `has_decisions` green checkmark SVG block (lines ~240–245)
  - Remove the entire `{song.has_decisions && ...}` "Handoff to Logic" button
    block (lines ~264–282)
  - Remove `startHandoff`, `getHandoffStatus` imports from `@/lib/api` if
    no longer used elsewhere on this page
  - Remove `handoffingId` state and `handleHandoff` function
  - Remove `handoffPollRef` ref and its cleanup `useEffect`
- [x] 3.8 Remove `has_decisions`-related tests from
  `packages/api/tests/test_candidate_server.py`

## 4. candidate_server.py — schema_version, stub, invalid stage, regress endpoint

- [x] 4.1 Update `scan_songs()` to include `schema_version` (default `"1.x"` when
  absent) and `stub` (default `False`) in the returned song dict
- [x] 4.2 Update `_compute_stage()` to return `"invalid"` when
  `manifest_bootstrap.yml` is unreadable, is missing both `title` and
  `rainbow_color`, or carries an unrecognised `schema_version` prefix (anything
  that is not absent, `"1"`, `"1.x"`, or `"2.*"`)
- [x] 4.3 Add `POST /composition/regress` endpoint:
  - Body: `{ target_stage: str, confirmed: bool, diary_entry: str | null }`
  - Calls `regression_info(current_stage, target_stage)` from `logic_handoff.py`
  - If `confirmed: false`: return `{ destructive, files_to_delete }` (dry-run)
  - If `confirmed: true`: delete listed files, call `write_stage()`, write diary
    entry via `white_diary` if `diary_entry` is non-empty, return `{ ok, stage }`

## 5. logic_handoff.py — regression support

- [x] 5.1 Add `REGRESSION_FILE_MAP: dict[str, list[str]]` constant mapping each
  stage value to glob patterns deleted when moving backward past that stage:
  ```
  "lyrics":             ["lyrics*.txt", "*.lrc"]
  "vocal_placeholders": ["MIDI/melody/vocal_placeholder*.mid", "MIDI/melody/assembled*.mid"]
  "recording":          ["Recordings/*"]
  "augmentation":       ["Augmented/*"]
  "cleaning":           ["Cleaned/*"]
  ```
  Stages not in the map (`rough_mix`, `mix_candidate`, `final_mix`) produce no
  file deletions
- [x] 5.2 Add `regression_info(logic_song_dir: Path, current: str, target: str) -> dict`
  that validates both stages, raises `ValueError` on forward movement or invalid
  names, collects patterns for all stages passed through, resolves existing files,
  and returns `{ "destructive": bool, "files_to_delete": list[str] }`

## 6. UI — schema_version badge and Invalid stage

- [x] 6.1 Add `schema_version: string` and `stub: boolean` fields to `SongEntry`
  in `lib/types.ts`
- [x] 6.2 Add `"invalid"` to `SongEntry["stage"]` union in `lib/types.ts`
- [x] 6.3 In `page.tsx`, add `"invalid"` to `STAGE_LABELS`, `STAGE_BADGE_CLS`
  (use `bg-red-900/40 text-red-300 border-red-800`), and `ALL_SONG_STAGES`
- [x] 6.4 In `page.tsx`, render a `schema_version` / `stub` badge on each song
  card (show only when value differs from `"2.0.0"` or `stub: true`)
- [x] 6.5 Stub cards show "Incomplete metadata" in place of key/BPM line
- [x] 6.6 Invalid song cards show a toast on click instead of navigating

## 7. UI — phase regression on board page

- [x] 7.1 In `api.ts`, add `regressStage(targetStage, confirmed, diaryEntry)`
  calling `POST /composition/regress`
- [x] 7.2 In `lib/types.ts`, add `RegressionInfo: { destructive: boolean; files_to_delete: string[] }`
- [x] 7.3 In `board/page.tsx`, add "←" back button adjacent to the current stage
  indicator; hidden when current stage is `"structure"`
- [x] 7.4 On "←" click: call `regressStage(prevStage, false, null)` (dry-run) to
  get `RegressionInfo`, then show `RegressionModal`
- [x] 7.5 Add `RegressionModal` component in `board/page.tsx`:
  - "Move back to [Stage]?" heading
  - Scrollable file list when destructive
  - Optional diary textarea with placeholder
    "e.g. Had instrumental melodies on vocal track — need clean lyrics pass"
  - "Cancel" + "Confirm" (red when destructive) buttons
- [x] 7.6 On Confirm: call `regressStage(targetStage, true, diaryEntry)`,
  close modal, refresh composition state, show toast

## 8. Tests

- [x] 8.1 Unit test `write_manifest()` asserts `schema_version: "2.0.0"` as first key
- [x] 8.2 Unit test `scaffold_song_productions()` asserts `schema_version` in bootstrap
- [x] 8.3 Unit test `migrate_manifests()` Pass 1: legacy → versioned; idempotent; dry-run
- [x] 8.4 Unit test `migrate_manifests()` Pass 2 — Class B: stub manifest.yml written;
  bootstrap stub from slug; no-double-underscore fallback
- [x] 8.5 Unit test `_synthesize_bootstrap_stub()`:
  - `black__sequential_dissolution_v2` → `rainbow_color: Black`, `title: "Sequential Dissolution"`
  - `indigo_indigo_proposal_1770990261584` (no `__`) → title derived from full slug
- [x] 8.6 Unit test `scan_songs()`: legacy bootstrap → `schema_version: "1.x"`;
  stub bootstrap → `stub: true`; removed `has_decisions` field absent from response
- [x] 8.7 Unit test `_compute_stage()` with corrupt manifest → `"invalid"`
- [x] 8.8 Unit test `regression_info()`: forward raises; `vocal_placeholders → lyrics`
  destructive; `mix_candidate → rough_mix` non-destructive; multi-stage collects all
- [x] 8.9 Update existing shrinkwrap tests to expect `schema_version` field

## 9. Ops — run migration on live data

- [ ] 9.1 Run `python -m white_composition.shrinkwrap_chain_artifacts --migrate --dry-run` and review output
- [ ] 9.2 Run without `--dry-run` to patch existing manifests and scaffold
  missing files for `all-frequencies-*` and `the-breathing-machine-*` threads
- [ ] 9.3 Verify legacy songs appear in the candidate server song list after migration

## Dependencies and order

Tasks 1 → 2 → 4 → 6 must be completed in order (schema constant before migration before server before UI).
Task 3 (production_decisions removal) is independent and can be done in parallel with tasks 1–2.
Task 5 (logic_handoff regression) must complete before task 4.3 and task 7.
Task 8 can be written alongside the implementation tasks they cover.
Task 9 is a manual ops step after all code is merged and tested.
