## 0. Shrinkwrap — scaffold production directories

- [x] 0.1 Add `scaffold_song_productions(thread_dest_dir: Path, yml_dir: Path) -> list[str]` to `shrinkwrap_chain_artifacts.py`: iterates YAML files in `yml_dir`, skips known non-proposal files (`evp.yml`, `all_song_proposals.yml`, files missing `bpm`/`key`/`rainbow_color`), creates `thread_dest_dir/production/<slug>/` for each, writes a minimal `manifest_bootstrap.yml` containing `title`, `key`, `bpm`, `rainbow_color`, `singer` (null if absent). Returns list of slugs created. Skips if `manifest_bootstrap.yml` already exists (idempotent).
- [x] 0.2 Call `scaffold_song_productions(dest_dir, dest_dir / "yml")` inside `shrinkwrap_thread()` after `copy_thread_files()` (dry-run: print slugs that would be created instead). Add `--no-scaffold` flag to skip this step.
- [x] 0.3 Backfill existing shrink_wrapped threads: add `--scaffold-only` flag to `shrinkwrap_chain_artifacts.py` that walks already-copied thread dirs (not chain_artifacts), runs `scaffold_song_productions` on each, and exits.
- [x] 0.4 Unit tests: `tests/util/test_shrinkwrap_scaffold.py` — test proposal detection (skips evp.yml, all_song_proposals.yml, files missing bpm/key/rainbow_color), correct directory naming, idempotency (no overwrite if manifest_bootstrap.yml exists), dry-run output.

## 1. Backend — new launch mode and song-list scanning
- [ ] 1.1 Add `--shrink-wrapped-dir` argument to `candidate_server.py`; make `--production-dir` optional (required only when `--shrink-wrapped-dir` is absent)
- [ ] 1.2 Add `_shrink_wrapped_dir: Path | None` module-level variable alongside `_production_dir`
- [ ] 1.3 Implement `scan_songs(shrink_wrapped_dir)` helper: walks `*/production/*/manifest_bootstrap.yml`, reads title/key/bpm/color/singer, returns list of `SongEntry` dicts with `id` (`{thread_slug}__{production_slug}`), `thread_slug`, `production_slug`, `production_path`, and manifest fields
- [ ] 1.4 Add `GET /songs` endpoint — calls `scan_songs` and returns the list; returns 503 if server was not launched in album mode
- [ ] 1.5 Add `POST /songs/activate` endpoint (body: `{"id": "<thread_slug>__<production_slug>"}`) — resolves and validates the production path, sets `_production_dir`; returns `{"ok": true, "production_dir": "..."}`
- [ ] 1.6 Add `GET /songs/active` endpoint — returns the active song entry or `{"active": null}` if none selected
- [ ] 1.7 Guard all existing candidate endpoints: if `_production_dir` is None, return 503 with message `"No song selected — POST /songs/activate first"`
- [ ] 1.8 Update startup print to show `Serving album from: <shrink_wrapped_dir>` when in album mode

## 2. Frontend — song index page
- [ ] 2.1 Create `web/app/page.tsx` as the song index — fetches `GET /songs` on mount; shows a loading state and a "No songs found" empty state
- [ ] 2.2 Render one card per song showing: title, thread slug (dimmed), key, BPM, color badge, singer (if present)
- [ ] 2.3 On card click: call `POST /songs/activate` then `router.push("/candidates")`; show a spinner on the clicked card while activating
- [ ] 2.4 If `GET /songs` returns 503 (server in single-song mode), render a redirect notice and navigate immediately to `/candidates`
- [ ] 2.5 Style cards: dark background matching existing zinc palette, color badge uses the song's `rainbow_color` value

## 3. Frontend — candidate browser breadcrumb
- [ ] 3.1 Move existing `web/app/page.tsx` to `web/app/candidates/page.tsx`
- [ ] 3.2 Add `GET /songs/active` call on mount in the candidate page to retrieve the active song title
- [ ] 3.3 Render breadcrumb above the "Candidate Browser" heading: `← Songs  /  <song title>` — "← Songs" is a `<Link href="/">` component; hidden when active song title is unavailable (single-song mode)
- [ ] 3.4 Update `web/lib/api.ts` with `fetchSongs()` and `activateSong(id)` helpers
- [ ] 3.5 Update `web/app/layout.tsx` if needed (no new dependencies expected)

## 4. Validation
- [ ] 4.1 Manual smoke test: `python -m app.tools.candidate_server --shrink-wrapped-dir shrink_wrapped/` → browser opens `/`, shows song cards; click a card → `/candidates` shows that song's candidates with breadcrumb
- [ ] 4.2 Legacy smoke test: `python -m app.tools.candidate_server --production-dir <path>` → browser opens `/candidates` directly, no breadcrumb shown
- [ ] 4.3 Verify TypeScript build passes: `cd web && npm run build`
