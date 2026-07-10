## 1. Backend — manifest schema + stage computation
- [x] 1.1 `_compute_stage()` in `candidate_server.py`: insert lifecycle guard at top —
      read `lifecycle_status` from `manifest_bootstrap.yml`; if it is
      `"merged"`, `"abandoned"`, or `"scrapped"`, return it directly
- [x] 1.2 `scan_songs()`: surface `lifecycle_status`, `merged_with`, and
      `uses_parts_from` in each song entry dict
- [x] 1.3 Add `lifecycle_status: str | None`, `merged_with: list[str]`, and
      `uses_parts_from: list[str]` to the API response documentation / inline comments

## 2. Backend — new API endpoints
- [x] 2.1 `POST /songs/{id}/lifecycle` — resolve song path, patch `manifest_bootstrap.yml`
      with `lifecycle_status`; when status is `"merged"` and `merged_with` is provided,
      patch the partner's manifest too (bilateral write); return 404 if any ID not found
- [x] 2.2 `GET /songs/scrapped` — call `scan_songs()`, filter to `lifecycle_status == "scrapped"`,
      return array
- [x] 2.3 `PATCH /songs/{id}/uses-parts-from` — resolve song path, patch
      `manifest_bootstrap.yml` with `uses_parts_from` list; return `{"ok": true}`

## 3. Backend — tests
- [x] 3.1 `test_compute_stage_lifecycle_*` — merged/abandoned/scrapped bypass mix-stage logic
- [x] 3.2 `test_lifecycle_endpoint_abandon` / `test_lifecycle_endpoint_scrap` —
      manifest patched, `{"ok": true}` returned
- [x] 3.3 `test_lifecycle_endpoint_merge_bilateral` — both manifests updated
- [x] 3.4 `test_lifecycle_endpoint_merge_unknown_partner` — 404, no manifest modified
- [x] 3.5 `test_get_scrapped_songs` — returns only scrapped entries
- [x] 3.6 `test_patch_uses_parts_from` — `uses_parts_from` written to manifest

## 4. Client — types and API helpers
- [x] 4.1 Extend `SongEntry["stage"]` union in `types.ts` with
      `"merged" | "abandoned" | "scrapped"`; add `lifecycle_status`, `merged_with`,
      `uses_parts_from` optional fields to `SongEntry`
- [x] 4.2 Add `setLifecycleStatus()`, `fetchScrappedSongs()`, and `setUsesPartsFrom()`
      helpers to `api.ts`

## 5. Client — song index filter pills
- [x] 5.1 Redefine "all" count to exclude lifecycle-terminal songs
      (`stage` not in `["merged", "abandoned", "scrapped"]`)
- [x] 5.2 Extend `StageFilter` type and `ALL_SONG_STAGES` / pill render list to include
      `merged`, `abandoned`, `scrapped` (ordered after `stub`, before `invalid`)
- [x] 5.3 Add `STAGE_LABELS` and `STAGE_BADGE_CLS` entries for the three new stages
- [x] 5.4 Update the card filter predicate so `all` excludes terminal songs

## 6. Client — board page Song Lifecycle panel
- [x] 6.1 Add a collapsible Song Lifecycle section to `/board/page.tsx` below the
      existing stage controls
- [x] 6.2 **Abandon button + modal** — whimsical confirmation copy,
      on confirm call `setLifecycleStatus(id, "abandoned")`
- [x] 6.3 **Scrap button + modal** — confirmation noting parts can still be reused
      downstream, on confirm call `setLifecycleStatus(id, "scrapped")`
- [x] 6.4 **Merge button + modal** — fetch all active (non-terminal) songs for the picker;
      searchable dropdown; on confirm call
      `setLifecycleStatus(id, "merged", { merged_with: [chosen_id] })`
- [x] 6.5 Terminal state display — when `activeSong.lifecycle_status` is set,
      show a status banner and suppress action buttons

## 7. Client — "Uses parts from" widget on board page
- [x] 7.1 Add collapsible "Uses parts from" sub-section inside the Song Lifecycle panel
      in `/board/page.tsx`
- [x] 7.2 On expand, call `fetchScrappedSongs()`; render a multi-select list of
      scrapped song titles
- [x] 7.3 Pre-select IDs already present in `activeSong.uses_parts_from`
- [x] 7.4 "Save" button calls `setUsesPartsFrom(activeSong.id, selectedIds)`;
      show inline success/error toast
- [x] 7.5 Show "No scrapped songs available" when the list is empty

## 8. Validation
- [x] 8.1 Run `pytest packages/api/tests/` — 170 passed
- [ ] 8.2 Manual smoke: abandon a song → verify index shows it only under "abandoned" pill
- [ ] 8.3 Manual smoke: merge two songs → verify both cards show "merged" in their
      respective filtered views
- [ ] 8.4 Manual smoke: scrap a song, then open another song's board → verify
      the scrapped song appears in "Uses parts from" list
