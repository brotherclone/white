## 1. Backend
- [ ] 1.1 In `scan_songs` (`candidate_server.py`), read `song_context.yml` for the production dir and add `concept: str | None` to the returned dict (empty string → None; missing file → None)
- [ ] 1.2 Update `packages/api/tests/test_candidate_server.py` song-shape test to include `concept` field

## 2. Frontend types
- [ ] 2.1 Add `concept: string | null` to `SongEntry` interface in `packages/client/lib/types.ts`

## 3. Frontend UI
- [ ] 3.1 In `app/candidates/page.tsx`, add a concept block below the breadcrumb and above the toolbar: 3-line clamp, "Show more / Show less" toggle, only rendered when `activeSong?.concept` is non-null
