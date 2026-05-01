## 1. Backend — Lyric Candidates Endpoints

- [ ] 1.1 Add `GET /lyrics` to `candidate_server.py`
  - Read `melody/lyrics_review.yml`; 404 if absent, 503 if no active song
  - Load each candidate's `.txt` file from `melody/candidates/`
  - Compute `fitting_verdict` (worst-case across sections)
  - Set top-level `status: "promoted"` if `melody/lyrics.txt` exists
- [ ] 1.2 Add `POST /lyrics/<id>/approve` to `candidate_server.py`
  - Set `status: approved` on named candidate; set all others to `pending`
  - Return `{"ok": true}`; 422 if id not found

## 2. Frontend — Types and API Client

- [ ] 2.1 Add `LyricCandidate` and `LyricsResponse` interfaces to `lib/types.ts`
- [ ] 2.2 Add `fetchLyrics()` and `approveLyric(id)` to `lib/api.ts`

## 3. Frontend — Lyrics Column UI

- [ ] 3.1 Fetch lyrics data in `app/board/page.tsx` alongside composition fetch
- [ ] 3.2 Render `[v1] [v2] [v3]` buttons in lyrics column when status is `"pending"`
- [ ] 3.3 Render `[See Lyrics]` button when status is `"promoted"`

## 4. Frontend — Lyric Popup Modal

- [ ] 4.1 Build `LyricModal` component (inline in `board/page.tsx` or `components/`)
  - Props: `candidate`, `readOnly`, `onClose`, `onPromote`
  - Scrollable pre-formatted lyric text block
  - Metadata row: match score + fitting verdict badge
  - Promote button (hidden when `readOnly`)
  - `[×]` close button + backdrop click to dismiss
- [ ] 4.2 Wire promote action: `approveLyric(id)` → `promotePhase("lyrics")` → close → refresh
- [ ] 4.3 Wire `[See Lyrics]` to open modal in read-only mode with promoted candidate text

## 5. Validation

- [ ] 5.1 `openspec validate lyrics-review-board --strict` passes
- [ ] 5.2 TypeScript check clean (`npx tsc --noEmit` in `packages/client`)
- [ ] 5.3 Manual smoke test: generate lyrics → open v1/v2/v3 popups → promote → See Lyrics
