## 1. Backend — retrieve_samples module
- [ ] 1.1 Create `packages/composition/src/white_composition/retrieve_samples.py` with
  `load_clap_index(parquet_path, meta_parquet_path) → DataFrame` and
  `retrieve_by_color(df, color, top_n) → list[dict]` (fields: segment_id, source_audio_file,
  match, song_slug, color)

## 2. Backend — API endpoints
- [ ] 2.1 Add `GET /samples?top_n=N` to `candidate_server.py` — calls `retrieve_by_color`
  for the active song's color; returns JSON array of `SampleEntry` objects including `audio_url`
- [ ] 2.2 Add `GET /audio/{segment_id}` — resolves the WAV path from the CLAP index and
  streams the file; 404 if not found
- [ ] 2.3 Add `POST /samples/{segment_id}/export` — copies the segment WAV to
  `$LOGIC_OUTPUT_DIR/<thread>/<title>/Samples/<segment_id>.wav`; 503 if `LOGIC_OUTPUT_DIR`
  unset; 404 if WAV absent; returns `{"ok": true, "dest": "..."}`
- [ ] 2.4 Update `packages/api/tests/test_candidate_server.py` with shape tests for the
  three new endpoints

## 3. Backend — Logic handoff samples sweep
- [ ] 3.1 In `logic_handoff.handoff()`, after the MIDI copy step, copy all WAVs from
  `<production_dir>/samples/` (if the dir exists) to `$LOGIC_OUTPUT_DIR/.../Samples/`

## 4. Frontend — types and API helpers
- [ ] 4.1 Add `SampleEntry` interface to `packages/client/lib/types.ts`
  (`segment_id`, `song_slug`, `color`, `match`, `audio_url`)
- [ ] 4.2 Add `fetchSamples(topN?)`, `exportSample(segmentId)` helpers to `packages/client/lib/api.ts`

## 5. Frontend — Samples panel
- [ ] 5.1 Add a collapsible "Samples" section in `app/candidates/page.tsx` below the
  candidate table, always rendered regardless of phase filter
- [ ] 5.2 Each row shows: rank, segment_id, song_slug, color chip, match score, inline
  `<audio>` player (src = audio_url), and an "Export" button
- [ ] 5.3 "Export" button calls `exportSample(segmentId)` and shows a per-row success/error state

## 6. Frontend — Quartet button fix
- [ ] 6.1 Fix `canRunQuartet` so it is true when `quartetStatus` is `undefined`, `null`, or `"pending"`
- [ ] 6.2 Verify the "Generate Quartet" button appears alongside "Handoff to Logic" when
  melody is promoted (no layout change needed; confirm the existing code satisfies the spec)
