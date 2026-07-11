# Change: Integrate LP-side sequencing into navigation, song status, and a Claude-facing analysis tool

## Why
`add-lp-side-sequencing` introduces `sides.yml` and the assignment API/UI, but three
things are still missing for it to be usable day-to-day: a discoverable path to the new
page from the existing client, a way to see at a glance which songs are being considered
for LP placement (distinct from mix/lifecycle stage), and a way for Claude to reason
about the current sequencing against the album's aesthetic goals (chromatic color
balance, mood/energy flow across sides) rather than just raw duration math.

## What Changes
- Add a "Sides" nav entry to the client alongside the existing board/candidates/songs/
  collaborators links.
- Add an `lp_consideration` field to `manifest_bootstrap.yml` (alongside the existing
  `lifecycle_status` from `add-song-lifecycle-statuses`), with values `not_considered`
  (default), `candidate`, `placed`. Surfaced as a filter pill and badge in the song list,
  following the same pattern as the `merged`/`abandoned`/`scrapped` pills.
- New `POST /songs/{id}/lp-consideration` endpoint to set the flag directly; the side
  assignment endpoints from `add-lp-side-sequencing` also auto-set it to `placed` on
  assignment and `candidate` when a song has a mix but isn't yet on any side.
- New CLI script `lp_sequence_advisor.py` (`packages/composition`) that Claude runs via
  Bash (same pattern as `song_dashboard.py` / `plan_drift_report.py`): reads `sides.yml`
  plus each placed song's chromatic color/mood/BPM metadata and cached duration, and
  prints/writes a report on aesthetic flow per side (color clustering, energy arc,
  duration balance) with suggestions. It is read-only — it never modifies `sides.yml`.

## Impact
- Affected specs: `candidate-browser-web` (ADDED: nav entry, `lp_consideration`
  status/filter/endpoint), `lp-sequencing-analysis` (new capability: CLI advisor tool)
- Affected code: `packages/client/app/*` (nav layout, song list badge/filter),
  `packages/api/src/white_api/candidate_server.py` (status endpoint, hook into side
  assignment), new `packages/composition/src/white_composition/lp_sequence_advisor.py`
- Depends on: `add-lp-side-sequencing` (requires `sides.yml` schema and assignment
  endpoints to exist first)
