## Context
White is the only double-album (4-side) release in the catalog. Songs are scanned by
`scan_songs()` in `candidate_server.py` (`packages/api/src/white_api/candidate_server.py:262`)
from `manifest_bootstrap.yml` files, each keyed by `id = f"{thread_slug}__{production_slug}"`.
Mix files are tracked per-song in `song_context.yml` (`mix_file` field) but no duration is
stored anywhere today. There is no cross-song "album assembly" concept in the data model —
every existing file (`review.yml`, `production_plan.yml`, `manifest_bootstrap.yml`) is
scoped to a single song.

## Goals / Non-Goals
- Goals: track 4 sides (A–D), each an ordered list of songs with a running duration total
  against a ~20-minute soft limit; drag-and-drop assignment/reordering in the client;
  read mix duration without adding a new heavy dependency.
- Non-Goals: enforcing the limit as a hard constraint (it's a soft creative guide, not a
  physical pressing — over-limit sides are allowed, just flagged); no audio transcoding
  or waveform analysis; no automatic bin-packing/auto-sequencing (that's covered by the
  separate `lp-sequencing-analysis` advisor tool in `add-lp-sequencing-integration`).

## Decisions

- **Decision: `sides.yml` lives at the album root (`$SHRINKWRAP_OUTPUT_DIR/sides.yml`)**,
  matching the existing `index.yml` / `negative_constraints.yml` convention — one
  album-scoped file, not one per song.
  - Shape:
    ```yaml
    side_limit_seconds: 1200
    sides:
      A: {songs: [{song_id: "violet__foo", duration_seconds: 187.4}]}
      B: {songs: []}
      C: {songs: []}
      D: {songs: []}
    ```
  - Alternatives considered: storing side assignment inside each song's
    `manifest_bootstrap.yml` — rejected because ordering within a side is a cross-song
    concern (position matters) that a per-song file can't represent cleanly.

- **Decision: duration is computed on read via `soundfile.info(path).frames / samplerate`**,
  not stored redundantly in `song_context.yml`. `sides.yml` caches the duration at the
  moment a song is assigned to a side (so the UI doesn't need to re-probe every mix file
  on every load), and the cached value is refreshed whenever the song is re-assigned or
  an explicit "refresh durations" action is triggered.
  - Alternatives considered: writing `mix_duration_seconds` into `song_context.yml` at mix-set
    time — rejected as unnecessary duplication; the existing `/production/mix/info`
    endpoint is the natural place to expose duration on demand.

- **Decision: use `@dnd-kit/core` for the client drag-and-drop board.** It's actively
  maintained, has no legacy-context issues with React 19 (used by `packages/client`
  today), and is a common, boring choice — no drag-and-drop library exists in the client
  yet, so this is a new dependency either way.
  - Alternatives considered: `react-beautiful-dnd` (unmaintained, React 18 issues),
    hand-rolled HTML5 drag events (more code, worse accessibility) — both rejected.

- **Decision: only songs with `has_mix: true` are assignable.** Songs without a mix
  appear in an "available" list but are visually disabled/non-draggable, since duration
  cannot be computed without an audio file.

## Risks / Trade-offs
- Soft-limit-only enforcement means sides can silently grow past 20 minutes — mitigated
  by a clear visual warning (not a block) in the UI, matching the user's stated intent
  that this is a creative constraint, not a hard rule.
- Duration caching in `sides.yml` can go stale if a mix file is replaced without
  re-assigning — mitigated by a manual "refresh durations" action rather than trying to
  detect file changes automatically (keeps the implementation simple per project
  conventions).

## Migration Plan
Net-new feature; no existing data migrates. `sides.yml` is created on first write (empty
A–D sides) if it doesn't exist.

## Open Questions
- None outstanding — the two ambiguous points (tool shape for the advisor, proposal
  split) were resolved before scaffolding and are captured in `add-lp-sequencing-integration`.
