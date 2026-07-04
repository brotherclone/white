# Change: Add schema_version to shrinkwrap manifests and migrate legacy threads

## Why

The move to a uv workspace restructured all packages under `packages/` and
introduced `song_context.yml` as the canonical per-song source of truth. Artifact
files written before that migration (`manifest.yml`, `manifest_bootstrap.yml`,
`index.yml`) carry no version field, making it impossible to tell which schema
generation they belong to, and causing older threads to be invisible to the
client because they were never scaffolded with `manifest_bootstrap.yml`.

The two oldest threads (`all-frequencies-return-to-source-*` and
`the-breathing-machine-learns-to-sing`) — which include the first song, "The
Archivist's Rebellion" — have production directories but no
`manifest_bootstrap.yml` files, so `scan_songs()` silently skips them.

## Two Classes of Legacy Thread

Inspection of `shrink_wrapped/` reveals two distinct legacy conditions that need
separate recovery strategies:

**Class A — Partial (has `manifest.yml` but bootstrap files pre-date `schema_version`):**
e.g. `violet-fallback-*`, `white-everyone-you-love-is-here`. These have a
thread-level `manifest.yml` and `manifest_bootstrap.yml` files in production dirs;
they just lack the `schema_version` field. Fix = add field in place.

**Class B — Pre-scaffold (no `manifest.yml`, no `manifest_bootstrap.yml`, no `yml/` dir):**
e.g. `the-breathing-machine-learns-to-sing`, `all-frequencies-return-to-source-*`.
These threads were shrinkwrapped before the manifest/bootstrap system existed. They
only contain `midi/`, `wav/`, `production/<slug>/{bass,chords,drums,melody}/`. There
is no source YML to scaffold from, and the thread-level `manifest.yml` does not exist.

For Class B threads the migration must synthesize both artifacts from available
signals:

- **Thread `manifest.yml` stub**: title derived by humanizing the thread dir slug;
  `bpm`, `key`, `concept`, `mood`, `genres` all `null`; `thread_id` derived from
  `index.yml` if present, else `null`.
- **Production `manifest_bootstrap.yml` stub**: old production slugs follow the
  convention `{color}__{title_slug}_v{n}` (double underscore separates color from
  title). The migration parses the color prefix and humanizes the remainder as the
  title; `bpm`, `key`, `singer` are `null`.

Both stub files are marked `schema_version: "2.0.0"` and `stub: true` so they are
distinguishable from fully-scaffolded manifests.

## What Changes

- **BREAKING** — `manifest.yml`, `manifest_bootstrap.yml`, and `index.yml` gain
  a `schema_version: "2.0.0"` field. Files without this field are treated as
  schema 1.x (pre-uv-workspace) by readers.
- `song_context.yml` schema_version bumps from `"1"` to `"2.0.0"` to align with
  the rest of the artifact family.
- `shrinkwrap_chain_artifacts.py` gains a `--migrate` CLI flag with two passes:
  1. **Pass 1**: backfill `schema_version: "2.0.0"` on all existing manifest files
     that are missing it.
  2. **Pass 2**: for Class B threads missing a `manifest.yml`, write a stub thread
     manifest; for production dirs missing `manifest_bootstrap.yml`, first try to
     scaffold from source YML (if `yml/` exists), then fall back to a stub derived
     from the production slug's color-prefix convention.
- `scan_songs()` in `candidate_server.py` surfaces the `schema_version` and `stub`
  fields on each returned song entry, tolerating their absence (legacy = `"1.x"`).
- `SHRINKWRAP_SCHEMA_VERSION = "2.0.0"` constant is introduced in
  `shrinkwrap_chain_artifacts.py` and imported by `init_production.py`.

## UI Changes Bundled in This Change

Three UI improvements that directly depend on or enable the migration work:

**1. Schema version badge on song cards**
The song index (`/`) shows each song's `schema_version` as a small badge so you can
instantly spot legacy (1.x) or stub cards that need attention after migration.
Stub cards also get a distinct visual treatment.

**2. "Invalid" stage**
A new `"invalid"` stage value is added across the stack. `_compute_stage()` returns
`"invalid"` when `manifest_bootstrap.yml` cannot be parsed, is missing required
fields, or carries an unrecognised `schema_version`. The stage filter on the song
index gains an "Invalid" option so you can isolate broken songs.
Stage order for the filter: `All → Ideation → Generation → Composition → Production → Mixing → Complete → Invalid`

**3. Phase regression with diary confirmation modal**
On the `/board` page, every stage in the MixStage strip gains a "← " back button so
stuck songs can be un-stuck. Backward movement always requires a confirmation modal.
When the regression is destructive (files were written at/after the target stage),
the modal lists the specific files that will be deleted. The modal also offers an
optional diary-entry text area so the reason for the regression can be recorded
inline. On confirmation: files are deleted, stage is set, diary entry written if
text was provided.

Destructive regression file map (files deleted when moving *backward past* a stage):
| Stage passed back through | Files deleted from Logic song dir |
|---|---|
| `lyrics` | `lyrics*.txt`, `*.lrc` |
| `vocal_placeholders` | `MIDI/melody/vocal_placeholder*.mid`, `MIDI/melody/assembled*.mid` |
| `recording` | `Recordings/` directory contents |
| `augmentation` | `Augmented/` directory contents |
| `cleaning` | `Cleaned/` directory contents |

Non-destructive regressions (rough_mix → mix_candidate → final_mix) show a simple
"Are you sure?" confirmation with no file list, but still offer the diary textarea.

New backend: `POST /composition/regress` with body
`{ target_stage, confirmed, diary_entry }`. When `confirmed: false` (dry-run) it
returns `{ destructive, files_to_delete }` without changing anything. When
`confirmed: true` it executes the regression and returns `{ ok: true, stage }`.

## Impact

- Affected specs: `chain-artifacts`, `candidate-browser-web`, `logic-handoff`
- Affected code:
  - `packages/composition/src/white_composition/shrinkwrap_chain_artifacts.py` —
    `write_manifest()`, `write_index()`, `scaffold_song_productions()`, new
    `migrate_manifests()` and `_synthesize_thread_manifest_stub()` and
    `_synthesize_bootstrap_stub()` functions, `--migrate` CLI flag
  - `packages/composition/src/white_composition/init_production.py` —
    `write_initial_proposal()` schema_version constant
  - `packages/composition/src/white_composition/logic_handoff.py` —
    new `regression_info(current, target)` function and `REGRESSION_FILE_MAP`
  - `packages/api/src/white_api/candidate_server.py` — `scan_songs()` return dict,
    new `POST /composition/regress` endpoint, `_compute_stage()` invalid case
  - `packages/client/lib/types.ts` — `SongEntry.stage` union, `schema_version`/`stub` fields
  - `packages/client/app/page.tsx` — schema_version badge, stub indicator, invalid stage filter
  - `packages/client/app/board/page.tsx` — regression back button, confirmation+diary modal

## Version Semantics

| Version | Era | Trigger |
|---|---|---|
| _(absent)_ | Pre-uv-workspace (1.x legacy) | Files written before monorepo restructure |
| `"1"` | `song_context.yml` only (transitional) | init_production schema_version before this change |
| `"2.0.0"` | uv workspace + full artifact family | This change |

All readers treat any string starting with `"1"` or an absent field as schema 1.x.
