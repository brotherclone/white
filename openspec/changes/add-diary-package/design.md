# Design: Song Diary

## Context
The diary is the narrative layer of the White pipeline — a per-song journal of creative
events authored by agents, automation, Claude, and Gabriel. It must be writable from
inside running Python processes (no HTTP dependency) and readable by the Next.js client
(HTTP).

## Goals / Non-Goals
- **Goals:** lite flexible model; direct Python write path callable from pipeline hooks;
  HTTP CRUD for the client; no schema migrations needed as hooks evolve
- **Non-Goals:** full-text search; cross-song aggregation; event streaming; MDX rendering;
  version history

## Decisions

**Filesystem YAML storage** — one `.yml` file per entry under `packages/diary/src/entries/`.
No database.

**`white_diary.ENTRIES_DIR`** — a `Path` constant exported from the package, derived from
`__file__` so it resolves correctly regardless of working directory. All callers (pipeline
hooks, API routes, agents) use this as the entries root. No env var or config required.

**Open `author: str`, free `phase: str | None`** — the set of authors and phases is
open-ended (future color agents, new pipeline phases). Enum would require migration when
a new agent is added. Downstream code that needs to filter by type can match on known
strings.

**`metadata: dict[str, Any]`** catch-all — hooks can attach phase-specific data without
touching the model. New keys are backward-compatible.

**Two surfaces, one direction** — `white_diary` store functions are the source of truth.
HTTP routes call into `white_diary`; `white_diary` has no web-framework dependency.

**No dedicated MCP** — the existing `lucid_nonsense_access` MCP already exposes file
read/write on the project root. YAML diary files are accessible via `open_lucid_nonsense_file`
and `find_files_in_lucid_nonsense` without a new server.

## Risks / Trade-offs
- YAML files aren't queryable cross-song (e.g. "all entries by prism this week").
  Acceptable for current scale. Add an index file or SQLite if cross-song queries emerge.
- No file locking on concurrent writes. Acceptable: only one pipeline phase runs at a
  time per song; no parallel writers expected.

## Directory layout
```
packages/diary/src/
  entries/
    <song_slug>/
      <uuid>.yml    ← writable at any lifecycle stage, by any author
  white_diary/
    __init__.py     ← exports ENTRIES_DIR, DiaryEntry, store functions
    diary_entry.py
    store.py
```

The diary root is `white_diary.ENTRIES_DIR` — fully independent of `shrink_wrapped_dir`,
production directories, and pipeline state. ThreadKeepr can write an entry (e.g. an EVP
transcription) before a song has ever been initialized as a production project.

`make_diary_router(ENTRIES_DIR)` is the canonical registration in `candidate_server.py`.
POST never returns 404 — the per-song subdirectory is created on first write.
GET/PUT/DELETE on a missing `entry_id` return 404.
