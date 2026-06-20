# Song Diary

A lightweight per-song diary system that lets any agent, hook, or human collaborator
append timestamped narrative entries to a song's production history — independent of
pipeline state.

Entries live at `packages/diary/src/entries/<song_slug>/` so ThreadKeepr, Prism, and
system hooks can write to a diary before any production directory exists.

---

### Requirement: DiaryEntry Model
The `white_diary` package SHALL provide a `DiaryEntry` Pydantic model with the following
fields:

| Field | Type | Default | Notes |
|---|---|---|---|
| `id` | `str` | UUID auto-generated | Unique entry identifier |
| `song_slug` | `str` | required | Identifies the song |
| `phase` | `str \| None` | `None` | Free-form pipeline phase label |
| `author` | `str` | required | Who wrote it — e.g. "prism", "threadkeepr", "system", "claude", "gabriel" |
| `created_at` | `datetime` | UTC now | Set automatically on construction |
| `title` | `str \| None` | `None` | Optional headline |
| `body` | `str` | required | Markdown narrative content |
| `tags` | `list[str]` | `[]` | Freeform labels |
| `metadata` | `dict[str, Any]` | `{}` | Catch-all for hook-specific fields |

#### Scenario: Minimal construction
- **WHEN** a `DiaryEntry` is constructed with only `song_slug`, `author`, and `body`
- **THEN** `id` is a non-empty UUID string, `created_at` is a UTC-aware datetime,
  `phase` is `None`, `tags` is `[]`, and `metadata` is `{}`

#### Scenario: Full YAML round-trip
- **WHEN** a fully-populated `DiaryEntry` is serialised to YAML and loaded back
- **THEN** all fields are equal to the original, including `created_at` timezone


### Requirement: Diary Store
The `white_diary` package SHALL expose four functions importable directly without any
web-framework dependency. Each function takes a `diary_dir: Path` — the per-song
directory under the diary root (e.g. `<entries_dir>/<song_slug>/`).

- `write_entry(entry: DiaryEntry, diary_dir: Path) -> None`
  — writes or overwrites `<diary_dir>/<entry.id>.yml`; creates the directory if absent
- `load_entry(entry_id: str, diary_dir: Path) -> DiaryEntry`
  — reads and deserialises the entry; raises `FileNotFoundError` if absent
- `list_entries(diary_dir: Path) -> list[DiaryEntry]`
  — returns all entries sorted by `created_at` ascending; returns `[]` if directory absent
- `delete_entry(entry_id: str, diary_dir: Path) -> None`
  — removes the file; raises `FileNotFoundError` if absent

#### Scenario: Write and load round-trip
- **WHEN** `write_entry(entry, diary_dir)` is called
- **AND** `load_entry(entry.id, diary_dir)` is called
- **THEN** the loaded entry is equal to the original

#### Scenario: List sorted ascending
- **WHEN** three entries with distinct `created_at` values are written in non-chronological order
- **THEN** `list_entries` returns them ordered earliest-first

#### Scenario: List on missing diary dir
- **WHEN** `list_entries` is called for a path that does not exist
- **THEN** an empty list is returned (no exception)

#### Scenario: Delete removes entry
- **WHEN** `delete_entry(entry.id, diary_dir)` is called for an existing entry
- **THEN** a subsequent `load_entry` for the same id raises `FileNotFoundError`

#### Scenario: Load missing entry
- **WHEN** `load_entry` is called with an id that has no corresponding file
- **THEN** `FileNotFoundError` is raised


### Requirement: Diary HTTP Routes
`white_api` SHALL expose a diary router registered under the prefix `/diary` with the
following endpoints:

| Method | Path | Success status | Description |
|---|---|---|---|
| GET | `/diary/{song_slug}` | 200 | List all entries for a song, sorted by `created_at` ascending |
| GET | `/diary/{song_slug}/{entry_id}` | 200 | Fetch a single entry |
| POST | `/diary/{song_slug}` | 201 | Create a new entry (id and created_at may be omitted; server fills them) |
| PUT | `/diary/{song_slug}/{entry_id}` | 200 | Replace an existing entry |
| DELETE | `/diary/{song_slug}/{entry_id}` | 204 | Remove an entry |

The router SHALL be constructed via `make_diary_router(entries_dir: Path)` and
registered in `candidate_server.py` as `make_diary_router(ENTRIES_DIR)` where
`ENTRIES_DIR` is imported from `white_diary`.

Per-song entries SHALL live at `<entries_dir>/<song_slug>/` and the subdirectory SHALL
be created on first write. No production directory or shrink-wrapped dir is required.
POST to any `song_slug` SHALL always succeed.
GET/PUT/DELETE for a non-existent `entry_id` SHALL return 404.

#### Scenario: Create and retrieve
- **WHEN** a POST to `/diary/new-song` is made before any production dir exists for that slug
- **THEN** a 201 is returned containing the entry with a server-generated `id`
- **AND** a subsequent GET `/diary/new-song/{id}` returns 200 with the same body

#### Scenario: List returns entries in order
- **WHEN** GET `/diary/my-song` is called after two entries have been created
- **THEN** a 200 is returned with both entries in ascending `created_at` order

#### Scenario: List for song with no entries returns empty list
- **WHEN** GET `/diary/brand-new-song` is called and no entries exist yet
- **THEN** a 200 is returned with an empty list

#### Scenario: Get missing entry returns 404
- **WHEN** GET `/diary/my-song/nonexistent-id` is called
- **THEN** a 404 is returned

#### Scenario: Delete then get returns 404
- **WHEN** DELETE `/diary/my-song/{entry_id}` is called for an existing entry
- **THEN** a 204 is returned
- **AND** a subsequent GET `/diary/my-song/{entry_id}` returns 404


### Requirement: DiaryEntry Client Type
`packages/client/lib/types.ts` SHALL export a `DiaryEntry` interface matching the
server model:

```ts
export interface DiaryEntry {
  id: string;
  song_slug: string;
  phase: string | null;
  author: string;
  created_at: string;
  title: string | null;
  body: string;
  tags: string[];
  metadata: Record<string, unknown>;
}
```

#### Scenario: Type is importable
- **WHEN** `DiaryEntry` is imported from `@/lib/types`
- **THEN** TypeScript resolves it without error


### Requirement: Diary API Client Functions
`packages/client/lib/api.ts` SHALL export two functions:

- `fetchDiaryEntries(songSlug: string): Promise<DiaryEntry[]>` — GET `/diary/{songSlug}`
- `createDiaryEntry(songSlug: string, entry: Omit<DiaryEntry, "id" | "created_at">): Promise<DiaryEntry>` — POST `/diary/{songSlug}`

#### Scenario: Fetch entries
- **WHEN** `fetchDiaryEntries("my-song")` is called and the server returns a list
- **THEN** the resolved value is an array of `DiaryEntry` objects

#### Scenario: Create entry
- **WHEN** `createDiaryEntry("my-song", { author: "gabriel", body: "note", ... })` is called
- **THEN** a POST is made to `/diary/my-song` and the resolved value includes a server-generated `id`


### Requirement: Diary Entry Modal on Composition Board
The composition board (`app/board/page.tsx`) SHALL replace the inline version-card
note `<input>` with a small `+ diary entry` button. Clicking it opens a `DiaryModal`
that follows the existing `LyricModal` visual pattern:

- Overlay: `fixed inset-0 z-50 bg-black/70 backdrop-blur-sm`, click outside to dismiss
- Card: `bg-zinc-900 border border-zinc-700 rounded-xl max-w-lg`, `shadow-2xl`
- Header: title "New diary entry", close `×` button, `border-b border-zinc-800`
- Form fields (`font-sans text-xs`):
  - **author** — text input, required, placeholder `"gabriel"`
  - **phase** — text input, pre-filled with the card's `MixStage` label, editable
  - **title** — text input, optional, placeholder `"optional headline"`
  - **body** — `<textarea>`, required, min 4 rows, placeholder `"what happened?"`
- Submit: full-width, `bg-violet-700 hover:bg-violet-600 text-white`, label `"Save entry"` / `"Saving…"` while in-flight
- On success: modal closes; on error: error message shown inside modal

The Version History section (rendered below the swim-lane columns) SHALL be removed.

#### Scenario: Button opens modal
- **WHEN** the `+ diary entry` button inside a version card is clicked
- **THEN** the `DiaryModal` opens with `phase` pre-filled to the card's stage label

#### Scenario: Submit creates entry and closes modal
- **WHEN** author and body are filled and "Save entry" is clicked
- **THEN** `createDiaryEntry` is called and the modal closes on success

#### Scenario: Empty body is blocked
- **WHEN** "Save entry" is clicked with an empty body field
- **THEN** the form does not submit (HTML5 `required` validation)

#### Scenario: Version history is absent
- **WHEN** the composition board renders with versions present
- **THEN** no "Version History" heading or list appears below the swim lanes
