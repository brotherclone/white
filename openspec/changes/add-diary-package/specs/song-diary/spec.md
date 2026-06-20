## ADDED Requirements

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
directory under the diary root (e.g. `<shrink_wrapped_dir>/diary/<song_slug>/`).

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
