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
web-framework dependency:

- `write_entry(entry: DiaryEntry, production_dir: Path) -> None`
  — writes or overwrites `<production_dir>/diary/<entry.id>.yml`; creates the
  `diary/` subdirectory if absent
- `load_entry(entry_id: str, production_dir: Path) -> DiaryEntry`
  — reads and deserialises the entry; raises `FileNotFoundError` if absent
- `list_entries(production_dir: Path) -> list[DiaryEntry]`
  — returns all entries in the diary directory sorted by `created_at` ascending;
  returns `[]` if the directory does not exist
- `delete_entry(entry_id: str, production_dir: Path) -> None`
  — removes the file; raises `FileNotFoundError` if absent

#### Scenario: Write and load round-trip
- **WHEN** `write_entry(entry, production_dir)` is called
- **AND** `load_entry(entry.id, production_dir)` is called
- **THEN** the loaded entry is equal to the original

#### Scenario: List sorted ascending
- **WHEN** three entries with distinct `created_at` values are written in non-chronological order
- **THEN** `list_entries` returns them ordered earliest-first

#### Scenario: List on missing diary dir
- **WHEN** `list_entries` is called for a `production_dir` that has no `diary/` subdirectory
- **THEN** an empty list is returned (no exception)

#### Scenario: Delete removes entry
- **WHEN** `delete_entry(entry.id, production_dir)` is called for an existing entry
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

The router SHALL be constructed via `make_diary_router(get_shrink_wrapped_dir)` and
registered in `candidate_server.py`.

GET, PUT, and DELETE for a non-existent `entry_id` SHALL return 404.
POST or GET to a `song_slug` whose production directory cannot be resolved SHALL return 404.

#### Scenario: Create and retrieve
- **WHEN** a POST to `/diary/my-song` is made with `author`, `body`, and `song_slug`
- **THEN** a 201 is returned containing the entry with a server-generated `id`
- **AND** a subsequent GET `/diary/my-song/{id}` returns 200 with the same body

#### Scenario: List returns entries in order
- **WHEN** GET `/diary/my-song` is called after two entries have been created
- **THEN** a 200 is returned with both entries in ascending `created_at` order

#### Scenario: Get missing entry returns 404
- **WHEN** GET `/diary/my-song/nonexistent-id` is called
- **THEN** a 404 is returned

#### Scenario: Delete then get returns 404
- **WHEN** DELETE `/diary/my-song/{entry_id}` is called for an existing entry
- **THEN** a 204 is returned
- **AND** a subsequent GET `/diary/my-song/{entry_id}` returns 404

#### Scenario: Post to unknown song returns 404
- **WHEN** POST `/diary/no-such-song` is called and no production directory exists for that slug
- **THEN** a 404 is returned
