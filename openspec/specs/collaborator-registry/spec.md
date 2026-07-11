# collaborator-registry Specification

## Purpose
TBD - created by archiving change add-collaborator-work-orders. Update Purpose after archive.
## Requirements
### Requirement: Collaborator Pydantic Models
`white_core` SHALL provide a `Collaborator` model and supporting types representing
a musician or audio professional who works on White songs.

The `Collaborator` model SHALL include:
- `id: str` — kebab-case slug, unique within the registry (e.g. `kate-koherence`)
- `name: str`
- `roles: list[CollaboratorRole]`
- `email: str | None`
- `photo_url: str | None`
- `platforms: list[PlatformProfile]` — each with `platform: CollaboratorPlatform` and `url: str`
- `website: str | None`
- `socials: dict[str, str]` — keys are platform names (`instagram`, `twitter`, etc.), values are URLs
- `pro_affiliation: PROAffiliation` — defaults to `PROAffiliation.NONE`
- `pro_number: str | None`
- `availability_windows: list[AvailabilityWindow]` — user-entered unavailability ranges
- `notes: str` — freeform, e.g. "prefers mp3 rough, not MIDI"

`AvailabilityWindow` SHALL have `unavailable_from: date`, `unavailable_until: date`,
and optional `reason: str`.

`PlatformProfile` SHALL have `platform: CollaboratorPlatform` and `url: str`.

Enums (`CollaboratorRole`, `CollaboratorPlatform`, `PROAffiliation`) SHALL be `str, Enum`
and live in `white_core/enums/`.

`CollaboratorRole` values: `vocalist`, `drummer`, `guitarist`, `bassist`, `keys`,
`strings`, `brass`, `mixing`, `mastering`, `other`.

`CollaboratorPlatform` values: `airgigs`, `soundbetter`, `direct`, `other`.

`PROAffiliation` values: `ascap`, `bmi`, `sesac`, `socan`, `prs`, `other`, `none`.

#### Scenario: Round-trip through YAML
- **WHEN** a `Collaborator` is serialised to YAML and reloaded via `model_validate`
- **THEN** all fields including nested `AvailabilityWindow` and `PlatformProfile` round-trip
  without loss

#### Scenario: Enum values serialise as strings
- **WHEN** a `Collaborator` with `pro_affiliation=PROAffiliation.ASCAP` is serialised to JSON
- **THEN** the JSON contains `"pro_affiliation": "ascap"` (not the Python enum repr)

### Requirement: Collaborator Registry Storage
`white_production` SHALL maintain a global collaborator registry as a directory of YAML
files at `<project_root>/collaborators/<collaborator_id>.yml`, one file per collaborator.

The registry module SHALL expose:
- `load_collaborator(id: str) -> Collaborator` — reads and validates the YAML file
- `save_collaborator(collaborator: Collaborator) -> None` — writes YAML (creates or overwrites)
- `list_collaborators() -> list[Collaborator]` — loads all files in the directory
- `delete_collaborator(id: str) -> None` — removes the YAML file; raises `ValueError` if
  the collaborator has active work orders in any production directory

The registry directory SHALL default to
`packages/core/src/white_core/music/core/collaborators/` relative to the project root,
where the project root is resolved from the `WHITE_PROJECT_ROOT` environment variable,
falling back to four directory levels above the `white_production` package source root.

#### Scenario: Save and reload
- **WHEN** `save_collaborator(c)` is called with a valid `Collaborator`
- **THEN** `<registry_dir>/<c.id>.yml` is created/overwritten
- **AND** `load_collaborator(c.id)` returns a model equal to `c`

#### Scenario: List returns all collaborators
- **WHEN** the registry directory contains three YAML files
- **THEN** `list_collaborators()` returns exactly three `Collaborator` instances

#### Scenario: Load unknown ID raises
- **WHEN** `load_collaborator("nobody")` is called and no matching file exists
- **THEN** a `FileNotFoundError` is raised

### Requirement: Collaborator API Endpoints
`white_api` SHALL expose REST endpoints for collaborator CRUD under `/api/v1/collaborators`.

| Method | Path | Description |
|--------|------|-------------|
| GET    | `/api/v1/collaborators` | List all collaborators |
| GET    | `/api/v1/collaborators/<id>` | Get single collaborator |
| POST   | `/api/v1/collaborators` | Create collaborator; body is `Collaborator` JSON |
| PUT    | `/api/v1/collaborators/<id>` | Update collaborator; body is full `Collaborator` JSON |
| DELETE | `/api/v1/collaborators/<id>` | Delete collaborator |

#### Scenario: Create collaborator
- **WHEN** `POST /api/v1/collaborators` is called with a valid `Collaborator` JSON body
- **THEN** the file is written and a 201 response returns the saved `Collaborator` as JSON

#### Scenario: Create with duplicate ID returns 409
- **WHEN** `POST /api/v1/collaborators` is called with an `id` that already exists
- **THEN** a 409 response is returned

#### Scenario: Delete with active work orders returns 409
- **WHEN** `DELETE /api/v1/collaborators/<id>` is called for a collaborator who has
  active (non-accepted) work orders
- **THEN** a 409 response is returned listing the affected song slugs

