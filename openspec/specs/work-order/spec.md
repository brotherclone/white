# work-order Specification

## Purpose
TBD - created by archiving change add-collaborator-work-orders. Update Purpose after archive.
## Requirements
### Requirement: Work Order Pydantic Models
`white_core` SHALL provide a `WorkOrder` model representing a request sent to a
collaborator for a specific role on a specific song.

The `WorkOrder` model SHALL include:

**Identity**
- `id: str` — slug (`<collaborator_id>-<song_slug>`)
- `song_slug: str`
- `collaborator_id: str`
- `role: CollaboratorRole`
- `platform: CollaboratorPlatform` — where the work order was/will be sent

**DAW Specs** (auto-populated from song proposal)
- `key: str` — e.g. `"F# minor"`
- `bpm: int`
- `time_signature: str` — e.g. `"4/4"`
- `sections: list[str]` — e.g. `["Intro (4 bars)", "Verse A (8 bars)", "Chorus (8 bars)"]`

**Creative Direction**
- `creative_direction: str` — drawn from song proposal concept + chromatic target description
- `part_notes: str` — role-specific phrasing notes (default `""`)

**Artifact Packet**
- `artifact_types: list[str]` — `ChainArtifactType` values to include; generator resolves
  these to file paths at send time

**Deliverables**
- `deliverable_format: str` — e.g. `"48kHz/24bit WAV, dry (no reverb)"`
- `deadline: date | None`

**Budget**
- `budget_agreed: float | None`
- `budget_paid: float` — defaults to `0.0`
- `budget_currency: str` — defaults to `"USD"`
- `budget_status: BudgetStatus`

**Calendar**
- `calendar_event_id: str | None` — GCal event ID for the active reminder/deadline
- `follow_up_date: date | None`
- `follow_up_reason: str | None` — e.g. `"on tour until Aug 3"`, `"budget available on payday"`

**Royalties**
- `royalty_split: RoyaltySplit | None`

**Status and timestamps**
- `status: WorkOrderStatus`
- `created_at: datetime`
- `updated_at: datetime`

`RoyaltySplit` SHALL have: `collaborator_id: str`, `song_slug: str`, `mechanical_pct: float`,
`performance_pct: float`, `sync_pct: float`, `notes: str`.

`BudgetStatus` enum values (in `white_core/enums/`): `pending`, `agreed`, `invoiced`, `paid`.

`WorkOrderStatus` enum values (in `white_core/enums/`): `draft`, `sent`, `in_progress`,
`delivered`, `accepted`, `revision_requested`.

#### Scenario: Round-trip through YAML
- **WHEN** a `WorkOrder` is serialised to YAML and reloaded
- **THEN** all fields including nested `RoyaltySplit` and `date` fields round-trip without loss

#### Scenario: Budget totals
- **WHEN** `budget_agreed=300.0` and `budget_paid=150.0`
- **THEN** the model stores both values independently; no computed field required in v1

### Requirement: Work Order Storage
`white_production` SHALL store work orders at
`<thread>/production/<song_slug>/work_orders/<collaborator_id>.yml` — one file per
collaborator per song.

The work order module SHALL expose:
- `load_work_order(production_dir, collaborator_id) -> WorkOrder`
- `save_work_order(production_dir, work_order) -> None`
- `list_work_orders(production_dir) -> list[WorkOrder]`

#### Scenario: Save creates directory
- **WHEN** `save_work_order(prod_dir, wo)` is called and `work_orders/` does not exist
- **THEN** the directory is created and the YAML file written

#### Scenario: Overwrite preserves updated_at
- **WHEN** `save_work_order` is called on an existing work order with a changed field
- **THEN** `updated_at` is set to the current UTC time

### Requirement: Work Order Generator
`white_production` SHALL provide `generate_work_order(production_dir, collaborator_id,
role, platform)` that returns a pre-populated `WorkOrder` by reading the song proposal
and approved pipeline phases.

The generator SHALL:
1. Load the song proposal to populate `key`, `bpm`, `time_signature`
2. Read the approved chord `review.yml` to build the `sections` list with bar counts
3. Build `creative_direction` from the proposal's `concept` text and the chromatic
   target description for the song's `rainbow_color`
4. Populate `artifact_types` with `["proposal", "chromatic_brief"]` by default;
   add `"character_sheet"` if a character sheet artifact exists in the shrinkwrap chain;
   add `"melody_midi_stem"` if a vocalist role is requested
5. Set `status = WorkOrderStatus.DRAFT`, `created_at` and `updated_at` to UTC now

The generator SHALL NOT write the file — the caller saves via `save_work_order`.

#### Scenario: Generator populates DAW specs
- **WHEN** `generate_work_order(prod_dir, "kate-koherence", "vocalist", "soundbetter")` is called
- **THEN** the returned `WorkOrder` has `key`, `bpm`, and `time_signature` matching the
  song proposal, and `sections` lists each approved chord label with bar count

#### Scenario: Character sheet included when present
- **WHEN** the song's shrinkwrap chain contains a `ChainArtifactType.CHARACTER_SHEET` artifact
- **THEN** `"character_sheet"` is in `artifact_types`

#### Scenario: Melody MIDI stem for vocalist
- **WHEN** `role == CollaboratorRole.VOCALIST`
- **THEN** `"melody_midi_stem"` is added to `artifact_types` if an approved melody MIDI exists

### Requirement: Work Order API Endpoints
`white_api` SHALL expose REST endpoints for work orders under
`/api/v1/production/work-orders`.

| Method | Path | Description |
|--------|------|-------------|
| GET    | `/api/v1/production/work-orders` | List all work orders for active song |
| GET    | `/api/v1/production/work-orders/<collaborator_id>` | Get single work order |
| POST   | `/api/v1/production/work-orders/generate` | Generate a pre-populated draft; body: `{collaborator_id, role, platform}` |
| PUT    | `/api/v1/production/work-orders/<collaborator_id>` | Update work order |
| POST   | `/api/v1/production/work-orders/<collaborator_id>/draft-email` | Create Gmail draft for this work order |

The `draft-email` endpoint SHALL call the Gmail MCP `create_draft` tool with the work order
body rendered as plain text, addressed to the collaborator's `email` field. If the collaborator
has no email, it SHALL return 422.

#### Scenario: Generate endpoint returns draft work order
- **WHEN** `POST /api/v1/production/work-orders/generate` is called with valid body
- **THEN** a 200 response returns the `WorkOrder` JSON with `status: "draft"`
- **AND** no file is written (generate does not persist)

#### Scenario: Draft email with no collaborator email returns 422
- **WHEN** `POST /api/v1/production/work-orders/<id>/draft-email` is called for a
  collaborator with `email: null`
- **THEN** a 422 response is returned with `{"detail": "Collaborator has no email address"}`

#### Scenario: Work order list requires active song
- **WHEN** the server is in album mode and no song is activated
- **THEN** `GET /api/v1/production/work-orders` returns 503 (same guard as candidates)

