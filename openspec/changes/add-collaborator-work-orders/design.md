# Design: Collaborator Registry, Work Orders, and Production Calendar

## Context

White now has a mixed pipeline: AI-generated MIDI loops reviewed and promoted by Gabriel, then
recorded over or replaced by human performers. The performers (Graham for drums, Karley/Kate
Koherence for vocals, etc.) are hired through AirGigs and SoundBetter, negotiated ad hoc, and
tracked informally. Neither AirGigs nor SoundBetter expose a public API, so White cannot push
to those platforms directly; it can only prepare structured content for copy-paste or email.

Google Calendar MCP and Gmail MCP are already available in the Claude Code harness and can
be called from the Flask API endpoints.

## Goals / Non-Goals

**Goals:**
- Structured, persistent collaborator profiles (global, not per-song)
- Work orders auto-populated from pipeline data with a single UI action
- Calendar reminders for budget timing and musician availability gaps
- Royalty split recording per song per collaborator
- Creative artifact packets drawn from the existing chain artifact system
- Budget ledger per work order

**Non-Goals:**
- Direct API integration with AirGigs or SoundBetter (no public API exists)
- Invoice generation or payment processing
- Contract templating (out of scope for v1)
- Touring calendar scraping from external sources (user-entered only)

## Data Model Decisions

### Collaborator — global, not per-song
A `Collaborator` lives in a project-level registry at
`packages/core/src/white_core/music/core/collaborators/`, one YAML file per person
(slug-named). Placing it under `white_core/music/core/` keeps it alongside the other
music production primitives (`key_signature.py`, `duration.py`, etc.) and makes it
importable from `white_core` without depending on `white_production`.
It is not tied to a song. Songs reference collaborators by ID inside their `WorkOrder` records.

### WorkOrder — per-song, per-collaborator
A `WorkOrder` lives alongside production data under
`<thread>/production/<song_slug>/work_orders/<collaborator_id>.yml`.
One work order per collaborator per song (enforced; revision means updating the same file).

### Artifact Packet — references ChainArtifactType
Rather than copying files, the work order stores a list of `ChainArtifactType` values the
sender wants to include. The generator resolves these to actual file paths from the song's
shrinkwrap chain at send time. This means the packet stays current if files are regenerated.

Three new values are added to `ChainArtifactType` in `white_core` (centralised vocabulary):
- `CHROMATIC_BRIEF` — generated prose description of the song's color target
- `PRODUCTION_PLAN_ARTIFACT` — the `production_plan.yml` rendered as a human-readable brief
- `MELODY_MIDI_STEM` — the approved melody MIDI file

Existing types used for the packet:
- `ChainArtifactType.PROPOSAL` — the song proposal YAML rendered as readable text
- `ChainArtifactType.CHARACTER_SHEET` — character sheet HTML (if generated)

### Pydantic models live in white_core/music/core/ per user decision
`Collaborator`, `WorkOrder`, and their nested types (`AvailabilityWindow`, `PlatformProfile`,
`RoyaltySplit`) live in `white_core/music/core/`, alongside `key_signature.py`,
`duration.py`, etc. They are music production entities, not White-world narrative concepts.

Supporting enums (`CollaboratorRole`, `CollaboratorPlatform`, `WorkOrderStatus`,
`PROAffiliation`, `BudgetStatus`) live in `white_core/enums/` per project convention — all
`str, Enum` so they round-trip cleanly through YAML and JSON.

### Calendar events
Each `WorkOrder` stores at most one `calendar_event_id` (GCal). Creating a new reminder
replaces the previous event_id (the old event is deleted first). Follow-up events use
the song title + collaborator name + reason in the event description.

### Budget
Simple flat fields on `WorkOrder`: `budget_agreed`, `budget_paid`, `budget_currency`.
No sub-ledger in v1 — if multiple payments are needed, `budget_paid` is updated incrementally.

## Alternatives Considered

**Option A: Store work orders in a central database**
Rejected — White is file-based throughout (YAML review files, production plans, etc.);
introducing a database for just this feature would break the pattern and complicate the
single-repo no-server-required local workflow.

**Option B: Per-song collaborator list (not global registry)**
Rejected — the same musicians (Graham, Karley, etc.) appear across songs; a global registry
avoids re-entering contact/PRO/split data per song.

**Option C: Separate `white_production` server process**
Rejected — routes are added to the existing `white_api` Flask app to avoid running two
servers during a session.

## Risks / Trade-offs

- GCal MCP calls from a Flask endpoint require the harness to be running in a context where
  MCP tools are available; in headless/CI this will silently no-op. Calendar features must
  degrade gracefully (work order saves without event_id on MCP failure).
- Gmail draft creation sends an email draft to the user's Gmail; confirm this is acceptable
  before implementing the draft endpoint.

## Migration Plan

No existing data to migrate — `white_production` is greenfield. New files are additive only.

## Open Questions

- ~~Should `CHROMATIC_BRIEF`, `PRODUCTION_PLAN_ARTIFACT`, `MELODY_MIDI_STEM` be added to
  `ChainArtifactType` or a separate enum?~~ **Resolved**: add to `ChainArtifactType` in
  `white_core` (centralised vocabulary).
- Should royalty splits default to a configurable template (e.g., "standard vocalist split") or
  always be blank? Recommendation: blank in v1, template support as follow-on.
