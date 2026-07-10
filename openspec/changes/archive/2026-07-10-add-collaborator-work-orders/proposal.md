# Change: Add Collaborator Registry, Work Orders, and Production Calendar

## Why

The White production pipeline now involves real human performers alongside the AI-generated
MIDI skeleton — drums from Graham, guitar/bass/synths from Gabriel, vocals from Kate Koherence
(via AirGigs/SoundBetter). There is no structured place to track who is working on what, what
they need to do their job, what was agreed financially, or when to follow up. Work orders are
currently assembled manually; budgets live in a spreadsheet; follow-up dates live in memory.

This change introduces a lightweight production CRM inside `white_production`: a
Collaborator Registry, Work Orders that auto-generate from pipeline artifacts, a calendar
integration for reminders and deadlines, and a UI button on the recording board cell to
enact all of the above without leaving the board.

## What Changes

- **Pydantic models** in `white_core` — `Collaborator`, `WorkOrder`, `RoyaltySplit`,
  `AvailabilityWindow`; new enums `CollaboratorRole`, `CollaboratorPlatform`, `WorkOrderStatus`,
  `PROAffiliation`
- **Collaborator Registry** in `white_production` — YAML-backed per-collaborator store
  (global, not per-song); CRUD via registry module
- **Work Order generator** in `white_production` — builds a `WorkOrder` from a song proposal
  + approved pipeline phases; selects chain artifacts for the creative packet
- **Production Calendar** — thin wrapper around Google Calendar MCP; creates/updates GCal
  events for work order deadlines and follow-up reminders; stores `calendar_event_id` on
  `WorkOrder`
- **API endpoints** in `white_api` — CRUD for collaborators and work orders; draft-email
  endpoint that delegates to Gmail MCP
- **Recording board UI** in `packages/client` — recording cell gains a work order HUD:
  status badge, budget indicator, calendar reminder chip, and a "Create Work Order" drawer

## Impact

- New specs: `collaborator-registry`, `work-order`, `production-calendar`, `recording-board`
- Affected code:
  - `packages/core/src/white_core/music/core/` — `collaborator.py`, `work_order.py`
  - `packages/core/src/white_core/enums/` — `collaborator_role.py`, `collaborator_platform.py`,
    `work_order_status.py`, `pro_affiliation.py`, `budget_status.py`
  - `packages/core/src/white_core/enums/chain_artifact_type.py` — add `CHROMATIC_BRIEF`,
    `PRODUCTION_PLAN_ARTIFACT`, `MELODY_MIDI_STEM`
  - `packages/production/src/white_production/` — `collaborator_registry.py`,
    `work_order_store.py`, `work_order_generator.py`, `production_calendar.py`
  - `packages/api/src/white_api/` — `collaborator_routes.py`, `work_order_routes.py`
  - `packages/client/app/board/` — recording cell work order drawer
