## 1. white_core — Enums

- [x] 1.1 `packages/core/src/white_core/enums/collaborator_role.py` — `CollaboratorRole`
- [x] 1.2 `packages/core/src/white_core/enums/collaborator_platform.py` — `CollaboratorPlatform`
- [x] 1.3 `packages/core/src/white_core/enums/pro_affiliation.py` — `PROAffiliation`
- [x] 1.4 `packages/core/src/white_core/enums/work_order_status.py` — `WorkOrderStatus`
- [x] 1.5 `packages/core/src/white_core/enums/budget_status.py` — `BudgetStatus`
- [x] 1.6 Add new `ChainArtifactType` values: `CHROMATIC_BRIEF`, `PRODUCTION_PLAN_ARTIFACT`,
       `MELODY_MIDI_STEM` to `chain_artifact_type.py`
- [x] 1.7 Export all new enums from `white_core/enums/__init__.py`

## 2. white_core — Pydantic Models

- [x] 2.1 `packages/core/src/white_core/music/core/collaborator.py` — `AvailabilityWindow`,
       `PlatformProfile`, `Collaborator`
- [x] 2.2 `packages/core/src/white_core/music/core/work_order.py` — `RoyaltySplit`, `WorkOrder`
- [x] 2.3 Tests: `packages/core/tests/music/core/test_collaborator.py` — round-trip YAML, enum serialisation
- [x] 2.4 Tests: `packages/core/tests/music/core/test_work_order.py` — round-trip YAML, date fields

## 3. white_production — Registry and Storage

- [x] 3.1 `packages/production/src/white_production/collaborator_registry.py` —
       `load_collaborator`, `save_collaborator`, `list_collaborators`, `delete_collaborator`
- [x] 3.2 `packages/production/src/white_production/work_order_store.py` —
       `load_work_order`, `save_work_order`, `list_work_orders`
- [x] 3.3 Tests: `packages/production/tests/test_collaborator_registry.py`
- [x] 3.4 Tests: `packages/production/tests/test_work_order_store.py`

## 4. white_production — Generator and Calendar

- [x] 4.1 `packages/production/src/white_production/work_order_generator.py` —
       `generate_work_order(production_dir, collaborator_id, role, platform) -> WorkOrder`
- [x] 4.2 `packages/production/src/white_production/production_calendar.py` —
       `create_followup_event`, `create_deadline_event`, `delete_event`,
       `update_work_order_calendar`; GCal MCP calls wrapped with try/except degrading to `None`
- [x] 4.3 Tests: `packages/production/tests/test_work_order_generator.py` (mock production dir)
- [x] 4.4 Tests: `packages/production/tests/test_production_calendar.py` (mock MCP)

## 5. white_api — Routes

- [x] 5.1 `packages/api/src/white_api/routes/collaborators.py` — FastAPI router with all
       `/collaborators` endpoints
- [x] 5.2 `packages/api/src/white_api/routes/work_orders.py` — FastAPI router with all
       `/production/work-orders` endpoints including `generate`, `draft-email`
- [x] 5.3 Register routers in `candidate_server.py` via `create_app`
- [x] 5.4 Tests: `packages/api/tests/test_collaborator_routes.py`

## 6. packages/client — Recording Board UI

- [x] 6.1 `WorkOrderHud` component in recording cell — status badge, budget line, calendar chip
- [x] 6.2 `WorkOrderDrawer` component — four-tab drawer (Brief, Collaborator, Budget, Calendar)
- [x] 6.3 `CollaboratorProfileCard` component within Collaborator tab
- [x] 6.4 Collaborator picker in drawer (fetches `GET /collaborators`)
- [x] 6.5 Calendar tab — follow-up and deadline date fields
- [x] 6.6 "Draft Email" button with toast
- [x] 6.7 Wire all new API types into `packages/client/lib/types.ts`
- [x] 6.8 Wire new API calls into `packages/client/lib/api.ts`

## 7. packages/client — Collaborator Manager UI

- [ ] 7.1 `/collaborators` page (or modal reachable from WorkOrderDrawer "Edit Collaborator" link)
- [ ] 7.2 List view: all collaborators with name, roles, availability status
- [ ] 7.3 Create/Edit form: id, name, roles (multi-select), email, photo_url, platforms, website,
       socials, pro_affiliation, pro_number, availability_windows, notes
- [ ] 7.4 Delete with confirmation (warns if active work orders exist — API already enforces)
- [ ] 7.5 Link from WorkOrderDrawer Collaborator tab → Collaborator Manager (open/edit profile)
