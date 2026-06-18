## 1. white_core — Enums

- [ ] 1.1 `packages/core/src/white_core/enums/collaborator_role.py` — `CollaboratorRole`
- [ ] 1.2 `packages/core/src/white_core/enums/collaborator_platform.py` — `CollaboratorPlatform`
- [ ] 1.3 `packages/core/src/white_core/enums/pro_affiliation.py` — `PROAffiliation`
- [ ] 1.4 `packages/core/src/white_core/enums/work_order_status.py` — `WorkOrderStatus`
- [ ] 1.5 `packages/core/src/white_core/enums/budget_status.py` — `BudgetStatus`
- [ ] 1.6 Add new `ChainArtifactType` values: `CHROMATIC_BRIEF`, `PRODUCTION_PLAN_ARTIFACT`,
       `MELODY_MIDI_STEM` to `chain_artifact_type.py`
- [ ] 1.7 Export all new enums from `white_core/enums/__init__.py`

## 2. white_core — Pydantic Models

- [ ] 2.1 `packages/core/src/white_core/music/core/collaborator.py` — `AvailabilityWindow`,
       `PlatformProfile`, `Collaborator`
- [ ] 2.2 `packages/core/src/white_core/music/core/work_order.py` — `RoyaltySplit`,
       `DeliverableSpec` (optional helper), `WorkOrder`
- [ ] 2.3 Tests: `packages/core/tests/test_collaborator.py` — round-trip YAML, enum serialisation
- [ ] 2.4 Tests: `packages/core/tests/test_work_order.py` — round-trip YAML, date fields

## 3. white_production — Registry and Storage

- [ ] 3.1 `packages/production/src/white_production/collaborator_registry.py` —
       `load_collaborator`, `save_collaborator`, `list_collaborators`, `delete_collaborator`
- [ ] 3.2 `packages/production/src/white_production/work_order_store.py` —
       `load_work_order`, `save_work_order`, `list_work_orders`
- [ ] 3.3 Tests: `packages/production/tests/test_collaborator_registry.py`
- [ ] 3.4 Tests: `packages/production/tests/test_work_order_store.py`

## 4. white_production — Generator and Calendar

- [ ] 4.1 `packages/production/src/white_production/work_order_generator.py` —
       `generate_work_order(production_dir, collaborator_id, role, platform) -> WorkOrder`
- [ ] 4.2 `packages/production/src/white_production/production_calendar.py` —
       `create_followup_event`, `create_deadline_event`, `delete_event`,
       `update_work_order_calendar`; GCal MCP calls wrapped with try/except degrading to `None`
- [ ] 4.3 Tests: `packages/production/tests/test_work_order_generator.py` (mock production dir)
- [ ] 4.4 Tests: `packages/production/tests/test_production_calendar.py` (mock MCP)

## 5. white_api — Routes

- [ ] 5.1 `packages/api/src/white_api/collaborator_routes.py` — Flask Blueprint with all
       `/api/v1/collaborators` endpoints
- [ ] 5.2 Work order endpoints added to `collaborator_routes.py` or separate
       `work_order_routes.py` — all `/api/v1/production/work-orders` endpoints including
       `generate`, `draft-email`, `remind`, `deadline`, and calendar DELETE
- [ ] 5.3 Register blueprints in `candidate_server.py`
- [ ] 5.4 Tests: `packages/api/tests/test_collaborator_routes.py`
- [ ] 5.5 Tests: `packages/api/tests/test_work_order_routes.py`

## 6. packages/client — Recording Board UI

- [ ] 6.1 `WorkOrderHud` component in recording cell — status badge, budget line, calendar chip
- [ ] 6.2 `WorkOrderDrawer` component — four-tab drawer (Brief, Collaborator, Budget, Calendar)
- [ ] 6.3 `CollaboratorProfileCard` component within Collaborator tab
- [ ] 6.4 Collaborator typeahead picker (fetches `GET /api/v1/collaborators`)
- [ ] 6.5 Calendar tab — "Set Reminder" and "Set Deadline" forms calling calendar endpoints
- [ ] 6.6 "Draft Email" button with toast
- [ ] 6.7 Wire all new API types into `packages/client/lib/types.ts`
- [ ] 6.8 Wire new API calls into `packages/client/lib/api.ts`
