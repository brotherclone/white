# production-calendar Specification

## Purpose
TBD - created by archiving change add-collaborator-work-orders. Update Purpose after archive.
## Requirements
### Requirement: Production Calendar Module
`white_production` SHALL provide a `production_calendar` module that creates and manages
Google Calendar events for work order deadlines and follow-up reminders, delegating to the
Google Calendar MCP.

The module SHALL expose:
- `create_followup_event(work_order, reason, follow_up_date) -> str | None` — creates a GCal
  event and returns the event ID, or `None` if the MCP is unavailable
- `create_deadline_event(work_order, deadline) -> str | None`
- `delete_event(event_id: str) -> None`
- `update_work_order_calendar(production_dir, work_order, *, reason=None, follow_up_date=None,
  deadline=None) -> WorkOrder` — convenience wrapper: deletes any existing event, creates the
  new one, updates `calendar_event_id` and `follow_up_date`/`deadline` on the work order,
  saves it, and returns the updated model

Calendar events SHALL include in their description:
- Song title
- Collaborator name and role
- Work order status
- The `reason` string
- A link / reference to the White production directory

Calendar events SHALL degrade gracefully: if the MCP call fails or the MCP server is
unavailable, the module logs a warning and returns `None`; the caller can still save a
work order without a `calendar_event_id`.

#### Scenario: Follow-up event created
- **WHEN** `create_followup_event(wo, reason="on tour until Aug 3", follow_up_date=date(2026, 8, 4))`
  is called
- **THEN** a GCal all-day event is created on `2026-08-04` with title
  `"White: Follow up — {collaborator.name} / {song_title}"`
- **AND** the returned event ID is a non-empty string

#### Scenario: MCP unavailable degrades gracefully
- **WHEN** the Google Calendar MCP server is not running
- **THEN** `create_followup_event` returns `None` and logs a warning
- **AND** no exception is raised

#### Scenario: Replacing an existing event
- **WHEN** `update_work_order_calendar` is called on a work order that already has
  a `calendar_event_id`
- **THEN** the old event is deleted before the new one is created
- **AND** the work order is saved with the new `calendar_event_id`

### Requirement: Production Calendar API Endpoints
`white_api` SHALL expose endpoints for calendar operations on work orders.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/production/work-orders/<collaborator_id>/remind` | Create a follow-up reminder |
| POST | `/api/v1/production/work-orders/<collaborator_id>/deadline` | Create a deadline event |
| DELETE | `/api/v1/production/work-orders/<collaborator_id>/calendar` | Remove calendar event |

`POST /remind` body: `{"reason": str, "follow_up_date": "YYYY-MM-DD"}`
`POST /deadline` body: `{"deadline": "YYYY-MM-DD"}`

Both endpoints save the updated work order and return it as JSON.

#### Scenario: Set follow-up reminder
- **WHEN** `POST /api/v1/production/work-orders/kate-koherence/remind` is called with
  `{"reason": "on tour until Aug 3", "follow_up_date": "2026-08-04"}`
- **THEN** a GCal event is created, the work order's `calendar_event_id` and
  `follow_up_date` are updated, and the updated `WorkOrder` JSON is returned

#### Scenario: Remove calendar event
- **WHEN** `DELETE /api/v1/production/work-orders/kate-koherence/calendar` is called
- **THEN** the GCal event identified by `calendar_event_id` is deleted
- **AND** `calendar_event_id` is set to `null` on the saved work order
- **WHEN** no `calendar_event_id` exists on the work order
- **THEN** a 404 response is returned

