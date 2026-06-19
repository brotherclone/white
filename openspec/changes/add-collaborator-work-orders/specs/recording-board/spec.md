## ADDED Requirements

### Requirement: Recording Cell Work Order HUD
The song board (`/board`) SHALL display a work order HUD inside each recording cell that
summarises collaborator status, budget, and upcoming calendar commitments at a glance.

The HUD SHALL show:
- Collaborator name and role (or "No collaborator assigned" if none)
- Work order status badge (`draft`, `sent`, `in_progress`, `delivered`, `accepted`,
  `revision_requested`)
- Budget line: agreed amount and paid amount (e.g. `$300 agreed / $150 paid`)
- Calendar chip: if `follow_up_date` is set, shows the date and reason as a pill
  (e.g. `📅 Follow up Aug 4 — on tour`)
- A "Create / View Work Order" button that opens the Work Order Drawer

#### Scenario: HUD with no work order
- **WHEN** a recording cell has no associated work order
- **THEN** the HUD shows "No work order" and a "Create Work Order" button
- **AND** no budget or calendar chip is shown

#### Scenario: HUD with draft work order
- **WHEN** a recording cell has a work order with `status: "draft"`
- **THEN** the HUD shows the collaborator name, a grey "draft" badge, and the budget agreed/paid line
- **AND** if `follow_up_date` is set, the calendar chip is visible

#### Scenario: Calendar chip visibility
- **WHEN** `follow_up_date` is today or in the future
- **THEN** the calendar chip renders in accent colour
- **WHEN** `follow_up_date` is in the past
- **THEN** the calendar chip renders in muted/warning colour to indicate overdue follow-up

### Requirement: Work Order Drawer
Clicking "Create / View Work Order" in the recording cell HUD SHALL open a right-side
drawer containing the full work order form.

The drawer SHALL have four tabs:
1. **Brief** — DAW specs (key, BPM, time sig, sections), creative direction, part notes,
   deliverable format, deadline
2. **Collaborator** — collaborator picker (typeahead from registry), photo, platform links,
   PRO affiliation, royalty split fields for this song
3. **Budget** — agreed amount, paid amount, currency selector, budget status selector
4. **Calendar** — follow-up reason input, date picker, "Set Reminder" button; deadline date
   picker, "Set Deadline" button; shows current GCal event details if one exists

The drawer SHALL include:
- A "Generate" button (only visible when no work order exists) that calls
  `POST /api/v1/production/work-orders/generate` and pre-fills the form
- A "Save" button that calls `PUT /api/v1/production/work-orders/<collaborator_id>`
- A "Draft Email" button that calls
  `POST /api/v1/production/work-orders/<collaborator_id>/draft-email` and shows a
  toast confirming the Gmail draft was created

#### Scenario: Generate pre-fills the drawer
- **WHEN** the user clicks "Generate" with a collaborator and role selected
- **THEN** the Brief tab fields are populated from the pipeline data
- **AND** the Collaborator tab shows the selected collaborator's profile
- **AND** the form is in an unsaved (dirty) state

#### Scenario: Save persists the work order
- **WHEN** the user edits the form and clicks "Save"
- **THEN** `PUT /api/v1/production/work-orders/<collaborator_id>` is called with the full
  `WorkOrder` JSON
- **AND** a success toast confirms "Work order saved"
- **AND** the HUD in the recording cell updates immediately

#### Scenario: Draft email
- **WHEN** the "Draft Email" button is clicked
- **THEN** `POST /api/v1/production/work-orders/<collaborator_id>/draft-email` is called
- **AND** a toast shows "Gmail draft created for <collaborator_name>"
- **WHEN** the collaborator has no email address
- **THEN** the toast shows "Add an email address to this collaborator first"

#### Scenario: Set reminder
- **WHEN** the user enters a follow-up reason and date in the Calendar tab and clicks
  "Set Reminder"
- **THEN** `POST /api/v1/production/work-orders/<collaborator_id>/remind` is called
- **AND** the calendar chip in the HUD updates to show the new date and reason
- **AND** a toast confirms "Reminder set for <date>"

### Requirement: Collaborator Profile Card
The Collaborator tab of the work order drawer SHALL render a profile card showing:
- Photo (or initials avatar if no `photo_url`)
- Name and roles
- Platform links as icon-buttons (AirGigs, SoundBetter, personal website)
- Social links
- PRO affiliation and number
- Availability windows — any current or upcoming unavailability shown as a warning banner
  (e.g. "On tour Jul 1 – Aug 3")

An "Edit Collaborator" link SHALL open a separate collaborator edit modal (not part of this
drawer's save flow).

#### Scenario: Availability warning
- **WHEN** today's date falls within any `AvailabilityWindow` for the collaborator
- **THEN** a yellow banner appears: "Currently unavailable: <reason>"
- **WHEN** an unavailability window starts within the next 14 days
- **THEN** an amber banner appears: "Upcoming unavailability: <date range>"

#### Scenario: Platform links
- **WHEN** the collaborator has a `PlatformProfile` for `airgigs`
- **THEN** an AirGigs icon-button is shown that opens the URL in a new tab
