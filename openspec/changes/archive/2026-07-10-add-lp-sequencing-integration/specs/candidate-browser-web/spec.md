## ADDED Requirements

### Requirement: Sides Navigation Entry
The client SHALL expose a "Sides" navigation link alongside the existing board,
candidates, songs, and collaborators links, routing to the LP-side sequencing page.

#### Scenario: Nav link present
- **WHEN** any client page renders the shared navigation
- **THEN** a "Sides" link is present and navigates to `/sides`

### Requirement: LP Consideration Status
Each song SHALL carry an `lp_consideration` status (`not_considered`, `candidate`,
`placed`) tracked in `manifest_bootstrap.yml`, independent of `lifecycle_status` and
mix stage.

#### Scenario: Default status
- **WHEN** a song's `manifest_bootstrap.yml` has no `lp_consideration` field
- **THEN** `scan_songs()` reports it as `not_considered`

#### Scenario: Manual status set
- **WHEN** `POST /songs/{id}/lp-consideration` is called with `{status: "candidate"}`
- **THEN** the song's `manifest_bootstrap.yml` is updated with
  `lp_consideration: candidate`
- **AND** `{"ok": true, "status": "candidate"}` is returned

#### Scenario: Auto-set to placed on side assignment
- **WHEN** a song is assigned to a side via the `add-lp-side-sequencing` assign/move
  endpoints
- **THEN** the song's `lp_consideration` is automatically set to `placed`

#### Scenario: Auto-revert on removal
- **WHEN** a song is removed from all sides via the remove endpoint
- **THEN** `lp_consideration` reverts to `candidate` if the song still has a mix file,
  or `not_considered` if it does not

#### Scenario: Filter pill and badge
- **WHEN** the song list filter bar renders
- **THEN** `lp: candidate` and `lp: placed` pills are available
- **AND** each song row shows a badge reflecting its current `lp_consideration` value
