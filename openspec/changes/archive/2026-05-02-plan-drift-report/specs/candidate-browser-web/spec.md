## ADDED Requirements

### Requirement: Plan Drift Report API

The candidate server SHALL expose three endpoints for generating and retrieving the plan
drift report for the active song:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/drift-report` | Return the current `plan_drift_report.yml` as JSON; 404 if absent |
| POST | `/drift-report` | Start a background job to generate (or regenerate) the drift report |
| GET | `/drift-report/status` | Return the current job state for the drift report background job |

`POST /drift-report` accepts an optional JSON body `{"use_claude": bool}` (default `true`).
It requires both `production_plan.yml` and `arrangement.txt` to exist in the production
directory; missing either returns 422.

The background job state follows the same shape as `/handoff/status`:
`{status, started_at, finished_at, error}` where `status` is one of `idle`, `running`,
`done`, or `error`.

#### Scenario: GET with report absent

- **WHEN** `GET /drift-report` is called and `plan_drift_report.yml` does not exist
- **THEN** a 404 response is returned

#### Scenario: GET with report present

- **WHEN** `GET /drift-report` is called and `plan_drift_report.yml` exists
- **THEN** a 200 response is returned with the report fields as JSON, including
  `song_title`, `proposed_sections`, `actual_sections`, `drift`, `bar_deltas`,
  `energy_arc_correlation`, and `summary`

#### Scenario: POST starts background job

- **WHEN** `POST /drift-report` is called and both `production_plan.yml` and
  `arrangement.txt` exist in the production directory
- **THEN** a background job is started and `{"status": "running", "started_at": "..."}` is returned

#### Scenario: POST missing arrangement

- **WHEN** `POST /drift-report` is called but `arrangement.txt` is absent
- **THEN** a 422 response is returned

#### Scenario: POST missing production plan

- **WHEN** `POST /drift-report` is called but `production_plan.yml` is absent
- **THEN** a 422 response is returned

#### Scenario: POST duplicate job

- **WHEN** `POST /drift-report` is called while a drift report job is already running
- **THEN** a 409 response is returned

#### Scenario: Status endpoint

- **WHEN** `GET /drift-report/status` is called
- **THEN** the current job state is returned with `status`, `started_at`, `finished_at`,
  and `error` fields
