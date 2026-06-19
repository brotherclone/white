## ADDED Requirements

### Requirement: Samples Retrieval Endpoints
The FastAPI backend SHALL expose three endpoints for chromatic sample retrieval and export:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/samples` | Return top-N CLAP-scored segments for the active song's color |
| GET | `/audio/{segment_id}` | Stream the pre-extracted WAV for a segment |
| POST | `/samples/{segment_id}/export` | Copy the segment WAV to the Logic Samples folder |

`GET /samples` SHALL accept an optional `?top_n=N` query parameter (default 20). It SHALL
call `white_composition.retrieve_samples.retrieve_by_color` using the active song's
`rainbow_color` field and return a JSON array of objects with fields:
`segment_id`, `song_slug`, `color`, `match` (float 0–1), `audio_url` (`/audio/{segment_id}`).

`GET /audio/{segment_id}` SHALL resolve the WAV path from the CLAP index and return the
file bytes with `Content-Type: audio/wav`. It SHALL return 404 if the WAV is not present
on disk. No time-slicing is required — `staged_raw_material` files are pre-extracted segments.

`POST /samples/{segment_id}/export` SHALL copy the segment WAV to
`$LOGIC_OUTPUT_DIR/<thread_slug>/<song_title>/Samples/<segment_id>.wav`, creating the
`Samples/` subdirectory if absent. It SHALL return 503 if `LOGIC_OUTPUT_DIR` is not set
and 404 if the WAV is not found.

`GET /samples` SHALL return 503 if no song is active (matching the behaviour of other
candidate endpoints).

#### Scenario: Samples list for active song
- **WHEN** `GET /samples` is called with an active song whose color is Orange
- **THEN** a JSON array is returned with up to 20 objects, each containing `segment_id`,
  `song_slug`, `color`, `match`, and `audio_url`
- **AND** results are sorted descending by `match`

#### Scenario: No active song
- **WHEN** `GET /samples` is called with no active song
- **THEN** a 503 response is returned

#### Scenario: Audio stream
- **WHEN** `GET /audio/{segment_id}` is called for a known segment
- **THEN** the WAV bytes are returned with `Content-Type: audio/wav`

#### Scenario: Audio not on disk
- **WHEN** `GET /audio/{segment_id}` is called but the WAV file is absent from the filesystem
- **THEN** a 404 response is returned

#### Scenario: Export to Logic
- **WHEN** `POST /samples/{segment_id}/export` is called with a valid segment and
  `LOGIC_OUTPUT_DIR` is set
- **THEN** the WAV is copied to `$LOGIC_OUTPUT_DIR/<thread_slug>/<title>/Samples/<segment_id>.wav`
- **AND** `{"ok": true, "dest": "<path>"}` is returned

#### Scenario: Export without LOGIC_OUTPUT_DIR
- **WHEN** `POST /samples/{segment_id}/export` is called and `LOGIC_OUTPUT_DIR` is not set
- **THEN** a 503 response is returned

---

### Requirement: Sample Browser Panel
The `/candidates` page SHALL display a **Samples** panel below the candidate table.
The panel SHALL be visible regardless of the active phase filter and SHALL load on page
mount using `GET /samples`.

Each row in the panel SHALL show:
- Rank (1-based)
- `segment_id`
- Source song slug
- Color chip (matching the segment's color)
- Match score (0–1, two decimal places)
- An inline `<audio>` player with `src` set to the segment's `audio_url`
- An **Export** button that calls `POST /samples/{segment_id}/export`

After a successful export, the row's Export button SHALL change to "Exported ✓" and
become disabled for the duration of the session. An error toast SHALL be shown if the
export fails (e.g., 503 LOGIC_OUTPUT_DIR not set).

The panel SHALL be collapsible (collapsed by default) so it does not crowd the
candidate list.

#### Scenario: Samples panel loads on mount
- **WHEN** `/candidates` is opened with an active song
- **THEN** the Samples panel is present (collapsed) at the bottom of the page

#### Scenario: Expanding shows ranked rows
- **WHEN** the user expands the Samples panel
- **THEN** up to 20 ranked segment rows are shown, each with audio player and Export button

#### Scenario: Audio playback
- **WHEN** the user clicks play on a sample row's audio player
- **THEN** the browser plays the WAV streamed from `GET /audio/{segment_id}`

#### Scenario: Export marks row as exported
- **WHEN** the Export button is clicked and the server returns 200
- **THEN** the button label changes to "Exported ✓" and is disabled

#### Scenario: Export failure toast
- **WHEN** the Export button is clicked and the server returns 503
- **THEN** an error toast is shown: "LOGIC_OUTPUT_DIR not set — add it to .env"

---

### Requirement: Quartet Button Alongside Handoff
The **Generate Quartet** button SHALL appear in the pipeline status strip alongside the
"Handoff to Logic" button whenever melody is promoted and the quartet phase has not yet
been generated.

The button SHALL be shown when `quartetStatus` is absent (`undefined`), `null`, or
`"pending"`. It SHALL be hidden (replaced by a status indicator) once quartet reaches
`"in_progress"`, `"generated"`, or `"promoted"`.

#### Scenario: Button shown when melody promoted and quartet not started
- **WHEN** melody phase is promoted AND quartet status is absent or "pending"
- **THEN** a "Generate Quartet" button appears in the pipeline strip alongside "Handoff to Logic"

#### Scenario: Button shown when quartet pending
- **WHEN** melody phase is promoted AND `song_context.yml` has `quartet: pending`
- **THEN** the "Generate Quartet" button is still shown (pending is treated as not-started)

#### Scenario: Button hidden while generating
- **WHEN** the quartet phase is in_progress or the generation job is running
- **THEN** a "⟳ strings…" status indicator replaces the button

#### Scenario: Button hidden after generation
- **WHEN** quartet status is "generated" or "promoted"
- **THEN** no Generate Quartet button is shown; a status indicator reflects the current state
