## ADDED Requirements

### Requirement: Root Landing Page
The application root (`/`) SHALL display a minimal landing page with two navigation
links: **Generation** (→ `/songs`) and **Composition Board** (→ `/board`).

The current songs index page SHALL move to `/songs`. All internal links that previously
pointed to `/` as the songs list SHALL be updated to `/songs`.

#### Scenario: Landing renders two links
- **WHEN** the user navigates to `/`
- **THEN** two clearly labelled links are rendered: "Generation" and "Composition Board"
- **AND** clicking "Generation" navigates to `/songs`
- **AND** clicking "Composition Board" navigates to `/board`

#### Scenario: Songs index accessible at /songs
- **WHEN** the user navigates to `/songs`
- **THEN** the full song list renders identically to the previous `/` behaviour

## MODIFIED Requirements

### Requirement: Next.js Frontend
The candidate browser SHALL display only the generation phases relevant to the
MIDI production pipeline. The `lyrics`, `decisions`, and `quartet` phases SHALL be
removed from the phase filter dropdown and the pipeline status strip.

The pipeline status strip SHALL show phases in this order:
`chords → drums → bass → melody`

Backend support for `lyrics`, `decisions`, and `quartet` (API endpoints, pipeline runner)
is preserved; only the web UI omits them.

The `← Songs` breadcrumb on `/candidates` SHALL link to `/songs` (not `/`).

#### Scenario: Phase filter shows generation phases only
- **WHEN** the user opens the phase filter dropdown on `/candidates`
- **THEN** the options are: All phases, chords, drums, bass, melody
- **AND** lyrics, decisions, and quartet are not listed

#### Scenario: Pipeline strip stops at melody
- **WHEN** the pipeline status strip renders
- **THEN** it shows status indicators for: chords, drums, bass, melody only

#### Scenario: Songs breadcrumb links to /songs
- **WHEN** the user is on `/candidates`
- **THEN** the `← Songs` breadcrumb links to `/songs`, not `/`
