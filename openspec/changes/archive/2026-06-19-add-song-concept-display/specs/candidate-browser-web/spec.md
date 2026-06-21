## ADDED Requirements

### Requirement: Song Concept Field
The song entry returned by `GET /songs` and `GET /songs/active` SHALL include a
`concept` field. The value SHALL be read from `song_context.yml` in the production
directory if that file exists and contains a non-empty `concept` key; otherwise the
field SHALL be `null`.

`manifest_bootstrap.yml` is NOT modified — concept is sourced exclusively from
`song_context.yml`.

#### Scenario: Concept present in song_context.yml
- **WHEN** `GET /songs` is called and a production directory has a `song_context.yml`
  with a non-empty `concept` field
- **THEN** the song entry for that production includes `"concept": "<text>"`

#### Scenario: Concept absent — no song_context.yml
- **WHEN** `GET /songs` is called and a production directory has no `song_context.yml`
  (song is in ideation stage)
- **THEN** the song entry for that production includes `"concept": null`

#### Scenario: Concept absent — song_context.yml exists but concept is empty
- **WHEN** `GET /songs` is called and `song_context.yml` exists but `concept` is an
  empty string or missing key
- **THEN** the song entry for that production includes `"concept": null`

#### Scenario: Active song includes concept
- **WHEN** `GET /songs/active` is called and the active song has a concept
- **THEN** the returned object includes `"concept": "<text>"`

---

### Requirement: Concept Display in Candidate Browser
The `/candidates` page SHALL render a concept block between the breadcrumb and the phase
toolbar when `activeSong.concept` is non-null and non-empty.

The block SHALL:
- Default to a 3-line clamp (overflow hidden)
- Show a "Show more" / "Show less" toggle below the text when the content exceeds 3 lines
- Be hidden (not rendered) when `activeSong` is null or `activeSong.concept` is null

#### Scenario: Concept block visible
- **WHEN** `/candidates` is loaded and the active song has a non-null concept
- **THEN** the concept text is displayed above the toolbar, clamped to 3 lines by default

#### Scenario: Toggle expands concept
- **WHEN** the user clicks "Show more" on the clamped concept block
- **THEN** the full concept text is revealed and the toggle label changes to "Show less"

#### Scenario: Toggle collapses concept
- **WHEN** the user clicks "Show less" on the expanded concept block
- **THEN** the text returns to a 3-line clamp

#### Scenario: Concept block hidden when absent
- **WHEN** `/candidates` is loaded and `activeSong.concept` is null
- **THEN** no concept block is rendered and the page layout is unchanged
