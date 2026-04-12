## ADDED Requirements

### Requirement: Auto Track Selection
The ACE Studio client SHALL expose a `find_available_track()` method that returns
the index of the first track with no clips. If all tracks have clips, it SHALL
return 0 with a warning.

#### Scenario: Empty track found
- **WHEN** `find_available_track()` is called
- **AND** at least one track has no clips
- **THEN** the index of the first empty track is returned

#### Scenario: All tracks occupied
- **WHEN** `find_available_track()` is called
- **AND** all tracks have one or more clips
- **THEN** 0 is returned with a logged warning

---

### Requirement: Section-Aware Clip Placement
The ACE Studio client SHALL expose an `add_section_clips()` method that places one
clip per section, each with pre-loaded notes and lyrics, based on tick-accurate
section boundaries derived from the song BPM and bar counts.

#### Scenario: Multiple sections exported as separate clips
- **WHEN** `add_section_clips()` is called with a list of section dicts
- **THEN** one clip is created per section at the correct tick position
- **AND** each clip contains its section's notes and lyrics
- **AND** clip names match the section label

#### Scenario: Single-section song
- **WHEN** `add_section_clips()` is called with one section
- **THEN** a single named clip is placed at tick 0
