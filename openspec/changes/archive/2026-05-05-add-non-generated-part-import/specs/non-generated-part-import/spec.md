## ADDED Requirements

### Requirement: Non-Generated Part Registration
The system SHALL accept an externally created MIDI file and register it as an approved
pipeline part, making it available to all downstream phases as if it had been promoted
through normal candidate review.

#### Scenario: Registration writes to approved directory
- **WHEN** `register_part(midi_path, phase="melody", section="chorus", label="chorus-nongenv1", production_dir=...)` is called
- **THEN** the MIDI file is copied to `<production_dir>/melody/approved/chorus-nongenv1.mid`
- **AND** `melody/review.yml` gains an entry with `generated: false`, `status: approved`, `label: chorus-nongenv1`, `section: chorus`, `scores: null`, `rank: null`

#### Scenario: Corrupt MIDI rejected
- **WHEN** `register_part()` is called with a MIDI file that cannot be parsed
- **THEN** a `ValueError` is raised and no files are written

#### Scenario: Duplicate label prevented
- **WHEN** `register_part()` is called with a label that already exists as `status: approved` in review.yml
- **THEN** a `ValueError` is raised with a message identifying the conflicting label

#### Scenario: Downstream phase consumes registered part
- **WHEN** a downstream phase (e.g. lyric pipeline) runs after a non-generated melody is registered
- **THEN** it reads the MIDI from `approved/chorus-nongenv1.mid` identically to a promoted generated melody

### Requirement: Registration API Endpoint
The candidate server SHALL expose a `POST /api/v1/production/register-part` endpoint that
accepts a MIDI file upload and registration metadata, delegating to `register_part()`.

#### Scenario: Successful registration via API
- **WHEN** a multipart POST is sent with a valid MIDI file, `production_dir`, `phase`, `section`, and `label`
- **THEN** the server responds 200 with the registered `CandidateEntry` as JSON

#### Scenario: Invalid MIDI returns 422
- **WHEN** the uploaded file is not a valid MIDI file
- **THEN** the server responds 422 with an error message
