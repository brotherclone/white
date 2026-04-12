## ADDED Requirements

### Requirement: ACE Studio Song Association Block
`song_context.yml` SHALL contain an `ace_studio` block written by the export
script on first export. The block records the project name, export timestamp,
track index, singer, and sections exported. A null `render_path` field is
included and filled in by `ace_studio_import` after render.

#### Scenario: Export writes association block
- **WHEN** `export_to_ace_studio()` completes successfully
- **THEN** `song_context.yml` is updated with an `ace_studio` block containing
  `project_name`, `exported_at`, `track_index`, `singer`, `sections_exported`
- **AND** `render_path` is null

#### Scenario: Re-export warns before overwriting
- **WHEN** `export_to_ace_studio()` is called and an `ace_studio` block already
  exists in `song_context.yml`
- **THEN** a warning is logged naming the previous export timestamp
- **AND** export continues (overwriting the block)
