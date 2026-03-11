# singer-voice-resolution Specification

## Purpose
TBD - created by archiving change wire-singer-voices-registry. Update Purpose after archive.
## Requirements
### Requirement: Singer Registry Load

`ace_studio_export.py` SHALL load `singer_voices.yml` via a helper function
`load_singer_registry(path)` that returns a dict keyed by lowercase singer name.

The registry path SHALL default to
`app/reference/mcp/ace_studio/singer_voices.yml` relative to the project root,
resolved from the module's `__file__`. Callers MAY supply an explicit path for
testing.

#### Scenario: Registry loads successfully

- **WHEN** `load_singer_registry()` is called
- **AND** `singer_voices.yml` exists and is valid YAML
- **THEN** a dict is returned keyed by lowercase singer name
- **AND** each value contains at least `ace_studio_voice`, `voice_type`, and
  `midi_range` fields

#### Scenario: Registry file missing

- **WHEN** `load_singer_registry()` is called
- **AND** the YAML file does not exist
- **THEN** an empty dict is returned
- **AND** a warning is logged naming the missing path

---

### Requirement: ACE Voice Name Resolution

`export_to_ace_studio` SHALL resolve the ACE Studio voice name from the registry
before calling `find_singer()`.

Resolution order:
1. Look up the singer name (case-insensitive) in the registry.
2. If found and `ace_studio_voice` is non-null → use that as the lookup name.
3. If found but `ace_studio_voice` is null → log a warning, use the White project
   name as a best-effort fallback.
4. If not found in the registry → use the White project name directly (debug log).

#### Scenario: Singer mapped with confirmed ACE voice

- **WHEN** `export_to_ace_studio` is called with singer `"Shirley"`
- **AND** the registry maps `shirley → ace_studio_voice: "Ember Rose"`
- **THEN** `ace.find_singer("Ember Rose")` is called
- **AND** the singer is loaded if found

#### Scenario: Singer in registry with null ACE voice

- **WHEN** `export_to_ace_studio` is called with singer `"Gabriel"`
- **AND** the registry maps `gabriel → ace_studio_voice: null`
- **THEN** a warning is logged: `"Singer 'Gabriel' has no ACE Studio voice assigned in singer_voices.yml; trying White name as fallback"`
- **AND** `ace.find_singer("Gabriel")` is called as best-effort
- **AND** if no match is found, singer loading is skipped and the export continues

#### Scenario: Singer absent from registry

- **WHEN** `export_to_ace_studio` is called with a singer name not in `singer_voices.yml`
- **THEN** a debug message is logged noting the name is not in the registry
- **AND** `ace.find_singer(<original_name>)` is called unchanged
- **AND** behaviour matches the current not-in-registry path

#### Scenario: Singer name empty or missing

- **WHEN** `export_to_ace_studio` is called and `singer_name` is empty or absent
  from `melody/review.yml`
- **THEN** singer loading is skipped entirely (no registry lookup)
- **AND** the export continues without error (existing behaviour preserved)

