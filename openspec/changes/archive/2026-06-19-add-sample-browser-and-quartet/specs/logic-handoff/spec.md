## ADDED Requirements

### Requirement: Samples Sweep During Handoff
`logic_handoff.handoff()` SHALL copy any WAV files found in
`<production_dir>/samples/` into `$LOGIC_OUTPUT_DIR/<thread>/<title>/Samples/`,
creating the `Samples/` subdirectory if it does not exist.

If `<production_dir>/samples/` is absent or empty, the step SHALL be silently skipped.
File names SHALL be preserved. This sweep runs after the approved MIDI copy step so
that a full re-handoff keeps the Logic project in sync with any samples previously
exported via `POST /samples/{segment_id}/export`.

#### Scenario: Samples present during handoff
- **WHEN** `handoff(production_dir)` is called and `<production_dir>/samples/` contains
  one or more `.wav` files
- **THEN** those files are copied to `$LOGIC_OUTPUT_DIR/<thread>/<title>/Samples/`
- **AND** file names are preserved

#### Scenario: No samples directory
- **WHEN** `handoff(production_dir)` is called and `<production_dir>/samples/` does not exist
- **THEN** the handoff completes normally with no error and no `Samples/` folder is created

#### Scenario: Samples directory empty
- **WHEN** `handoff(production_dir)` is called and `<production_dir>/samples/` exists
  but contains no WAV files
- **THEN** the handoff completes normally; `Samples/` is created but left empty
