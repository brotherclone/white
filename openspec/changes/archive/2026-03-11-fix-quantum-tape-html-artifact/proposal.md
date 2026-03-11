# Change: Fix Quantum Tape HTML Artifact — field mapping and UNKNOWN name

## Why

The Cassette Bearer (blue agent) generates a quantum tape label HTML artifact, but every
generated file is broken in two ways: (1) the file is named `UNKNOWN_ARTIFACT_NAME` because
`QuantumTapeLabelArtifact` never sets `artifact_name` from `title`, and (2) all nine
template content slots are blank because `quantum_tape.html` expects fields
(`year_documented`, `original_date`, `original_title`, `tapeover_date`, `tapeover_title`,
`subject_name`, `age_during`, `location`, `catalog_number`) that don't exist on the model.
The template was clearly designed with a richer data contract in mind, but the model and
agent wiring were never completed.

## What Changes

- Add the nine missing template fields to `QuantumTapeLabelArtifact`; all are optional with
  sane defaults so existing construction sites don't break **except** the `blue_agent.py`
  call-site which is updated in the same change.
- Fix `QuantumTapeLabelArtifact.__init__` to derive `artifact_name` from `title` (same
  pattern as `AlternateTimelineArtifact`).
- Update `blue_agent.generate_tape_label()` to map `AlternateTimelineArtifact` +
  `BiographicalPeriod` data into the new fields.
- Update `flatten()` to include the new fields.
- Update the mock YAML and unit tests to cover the new fields and assert HTML content is
  non-empty.

## Impact

- Affected specs: `quantum-tape-artifact` (new)
- Affected code:
  - `app/structures/artifacts/quantum_tape_label_artifact.py`
  - `app/agents/blue_agent.py` (generate_tape_label node)
  - `tests/mocks/quantum_tape_label_mock.yml`
  - `tests/structures/artifacts/test_quantum_tape_label_artifact.py`
