# quantum-tape-artifact Specification

## Purpose
TBD - created by archiving change fix-quantum-tape-html-artifact. Update Purpose after archive.
## Requirements
### Requirement: Artifact Name Derivation
`QuantumTapeLabelArtifact` SHALL derive `artifact_name` from `title` during `__init__`
when `artifact_name` is not already explicitly set, so that the generated `file_name`
reflects the tape title rather than the default `UNKNOWN_ARTIFACT_NAME`.

#### Scenario: title set at construction
- **WHEN** `QuantumTapeLabelArtifact` is constructed with a `title` value
- **THEN** `artifact_name` is set to the sanitized form of `title`
- **AND** `file_name` contains the sanitized title (not `UNKNOWN_ARTIFACT_NAME`)

#### Scenario: artifact_name explicitly overridden
- **WHEN** `QuantumTapeLabelArtifact` is constructed with an explicit `artifact_name`
- **THEN** the explicit `artifact_name` is preserved unchanged

---

### Requirement: Template Field Contract
`QuantumTapeLabelArtifact` SHALL expose all nine fields required by `quantum_tape.html`
so that `save_file()` produces HTML with fully populated content slots.

The nine required fields are:
- `year_documented` — year the tape was archived
- `original_date` — A-side real-timeline label date
- `original_title` — A-side real-timeline label text
- `tapeover_date` — B-side alternate-timeline date range (formatted string)
- `tapeover_title` — B-side alternate-timeline title
- `subject_name` — the biographical subject
- `age_during` — subject's age range during the period (e.g. "22–24")
- `location` — geographic location during the period
- `catalog_number` — unique tape catalog identifier

All nine fields SHALL be `Optional[str]` with `None` as default so that existing
construction sites that omit them do not error.

#### Scenario: All fields populated
- **WHEN** `QuantumTapeLabelArtifact` is constructed with all nine template fields set
- **THEN** `save_file()` writes HTML where none of the nine content slots are empty strings

#### Scenario: Fields omitted
- **WHEN** `QuantumTapeLabelArtifact` is constructed without the nine template fields
- **THEN** `save_file()` still succeeds without error (slots render as empty strings)

---

