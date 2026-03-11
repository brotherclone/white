## ADDED Requirements

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

### Requirement: Blue Agent Tape Label Population
`blue_agent.generate_tape_label()` SHALL populate all nine template fields by mapping
data from the `AlternateTimelineArtifact` and its nested `BiographicalPeriod`.

| Template field    | Source                                                           |
|-------------------|------------------------------------------------------------------|
| `year_documented` | `alternate.period.start_date.year`                               |
| `original_date`   | `str(alternate.period.start_date.year)`                          |
| `original_title`  | `f"Gabe Walsh — {alternate.period.start_date.year}"` (A-side label) |
| `tapeover_date`   | Formatted period range e.g. `"Jun 1998 – Nov 1999"`             |
| `tapeover_title`  | `alternate.title`                                                |
| `subject_name`    | `"Gabe Walsh"` (constant for blue agent)                        |
| `age_during`      | `f"{alternate.period.age_range[0]}–{alternate.period.age_range[1]}"` |
| `location`        | `alternate.period.location` → fallback `"Unknown"`              |
| `catalog_number`  | `f"QT-B-{year}-{thread_id[:6].upper()}"`                       |

The existing `title`, `date_range`, and `original_label_text` arguments SHALL be kept for
backwards-compatibility with `flatten()` and `for_prompt()`.

#### Scenario: Blue agent generates tape label from alternate timeline
- **WHEN** `generate_tape_label()` runs with a valid `AlternateTimelineArtifact` in state
- **THEN** the resulting `QuantumTapeLabelArtifact` has all nine template fields populated
- **AND** the saved HTML file contains the `alternate.title` in the tapeover slot
- **AND** the saved HTML file contains `"Gabe Walsh"` in the subject slot
- **AND** `file_name` does not contain `UNKNOWN_ARTIFACT_NAME`

#### Scenario: Period has no location
- **WHEN** `alternate.period.location` is `None`
- **THEN** `location` field is set to `"Unknown"`

---

### Requirement: Flatten Completeness
`QuantumTapeLabelArtifact.flatten()` SHALL include all nine template fields in its return
dict so downstream consumers (shrinkwrap, evaluator) can read the tape metadata.

#### Scenario: flatten includes template fields
- **WHEN** `flatten()` is called on a fully populated artifact
- **THEN** the returned dict contains all nine template field keys with their values
