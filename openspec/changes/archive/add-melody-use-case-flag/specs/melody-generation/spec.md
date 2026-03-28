## ADDED Requirements

### Requirement: Melody Use-Case Annotation
The melody pipeline SHALL write a `use_case` field (`"vocal"` or `"instrumental"`) into
every candidate entry in `review.yml`, derived from the `MelodyPattern.use_case` attribute
of the winning template.

#### Scenario: Vocal template candidate
- **WHEN** a candidate is generated from a `MelodyPattern` with `use_case="vocal"`
- **THEN** its `review.yml` entry contains `use_case: vocal`

#### Scenario: Instrumental template candidate
- **WHEN** a candidate is generated from a `MelodyPattern` with `use_case="instrumental"`
- **THEN** its `review.yml` entry contains `use_case: instrumental`

#### Scenario: Promoted entry preserves use-case
- **WHEN** a candidate is approved and written to the promoted loop list
- **THEN** the promoted entry retains the `use_case` field
