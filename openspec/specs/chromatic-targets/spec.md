# chromatic-targets Specification

## Purpose
TBD - created by archiving change fix-chromatic-targets-canonical-source. Update Purpose after archive.
## Requirements
### Requirement: Canonical Chromatic Targets Module
The system SHALL provide a single module `app/structures/concepts/chromatic_targets.py`
that derives `CHROMATIC_TARGETS` at import time from `the_rainbow_table_colors` (the
authoritative Pydantic instances in `app/structures/concepts/rainbow_table_color.py`).
No other file in the codebase SHALL define its own copy of these probability vectors.

The module SHALL also export `TEMPORAL_MODES`, `SPATIAL_MODES`, and `ONTOLOGICAL_MODES`
as tuples of mode label strings in the canonical vector ordering used throughout the pipeline:
- `TEMPORAL_MODES = ("past", "present", "future")`
- `SPATIAL_MODES = ("thing", "place", "person")`
- `ONTOLOGICAL_MODES = ("imagined", "forgotten", "known")`

#### Scenario: Vectors sum to their canonical totals
- **WHEN** `CHROMATIC_TARGETS` is imported
- **THEN** each vector for every color sums to 1.0 (±1e-6),
  except `CHROMATIC_TARGETS["Indigo"]["ontological"]`, which sums to 0.9 (±1e-6)
  (two-mode soft-label: KNOWN and FORGOTTEN each 0.4, IMAGINED 0.1)

#### Scenario: Correct values for canonical colors
- **WHEN** the module is imported
- **THEN** `CHROMATIC_TARGETS["Red"]["temporal"]` equals `[0.8, 0.1, 0.1]`,
  `CHROMATIC_TARGETS["Green"]["temporal"]` equals `[0.1, 0.1, 0.8]`,
  and `CHROMATIC_TARGETS["Violet"]["ontological"]` equals `[0.1, 0.1, 0.8]`

#### Scenario: Indigo two-mode ontological
- **WHEN** the module is imported
- **THEN** `CHROMATIC_TARGETS["Indigo"]["ontological"]` equals `[0.1, 0.4, 0.4]`
  (split evenly between KNOWN and FORGOTTEN, IMAGINED suppressed to 0.1)

#### Scenario: No duplicate full-target triples among non-uniform chromatic colors
- **WHEN** comparing combined (temporal, spatial, ontological) tuples for colors
  that are not fully uniform (i.e. excluding White and Black, which are both uniform by design)
- **THEN** no two non-uniform colors share an identical triple
  (Yellow and Green MUST differ — this was the root cause of the CDM collapse)

#### Scenario: Pure Python import, no ML dependencies
- **WHEN** `chromatic_targets.py` is imported in an environment without torch or onnxruntime
- **THEN** the import succeeds and `CHROMATIC_TARGETS` is fully populated

