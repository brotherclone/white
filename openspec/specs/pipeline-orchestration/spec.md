# pipeline-orchestration Specification

## Purpose
TBD - created by archiving change spike-pipeline-orchestration-design. Update Purpose after archive.
## Requirements
### Requirement: Pipeline Data Flow Map
The spike SHALL produce a complete map of cross-phase data dependencies, showing which
files each pipeline phase reads and writes and which fields flow between phases.

#### Scenario: Dependency graph produced
- **WHEN** the spike is complete
- **THEN** `design_report.md` contains a dependency graph (ASCII or Mermaid) covering
  all pipeline phases from chord generation through mix scoring

#### Scenario: sounds_like and concept fully traced
- **WHEN** the spike traces sounds_like and concept through the pipeline
- **THEN** every function that reads or writes either field is documented, including
  cases where the field is silently zeroed out or overwritten

---

### Requirement: load_song_proposal Consolidation Assessment
The spike SHALL audit all implementations of song proposal loading and produce a
recommendation on whether and how to consolidate them.

#### Scenario: All implementations identified
- **WHEN** the audit is complete
- **THEN** `design_report.md` lists every load_song_proposal variant with its file
  location, returned field set, and callers

#### Scenario: Consolidation is feasible
- **WHEN** a single loader can satisfy all callers
- **THEN** the report proposes a canonical implementation location and migration path

#### Scenario: Consolidation is not worth it
- **WHEN** divergence between callers is fundamental and unification would require
  significant caller changes
- **THEN** the report documents why and recommends leaving implementations separate
  with shared utilities for common parsing

---

### Requirement: Production Context Schema Proposal
The spike SHALL define a proposed `song_context.yml` schema that could serve as the
canonical source of truth for song metadata across all pipeline phases.

#### Scenario: Schema covers all pipeline needs
- **WHEN** the proposed schema is evaluated against each pipeline phase's metadata needs
- **THEN** every field currently re-read from the song proposal YAML per phase is
  represented in the schema

#### Scenario: Backward compatible migration exists
- **WHEN** existing production directories (lacking song_context.yml) are evaluated
- **THEN** the report documents a migration path that reconstructs song_context.yml
  from existing files without re-running any MIDI generation phase

---

### Requirement: Pipeline Design Spike Report
The spike SHALL produce `training/spikes/pipeline-orchestration/design_report.md`
with findings, schema proposals, migration strategy, and a scoped recommendation for
a follow-on refactor.

#### Scenario: Report contains all required sections
- **WHEN** the spike is complete
- **THEN** `design_report.md` contains sections covering the data flow map,
  load_song_proposal audit, phase review schema comparison, proposed song_context.yml
  schema, migration strategy, and recommended refactor scope

#### Scenario: Scope recommendation is actionable
- **WHEN** the refactor scope is assessed
- **THEN** the recommendation breaks the work into sequenced increments (e.g. "phase 1:
  consolidate loaders; phase 2: add song_context.yml; phase 3: update pipeline phases")
  rather than a single big-bang rewrite, with an honest estimate of effort per phase

