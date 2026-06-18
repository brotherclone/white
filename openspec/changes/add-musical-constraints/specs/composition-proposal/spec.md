## ADDED Requirements

### Requirement: Musical Constraints Field in Song Proposal

The song proposal YAML schema SHALL support an optional `musical_constraints` block.
When present, it SHALL be parsed by `load_song_proposal_unified` and included in the
returned dict under the key `"musical_constraints"`.

A `MusicConstraints` Pydantic model in `white_core.structures` SHALL define the schema:

```python
class MusicConstraints(BaseModel):
    harmonic_sequence: str | None = None   # space-separated Roman numeral tokens
    performance_notes: str | None = None   # prose only; no pipeline effect
```

When `musical_constraints` is absent from the YAML, `load_song_proposal_unified` SHALL
return `"musical_constraints": None` (not raise).

#### Scenario: Proposal with harmonic_sequence

- **GIVEN** a song proposal YAML containing:
  ```yaml
  musical_constraints:
    harmonic_sequence: "i iv i"
  ```
- **WHEN** `load_song_proposal_unified(proposal_path)` is called
- **THEN** the returned dict SHALL include `musical_constraints` as a `MusicConstraints`
  instance with `harmonic_sequence == "i iv i"`

#### Scenario: Proposal without musical_constraints

- **GIVEN** a song proposal YAML with no `musical_constraints` key
- **WHEN** `load_song_proposal_unified(proposal_path)` is called
- **THEN** the returned dict SHALL include `"musical_constraints": None`
- **AND** no exception SHALL be raised

#### Scenario: Proposal with performance_notes only

- **GIVEN** a song proposal containing `musical_constraints.performance_notes` but no
  `harmonic_sequence`
- **WHEN** `load_song_proposal_unified(proposal_path)` is called
- **THEN** `musical_constraints.harmonic_sequence` SHALL be `None`
- **AND** `musical_constraints.performance_notes` SHALL contain the prose text

#### Scenario: Single-chord sequence

- **GIVEN** a song proposal with `harmonic_sequence: "i"` (one token)
- **WHEN** parsed
- **THEN** `harmonic_sequence` SHALL equal `"i"` and the pipeline SHALL treat this
  as a one-chord progression

### Requirement: performance_notes Surfaced in review.yml

When `musical_constraints.performance_notes` is non-null, the chord pipeline SHALL
write it as a top-level `performance_notes` field in `chords/review.yml` so the
human reviewer sees the agent's intent during the MIDI review stage.

#### Scenario: performance_notes appears in review.yml

- **GIVEN** a proposal with `musical_constraints.performance_notes: "Sustained tonic"`
- **WHEN** the chord pipeline generates `chords/review.yml`
- **THEN** the YAML SHALL contain a top-level key `performance_notes: "Sustained tonic"`

#### Scenario: No performance_notes — field absent from review.yml

- **GIVEN** a proposal with no `performance_notes`
- **THEN** `chords/review.yml` SHALL NOT contain a `performance_notes` key
