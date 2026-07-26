# ideation-sounds-like Specification

## Purpose
TBD - created by archiving change add-ideation-sounds-like. Update Purpose after archive.
## Requirements
### Requirement: Reference Proposals Carry Sounds-Like Data
Each color agent (Black, Red, Orange, Yellow, Green, Violet) SHALL call the
existing `get_sounds_like_by_color(color_character)` and
`sample_reference_artists(artists)` helpers (already implemented in
`manifest_loader.py`, previously only called by Blue Agent) and include the
sampled artist names in its "reference works in this artist's style" prompt
section, alongside the existing `get_my_reference_proposals()` output.

#### Scenario: Color has sounds_like entries
- **WHEN** a color agent builds its reference-works prompt section
- **AND** `get_sounds_like_by_color()` for that color returns at least one
  artist name
- **THEN** the prompt text includes a sampled subset of those artist names

#### Scenario: Color has no sounds_like entries
- **WHEN** `get_sounds_like_by_color()` for that color returns an empty list
- **THEN** the prompt is built without a sounds-like line, with no error

### Requirement: Blue Agent Surfaces Its Own Sounds-Like Lookup
Blue Agent's `generate_alternate_song_spec` SHALL include
`state.musical_params.reference_artists` (already populated by
`extract_musical_parameters` via `get_sounds_like_by_color("B")` and
`sample_reference_artists`) in the counter-proposal prompt.

#### Scenario: Reference artists were sampled
- **WHEN** `state.musical_params.reference_artists` is a non-empty list at the
  time `generate_alternate_song_spec` builds its prompt
- **THEN** those artist names appear in the prompt text sent to the LLM

#### Scenario: No reference artists found
- **WHEN** `state.musical_params.reference_artists` is empty (no Blue-color
  manifests had `sounds_like` entries)
- **THEN** the prompt is built without a reference-artists line, with no
  error or warning beyond what already exists

