## ADDED Requirements

### Requirement: Composition Narrative Generation
`composition_proposal.py` SHALL expose a `generate_narrative(production_dir)` function
that calls the Claude API and writes `composition_narrative.yml` to the production
directory. The narrative describes each section along four controlled-vocabulary
dimensions: register, texture, harmonic_complexity, and rhythm_character. It also
designates `the_moment` (the emotional peak section) and provides free-text narrative
for each section.

#### Scenario: Narrative written with controlled vocabulary values
- **WHEN** `generate_narrative()` is called for a production directory with a
  production plan and song context
- **THEN** `composition_narrative.yml` is written with section entries containing
  valid values for register, texture, harmonic_complexity, and rhythm_character
- **AND** each section entry contains a `narrative` free-text field

#### Scenario: ACE Studio unreachable does not block generation
- **WHEN** the Claude API is unavailable
- **THEN** a warning is logged and None is returned (the narrative file is not written)

---

### Requirement: Narrative Constraints Module
`app/structures/music/narrative_constraints.py` SHALL provide a `NarrativeSection`
Pydantic model describing per-section generation constraints using controlled
vocabularies, and a `CompositionNarrative` top-level model holding the full document.

#### Scenario: Section constraints load from YAML
- **WHEN** a section entry is parsed from `composition_narrative.yml`
- **THEN** invalid vocabulary values raise a Pydantic validation error
- **AND** missing optional fields default to None

#### Scenario: Constraint extraction returns generation hints
- **WHEN** `extract_constraints(section_name, narrative)` is called
- **THEN** a dict of generation hints is returned (e.g. rhythm_character → target tag list)
