## MODIFIED Requirements

### Requirement: Composite Scoring
The drum pipeline composite scoring SHALL incorporate narrative rhythm_character
constraints when `composition_narrative.yml` is present for the section.

The `rhythm_character` dimension SHALL influence candidate scoring:
- `absent`  → prefer `ambient_pulse` or silence; penalise patterns with snare
- `minimal` → prefer ghost patterns; penalise busy patterns
- `present` → no adjustment (baseline)
- `busy`    → prefer dense patterns; penalise sparse
- `open`    → prefer patterns with open hi-hat; penalise closed hi-hat dominant

The narrative constraint adjustment SHALL be applied after the arc-based energy
adjustment, as a tag-weighted score delta.

#### Scenario: Minimal rhythm_character penalises busy drum pattern
- **WHEN** section narrative declares `rhythm_character: minimal`
- **AND** a busy drum pattern and a ghost_verse pattern are candidates
- **THEN** the ghost_verse pattern scores higher than the busy pattern

#### Scenario: Absent rhythm_character prefers ambient pattern
- **WHEN** section narrative declares `rhythm_character: absent`
- **AND** an ambient pulse pattern and a dense electronic pattern are candidates
- **THEN** the ambient pulse pattern scores higher

#### Scenario: Missing narrative falls back to arc-based energy
- **WHEN** no `composition_narrative.yml` is present
- **THEN** drum scoring uses arc and label heuristics as before
