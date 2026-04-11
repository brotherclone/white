## 1. Pydantic Models
- [x] 1.1 Create `app/structures/music/narrative_constraints.py` with `NarrativeSection` and `CompositionNarrative` Pydantic models
- [x] 1.2 Implement `extract_constraints(section_name, narrative) → dict` helper

## 2. Composition Narrative Module
- [x] 2.1 Create `app/generators/midi/production/composition_narrative.py`
- [x] 2.2 Implement `generate_narrative(production_dir) → Optional[Path]` — calls Claude API
- [x] 2.3 Implement `load_narrative(production_dir) → Optional[CompositionNarrative]`
- [x] 2.4 Implement `narrative_tag_adjustment(section, pattern_tags) → float`

## 3. Pipeline Integration
- [x] 3.1 In `drum_pipeline.py`: load narrative; apply rhythm_character tag adjustment
- [x] 3.2 In `bass_pipeline.py`: load narrative; apply texture + lead_voice tag adjustment
- [x] 3.3 In `melody_pipeline.py`: load narrative; apply register + lead_voice adjustment; skip section when `lead_voice: none`

## 4. Tests
- [x] 4.1 Test NarrativeSection validates controlled vocabulary
- [x] 4.2 Test extract_constraints returns correct hint dicts
- [x] 4.3 Test narrative_tag_adjustment produces expected score deltas
- [x] 4.4 Test melody pipeline skips section when lead_voice=none
