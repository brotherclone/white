## 1. Production Plan Schema
- [x] 1.1 Add `arc: float = 0.0` field to `PlanSection` dataclass
- [x] 1.2 Add `_infer_arc_from_label(label: str) → float` helper
- [x] 1.3 Populate `arc` in `generate_plan()` via `_infer_arc_from_label`
- [x] 1.4 Preserve `arc` in `sync_plan_from_arrangement` (human overrides survive refresh)
- [x] 1.5 Serialise `arc` as plain float in YAML save/load round-trip
- [x] 1.6 Tests: arc round-trip, label inference, human override survives refresh

## 2. Arc-Aware Energy in Pipelines
- [x] 2.1 Add `arc_to_energy(arc: float) → str` helper: <0.3 → low, 0.3–0.65 → medium, >0.65 → high
- [x] 2.2 In `drum_pipeline.py`: load arc from production_plan.yml; use `arc_to_energy` to override target_energy when arc is present
- [x] 2.3 In `bass_pipeline.py`: load arc and apply arc-based tag adjustment (low arc → +0.1 drone/pedal; high arc → −0.05 root_drone)
- [x] 2.4 In `melody_pipeline.py`: load arc and apply arc-based tag adjustment (low arc → +0.1 lamentful/sparse; high arc → +0.1 dense/arpeggiated)

## 3. Arc Helpers Module
- [x] 3.1 Add `arc_to_energy` and `arc_tag_adjustment` to `aesthetic_hints.py`

## 4. Tests
- [x] 4.1 Test `_infer_arc_from_label` for all canonical labels
- [x] 4.2 Test drum pipeline uses arc to override energy target
- [x] 4.3 Test bass pipeline arc adjustments
- [x] 4.4 Test melody pipeline arc adjustments
