## 1. Canonical module

- [ ] 1.1 Create `app/structures/concepts/chromatic_targets.py`:
      derive `CHROMATIC_TARGETS` at import time from `the_rainbow_table_colors`;
      export `TEMPORAL_MODES`, `SPATIAL_MODES`, `ONTOLOGICAL_MODES` tuples
- [ ] 1.2 Derivation logic (see design.md):
      single ontological mode → 0.8 at index, 0.1 elsewhere;
      two modes (Indigo) → 0.4 each, 0.1 for third;
      `None` → uniform `[1/3, 1/3, 1/3]`
- [ ] 1.3 Verify computed vectors match the table in proposal.md for all 9 colors

## 2. Deep audit & update — generation pipeline

- [ ] 2.1 `app/generators/midi/pipelines/chord_pipeline.py`: replace hardcoded
      `CHROMATIC_TARGETS` with import from `chromatic_targets.py`; confirm
      `compute_chromatic_match` uses mode labels that match `TEMPORAL_MODES` etc.
- [ ] 2.2 Grep entire `app/generators/midi/` for any other hardcoded color target dicts
      or inline `[0.8, 0.1, 0.1]`-style vectors — update any found
- [ ] 2.3 Verify `bass_pipeline.py`, `drum_pipeline.py`, `melody_pipeline.py` all
      source `CHROMATIC_TARGETS` from one place (currently via chord_pipeline import
      or direct copy — confirm and fix)

## 3. Deep audit & update — training / scoring

- [ ] 3.1 `training/refractor.py`: replace `_CDM_CHROMATIC_TARGETS` with alias import;
      move `TEMPORAL_MODES`, `SPATIAL_MODES`, `ONTOLOGICAL_MODES` constants to
      `chromatic_targets.py` and re-import in `refractor.py`
- [ ] 3.2 `training/validate_mix_scoring.py`: replace hardcoded `CHROMATIC_TARGETS`;
      confirm `_top1_color()` mode label ordering matches `TEMPORAL_MODES` etc.
- [ ] 3.3 `training/modal_train_refractor_cdm.py`: replace hardcoded `CHROMATIC_TARGETS`
- [ ] 3.4 `app/generators/midi/production/score_mix.py`: audit `compute_chromatic_match`
      and `write_mix_score` — confirm they use the dict keys from `TEMPORAL_MODES` etc.
      and not any hardcoded strings

## 4. Tests

- [ ] 4.1 Create `tests/structures/test_chromatic_targets.py`:
      - each color's derived vectors sum to 1.0 (±1e-6)
      - each vector has exactly 3 elements, all in [0, 1]
      - Red, White, Black match known-correct values (regression guard)
      - Indigo ontological = [0.1, 0.4, 0.4] (two-mode case)
      - no two non-transmigrational colors share identical target triples
        (catches the Yellow==Green collision that originally masked the CDM bug)
- [ ] 4.2 Add import-level smoke test: `from app.structures.concepts.chromatic_targets
      import CHROMATIC_TARGETS` succeeds without torch/onnx (pure Python dep only)

## 5. CDM re-validation

- [ ] 5.1 Re-run `python training/validate_mix_scoring.py` with corrected targets;
      record new per-color accuracy table
- [ ] 5.2 Update HF model card at `earthlyframes/refractor_cdm` with corrected color
      descriptions and new validation numbers
- [ ] 5.3 If overall accuracy drops below 70%: retrain CDM
      (`python training/extract_cdm_embeddings.py` + `modal run modal_train_refractor_cdm.py`)

## 6. Re-score affected artifacts

- [ ] 6.1 Identify any approved/promoted `review.yml` files for non-Red/non-White songs
      (chord, drum, bass, melody) — these have stale `chromatic_match` scores
- [ ] 6.2 Re-run scoring for any song that has been through the full production pipeline;
      check whether previously approved candidates still rank at the top with correct targets
- [ ] 6.3 For the base Refractor ONNX (`refractor.onnx`): **no retraining required** —
      `modal_midi_fusion.py` trains from per-segment mode labels in the HF dataset, never
      from `CHROMATIC_TARGETS`; the weights are correct as-is

## 7. Cleanup

- [ ] 7.1 Run full test suite; confirm 0 new regressions
