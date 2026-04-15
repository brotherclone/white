# Change: Fix CHROMATIC_TARGETS — derive from canonical Pydantic source of truth

## Why

`CHROMATIC_TARGETS` — the per-color probability vectors used for chromatic scoring, drift
reporting, and CDM inference — are hardcoded in at least five files and are wrong for 7 of 9
colors. The single source of truth is `app/structures/concepts/rainbow_table_color.py`
(`the_rainbow_table_colors`), but no code derives from it: every file hand-rolled its own
copy and they diverged from each other and from the Pydantic model.

Immediate pipeline impact: chord, bass, drum, and melody candidates for Orange, Yellow,
Green, Blue, Indigo, Violet, and Black albums have all been scored and pruned against
incorrect chromatic targets. Drift reports and chromatic match scores for those colors
are meaningless until this is corrected.

## What the correct targets are

Mode vector ordering (established in pipeline): temporal `[past, present, future]`,
spatial `[thing, place, person]`, ontological `[imagined, forgotten, known]`.

Derived from `the_rainbow_table_colors`:

| Color  | Temporal          | Spatial           | Ontological       |
|--------|-------------------|-------------------|-------------------|
| Red    | [0.8, 0.1, 0.1] ✓ | [0.8, 0.1, 0.1] ✓ | [0.1, 0.1, 0.8] ✓ |
| Orange | [0.8, 0.1, 0.1]   | [0.8, 0.1, 0.1]   | [0.8, 0.1, 0.1]   |
| Yellow | [0.1, 0.1, 0.8]   | [0.1, 0.8, 0.1]   | [0.8, 0.1, 0.1]   |
| Green  | [0.1, 0.1, 0.8]   | [0.1, 0.8, 0.1]   | [0.1, 0.8, 0.1]   |
| Blue   | [0.1, 0.8, 0.1]   | [0.1, 0.1, 0.8]   | [0.1, 0.8, 0.1]   |
| Indigo | [1/3, 1/3, 1/3]   | [1/3, 1/3, 1/3]   | [0.1, 0.4, 0.4]   |
| Violet | [0.1, 0.8, 0.1]   | [0.1, 0.1, 0.8]   | [0.1, 0.1, 0.8]   |
| White  | [1/3, 1/3, 1/3] ✓ | [1/3, 1/3, 1/3] ✓ | [1/3, 1/3, 1/3] ✓ |
| Black  | [1/3, 1/3, 1/3]   | [1/3, 1/3, 1/3]   | [1/3, 1/3, 1/3]   |

(✓ = already correct everywhere)

## Current state of each file

| File | Errors |
|------|--------|
| `app/generators/midi/pipelines/chord_pipeline.py` | Orange spatial, Yellow temporal+ontological, Green temporal, Blue temporal+ontological, Indigo all, Violet ontological |
| `training/refractor.py` (`_CDM_CHROMATIC_TARGETS`) | Orange temporal+ontological, Yellow temporal+ontological, Green temporal+ontological, Blue temporal+spatial, Indigo all, Violet temporal+ontological, Black all |
| `training/validate_mix_scoring.py` | Same errors as `refractor.py` |
| `training/modal_train_refractor_cdm.py` | Same errors as `refractor.py` |
| `app/generators/midi/production/score_mix.py` | (uses `compute_chromatic_match` which sources from refractor — needs audit) |

## What Changes

- **NEW: `app/structures/concepts/chromatic_targets.py`** — single module that derives
  `CHROMATIC_TARGETS` from `the_rainbow_table_colors` at import time; exported as a typed
  dict. All other files import from here; no hardcoded copies remain.
- **UPDATED: all five files above** — replace hardcoded dicts with import from
  `chromatic_targets.py`
- **UPDATED: `training/refractor.py`** — `_CDM_CHROMATIC_TARGETS` becomes an alias for
  the shared dict; mode-ordering constants (`TEMPORAL_MODES`, `SPATIAL_MODES`,
  `ONTOLOGICAL_MODES`) moved to `chromatic_targets.py` and re-exported
- **NEW: tests** — `tests/structures/test_chromatic_targets.py` asserts the computed
  vectors match the Pydantic model field values for every color; asserts vectors sum to 1.0;
  asserts no duplicate target vectors exist across colors
- **CDM retraining required** — the CDM ONNX inference applies `_CDM_CHROMATIC_TARGETS`
  at prediction time; once corrected, the loaded `refractor_cdm.onnx` will produce
  different (correct) score distributions. Retraining is not required (the CDM learns
  integer class labels, not targets), but `validate_mix_scoring.py` should be re-run to
  confirm accuracy is not degraded. The HF model card `refractor_cdm` should be updated.

## Impact

- Affected specs: `audio-mix-scoring`, `chord-generation`, `bass-generation`,
  `melody-generation`, `drum-generation` (all use CHROMATIC_TARGETS for scoring)
- Affected code: `app/generators/midi/pipelines/chord_pipeline.py`,
  `app/generators/midi/production/score_mix.py`, `training/refractor.py`,
  `training/validate_mix_scoring.py`, `training/modal_train_refractor_cdm.py`
- **Non-breaking for API callers**: dict keys (`temporal`/`spatial`/`ontological`) and
  mode label strings are unchanged; only the probability values change
- **Breaking for existing scored artifacts**: any `mix_score.yml`, `review.yml`, or
  drift report produced before this fix contains wrong chromatic_match and drift values
  for non-Red/non-White songs. Re-scoring is recommended before production decisions.
