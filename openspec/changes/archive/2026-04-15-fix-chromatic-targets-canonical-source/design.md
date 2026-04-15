## Context

`CHROMATIC_TARGETS` maps each rainbow color to three probability vectors — temporal
(past/present/future), spatial (thing/place/person), ontological (imagined/forgotten/known)
— and is the central signal used by every scoring function in the pipeline. It was
hand-rolled in multiple files and has diverged from the canonical Pydantic model in
`app/structures/concepts/rainbow_table_color.py`.

## Goals / Non-Goals

- **Goals**: single definition, derived from Pydantic source of truth, imported everywhere,
  tested against the model, no silent drift possible in future
- **Non-Goals**: changing the mode label strings or vector ordering (that's a separate,
  breaking refactor); retroactively re-scoring existing artifacts (recommended but manual)

## Decisions

### Decision: New module at `app/structures/concepts/chromatic_targets.py`

Lives in `app/structures/concepts/` alongside `rainbow_table_color.py`. Derives the dict
at import time by iterating `the_rainbow_table_colors`. Exports:
- `CHROMATIC_TARGETS: dict[str, dict[str, list[float]]]`
- `TEMPORAL_MODES: tuple[str, ...]` = `("past", "present", "future")`
- `SPATIAL_MODES: tuple[str, ...]` = `("thing", "place", "person")`
- `ONTOLOGICAL_MODES: tuple[str, ...]` = `("imagined", "forgotten", "known")`

**Why not inline in `rainbow_table_color.py`**: keeps the Pydantic model file focused on
data definition; the float-vector representation is a derived concern for the ML pipeline,
not the conceptual model itself.

**Derivation logic**:
- `temporal_mode`: `PAST→[0.8,0.1,0.1]`, `PRESENT→[0.1,0.8,0.1]`, `FUTURE→[0.1,0.1,0.8]`,
  `None→[1/3,1/3,1/3]`
- `objectional_mode`: `THING→[0.8,0.1,0.1]`, `PLACE→[0.1,0.8,0.1]`, `PERSON→[0.1,0.1,0.8]`,
  `None→[1/3,1/3,1/3]`
- `ontological_mode` (list): single mode → `0.8` at its index, `0.1` elsewhere;
  two modes (Indigo: Known+Forgotten) → `0.4` each, `0.1` for the third;
  `None→[1/3,1/3,1/3]`

### Decision: `training/` files import via `sys.path` insert, not package install

`training/` scripts run in Modal (remote) and locally without package install. They
currently use `sys.path.insert(0, str(repo_root))` or equivalent. The import will be:

```python
from app.structures.concepts.chromatic_targets import CHROMATIC_TARGETS
```

This is already the pattern used by `training/refractor.py` for other imports.

### Decision: `_CDM_CHROMATIC_TARGETS` in `refractor.py` becomes an alias

```python
from app.structures.concepts.chromatic_targets import CHROMATIC_TARGETS as _CDM_CHROMATIC_TARGETS
```

The name `_CDM_CHROMATIC_TARGETS` is kept for minimal diff; callers inside `refractor.py`
don't change.

### Decision: Mode ordering constants are authoritative in `chromatic_targets.py`

`TEMPORAL_MODES`, `SPATIAL_MODES`, `ONTOLOGICAL_MODES` move there. `refractor.py`
re-exports them for backward compatibility.

## Risks / Trade-offs

- **Existing scored artifacts are stale** — any review.yml or mix_score.yml produced
  before this fix has wrong chromatic_match values for 7 of 9 colors. No automated
  migration; the fix recommendation is to re-run scoring pipelines.
- **CDM validation will change** — accuracy numbers in the HF model card will shift
  because `_top1_color()` in `validate_mix_scoring.py` uses `CHROMATIC_TARGETS` to
  pick the predicted color. Could go up or down; requires a re-run to confirm.
- **No retraining needed for CDM weights** — CDM is a discriminative classifier trained
  on integer class labels; the weights don't encode target vectors. Only inference-time
  mapping changes.

## Open Questions

- Should Indigo's temporal and spatial axes use `[1/3, 1/3, 1/3]` or something that
  reflects its "ghost color" nature differently? Current derivation is uniform because
  `temporal_mode` and `objectional_mode` are `None` in the Pydantic model.
- Black and White both derive to fully uniform vectors. Is this intentional, or should
  Black's transmigrational mode (SPACE→TIME→INFORMATION) encode a directional bias?
  Leaving as uniform until the user specifies otherwise.
