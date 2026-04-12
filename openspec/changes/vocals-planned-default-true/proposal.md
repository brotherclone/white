# Change: Default vocals_planned to True in production plan

## Why

Every White song has a singer and lyrics. `vocals_planned: false` is the wrong
default — it forces a manual correction every time a production plan is created
and has caused assembly manifests to incorrectly omit vocal metadata. The field
should start `True` and only be set `False` for intentional instrumental productions.

## What Changes

- `ProductionPlan.vocals_planned` default changes from `False` to `True`
- `PlanSection.vocals` default changes from `False` to `True` — new sections
  inferred by `generate_plan` are assumed to carry vocals unless the section
  label is one of the conventionally instrumental types: `intro`, `outro`,
  `instrumental`, `solo`, `interlude`, `break`

## Impact

- Affected code: `app/generators/midi/production/production_plan.py`
- Affected spec: `production-plan`
- Existing `production_plan.yml` files on disk are unaffected (they have explicit
  values already)
- `refresh_plan` preserves all human-edited values, so this only affects
  freshly generated plans
