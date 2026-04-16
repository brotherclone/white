## 1. Data model defaults

- [x] 1.1 In `app/generators/midi/production/production_plan.py`, change
      `ProductionPlan.vocals_planned: bool = False` to `bool = True`
- [x] 1.2 Change `PlanSection.vocals: bool = False` to `bool = True`

## 2. Instrumental section inference

- [x] 2.1 In `generate_plan()`, after creating each `PlanSection`, set
      `section.vocals = False` when the label matches any of:
      `intro`, `outro`, `instrumental`, `solo`, `interlude`, `break`
      (case-insensitive, substring match on label)

## 3. Tests

- [x] 3.1 Update any tests that assert `vocals_planned=False` or `vocals=False`
      on freshly generated plans
- [x] 3.2 Add a test: `generate_plan` with a mix of section labels — chorus gets
      `vocals=True`, outro gets `vocals=False`
