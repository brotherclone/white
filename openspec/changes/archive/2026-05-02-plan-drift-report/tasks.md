## 1. Setup

- [x] 1.1 Add `pydantic>=2.0` to `packages/composition/pyproject.toml` dependencies
- [x] 1.2 Create `openspec/changes/plan-drift-report/specs/plan-drift-report/spec.md`

## 2. Core Implementation

- [x] 2.1 Create `packages/composition/src/white_composition/drift_report.py`
  - `BarDelta`, `DriftSummary`, `DriftReport` Pydantic models
  - `_normalize_label()` — strip version suffixes and normalise to snake_case
  - `_pearson_r()` — manual Pearson r computation (no scipy dependency)
  - `_arc_correlation()` — normalise sequences to 100 points, compute Pearson r
  - `_expand_proposed()` — expand plan sections by play_count into instance list
  - `_build_drift_summary()` — removed/added/reordered from proposed + actual label sets
  - `_build_bar_deltas()` — per-label total bars proposed vs actual
  - `_generate_summary()` — call Claude for prose summary; returns "" on failure
  - `compare_plans(plan, arrangement_path)` → `DriftReport`
  - `write_report(production_dir, report)` → `Path`
  - `load_report(production_dir)` → `DriftReport | None`
  - `main()` CLI entry point

## 3. Tests

- [x] 3.1 Create `packages/composition/tests/test_drift_report.py`
  - `test_normalize_label` — suffix stripping
  - `test_pearson_r_perfect_positive` / `_negative` / `_constant_returns_none`
  - `test_arc_correlation_same_length` / `_different_length` / `_too_short`
  - `test_expand_proposed` — play_count expansion
  - `test_drift_summary_removed` / `_added` / `_reordered` / `_same_order`
  - `test_bar_deltas_exact` / `_delta` / `_missing_actual`
  - `test_compare_plans_full` — end-to-end with real ProductionPlan + arrangement.txt fixture
  - `test_write_and_load_report` — round-trip YAML
  - `test_load_report_missing_returns_none`
  - `test_cli_missing_plan` / `_missing_arrangement`
  - `test_cli_no_claude_skips_api`

## 4. Validation

- [x] 4.1 Run `openspec validate plan-drift-report --strict`
- [x] 4.2 Run `pytest packages/composition/tests/test_drift_report.py -v`
