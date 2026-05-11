# Change: Production Plan Drift Report

## Why

Claude proposes a song arrangement in `production_plan.yml` — section order, repeat counts,
energy arc — and the human then arranges in Logic. We never capture how far the final
arrangement diverged from that proposal. That gap is a calibration signal: if Claude
consistently proposes intros that always get cut, future proposals should stop including them.

## What Changes

- **New module**: `white_composition/drift_report.py` — `compare_plans()`, `write_report()`,
  `load_report()`, CLI `python -m white_composition.drift_report`
- **New artifact**: `plan_drift_report.yml` written to the production directory
- **New spec**: `plan-drift-report` capability

## Impact

- Affected specs: `plan-drift-report` (new)
- Affected code: `packages/composition/src/white_composition/drift_report.py` (new),
  `packages/composition/pyproject.toml` (add pydantic dep)

## Design Decisions

**Section name mapping**: `parse_arrangement_sections()` in `production_plan.py` already
reads track-1 clip names from `arrangement.txt`. Clip names in Logic match promoted MIDI
labels (the promote step uses labels as filenames). No separate mapping file needed.

**Report location**: production directory alongside other pipeline artifacts — not the
Logic song folder. Lives in `shrink_wrapped/`, can be committed.

**Trigger**: CLI only (`--production-dir`). Board button is a future addition.

**Naming**: `plan_drift_report.yml` — not `drift_report.yml`, which the assembly-manifest
spec uses for timestamp drift. The prefix avoids confusion.

**Future aggregation**: Aggregating drift signals across N songs to tune the production plan
prompt is out of scope; this change only writes the per-song report.
