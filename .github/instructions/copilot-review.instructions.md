---
applyTo: "**"
---

# White Project — Copilot Review Instructions

## Ignore these paths

- `openspec/changes/**` — These are transient change proposals, not canonical
  documentation. They are scaffolded, reviewed by a human, and then archived after
  deployment. Do not flag field names, wording, or schema mismatches in these files;
  they are intentionally provisional and will be deleted.

- `openspec/changes/archive/**` — Completed and archived changes. Read-only history.

## Project conventions

- This project uses OpenSpec for spec-driven development. Canonical specs live in
  `openspec/specs/`. Change proposals live in `openspec/changes/` and are transient.

- Pipeline phases (chords → drums → bass → melody) follow a consistent pattern:
  `*_pipeline.py` generates candidates, `promote_part.py` promotes approved ones,
  `review.yml` is the human approval gate. Do not suggest restructuring this pattern.

- Scoring always uses 30% theory + 70% chromatic weights. Do not suggest changing
  these weights unless there is a failing test.

- `generate_plan_mechanical()` is the intentionally kept legacy name for the
  non-Claude path. Do not suggest renaming it.
