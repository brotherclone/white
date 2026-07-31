<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# Coding Conventions

## Import organisation

Imports must appear at the top of every file in exactly three blocks, separated by blank lines:

```python
import os          # 1. stdlib
import sys
from pathlib import Path

import yaml        # 2. third-party

from white_core.manifests.manifest import Manifest   # 3. first-party (white_* packages)
from white_generation.pipelines.chord_pipeline import run_chord_pipeline
```

- **Never put imports inside functions or methods**, except to break a genuine circular-import cycle. In that case add a `# circular import` comment so the reason is explicit.
- This is enforced by ruff (`I` rules / isort) in pre-commit — `ruff check --fix` will sort automatically.

## Prefer Pydantic for structured data

When a function returns a dict or JSON payload that has a defined shape — API responses, pipeline outputs, review entries, anything that flows between components — prefer a Pydantic model over a raw `dict`. Pydantic models live in `white_core/` (not `models/`, which is reserved for ML model definitions) and give both humans and Claude a self-documenting schema with validation and metadata.

This is a heuristic, not a hard rule. A one-off helper that returns two values doesn't need a model. But if the same shape appears in multiple places, or if it crosses a boundary (API response, YAML round-trip, pipeline stage handoff), a `structures/` Pydantic class is the right move.

## Prefer enums over string literals

When a value can only be one of a fixed set of options, use a Python `Enum` rather than a raw string. String matches are a typo waiting to happen — enums make invalid states unrepresentable and give autocomplete.

```python
# Instead of:
if repeat_type == "exact":   # silent bug

# Prefer:
class LyricRepeatType(str, Enum):
    EXACT = "exact"
    VARIATION = "variation"
    FRESH = "fresh"
```

Enums live in `white_core/enums/`. Use `str, Enum` (string-valued) so they serialise cleanly to/from YAML and JSON without extra conversion. When loading from external input (YAML, API), normalise to the enum early and let it be an enum everywhere inside the code.

## Versioning

This is a `uv` workspace: the root `pyproject.toml` and each `packages/*/pyproject.toml` carry their own independent `version`. Versions have gone stale (everything sitting at `0.1.0`) because bumping them was never made part of the routine — fix that going forward.

When preparing a PR, bump `version` in every `packages/*/pyproject.toml` whose package actually changed (source, not just its tests) — never bump a package that wasn't touched. Bump the root `pyproject.toml` version too when the change is workspace-wide (root-level config, cross-package refactors) rather than scoped to one package.

Default to a **minor** bump (`0.1.0` → `0.2.0`) for anything that adds or changes behavior — new pipeline phases, new fields, new CLI flags, changed defaults. All packages are pre-1.0, so under semver a breaking change still bumps minor (not major) at this stage — reserve a major bump for when a package is promoted to a stable 1.0 API. Use a **patch** bump (`0.1.0` → `0.1.1`) only for narrowly-scoped fixes: a bug fix with no behavior change beyond "it now works," a dependency version bump, docs/comments-only changes. Test-only or CI-only changes don't need a version bump at all.

Do this as a normal part of finishing the PR, without asking for confirmation each time — the version bump is a plain edit to a version string, not a risky action.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
