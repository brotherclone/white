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