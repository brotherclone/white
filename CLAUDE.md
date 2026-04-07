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
import os          # 1. bare stdlib/third-party imports
import sys

from pathlib import Path          # 2. from third-party import
import yaml

from app.generators.midi import x   # 3. from local/app import
from training.refractor import y
```

- **Never put imports inside functions or methods**, except to break a genuine circular-import cycle. In that case add a `# circular import` comment so the reason is explicit.
- This is enforced by ruff (`I` rules / isort) in pre-commit — `ruff check --fix` will sort automatically.

## Prefer Pydantic for structured data

When a function returns a dict or JSON payload that has a defined shape — API responses, pipeline outputs, review entries, anything that flows between components — prefer a Pydantic model over a raw `dict`. Pydantic models live in `app/structures/` (not `models/`, which is reserved for ML model definitions) and give both humans and Claude a self-documenting schema with validation and metadata.

This is a heuristic, not a hard rule. A one-off helper that returns two values doesn't need a model. But if the same shape appears in multiple places, or if it crosses a boundary (API response, YAML round-trip, pipeline stage handoff), a `structures/` Pydantic class is the right move.