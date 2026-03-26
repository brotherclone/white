# Change: Fix Python enum tags in chain artifact YAML output

## Why

Eight artifact `save_file()` methods call `yaml.dump(self.model_dump(mode="python"), ...)`.
`mode="python"` returns live Python enum instances; PyYAML's default Dumper has no representer
for them and falls back to the verbose `!!python/object/apply:app.structures.enums.xxx.ClassName`
tag format. This pollutes YAML outputs with Python internals, breaks `yaml.safe_load()` round-trips,
and makes files harder to read and diff.

Example of broken output:
```yaml
chain_artifact_file_type: !!python/object/apply:app.structures.enums.chain_artifact_file_type.ChainArtifactFileType
- yml
chain_artifact_type: !!python/object/apply:app.structures.enums.chain_artifact_type.ChainArtifactType
- newspaper_article
```

Expected clean output:
```yaml
chain_artifact_file_type: yml
chain_artifact_type: newspaper_article
```

## What Changes

- Switch the 8 affected `save_file()` calls from `model_dump(mode="python")` to `model_dump(mode="json")`.
  Pydantic's `mode="json"` serialises all enum fields to their `.value` strings automatically,
  producing clean, `safe_load`-compatible YAML with no Python tags.
- No shared utility required — the fix is a one-word change per call site.

## Impact

- Affected specs: `chain-artifacts`
- Affected code (8 files):
  - `app/structures/artifacts/alternate_timeline_artifact.py:112`
  - `app/structures/artifacts/arbitrarys_survey_artifact.py:93`
  - `app/structures/artifacts/last_human_artifact.py:116`
  - `app/structures/artifacts/newspaper_artifact.py:101`
  - `app/structures/artifacts/reaction_book_artifact.py:74`
  - `app/structures/artifacts/rescue_decision_artifact.py:87`
  - `app/structures/artifacts/species_extinction_artifact.py:136`
  - `app/structures/artifacts/symbolic_object_artifact.py:45`
- `evp_artifact.py` and `sigil_artifact.py` use `yaml.dump` but not `model_dump(mode="python")` — no change needed.
- No breaking changes to downstream consumers: enum values (e.g. `"yml"`, `"newspaper_article"`) are unchanged; only the YAML tag decoration is removed.
- Existing chain artifact YMLs already in `chain_artifacts/` are not retroactively fixed (they stay as-is).
