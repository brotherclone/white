## 1. Fix call sites

- [x] 1.1 `alternate_timeline_artifact.py:112` — change `mode="python"` to `mode="json"`
- [x] 1.2 `arbitrarys_survey_artifact.py:93` — change `mode="python"` to `mode="json"`
- [x] 1.3 `last_human_artifact.py:116` — change `mode="python"` to `mode="json"`
- [x] 1.4 `newspaper_artifact.py:101` — change `mode="python"` to `mode="json"`
- [x] 1.5 `reaction_book_artifact.py:74` — change `mode="python"` to `mode="json"`
- [x] 1.6 `rescue_decision_artifact.py:87` — change `mode="python"` to `mode="json"`
- [x] 1.7 `species_extinction_artifact.py:136` — change `mode="python"` to `mode="json"`
- [x] 1.8 `symbolic_object_artifact.py:45` — change `mode="python"` to `mode="json"`

## 2. Verify

- [ ] 2.1 For at least one artifact type (e.g. `SymbolicObjectArtifact`), instantiate and call `save_file()` in a scratch script or test, then confirm the output YAML contains no `!!python/object` tags and can be round-tripped with `yaml.safe_load()`.
- [ ] 2.2 Confirm `yaml.safe_load` on an existing broken file still loads (it uses `yaml.safe_load`, which rejects `!!python/object` — confirm the fix makes new files loadable).
