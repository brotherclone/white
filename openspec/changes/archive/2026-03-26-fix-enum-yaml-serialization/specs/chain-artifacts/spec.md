## ADDED Requirements

### Requirement: Chain Artifact YAML Serialization
Chain artifact `save_file()` implementations that emit YAML SHALL produce output that is
readable by `yaml.safe_load()` without Python-specific tags. Enum fields MUST be serialised
as their string values (e.g. `"yml"`, `"newspaper_article"`), not as
`!!python/object/apply:app.structures.enums.*` decorated objects.

This is achieved by using `model_dump(mode="json")` instead of `model_dump(mode="python")`
when constructing the dict passed to `yaml.dump()`.

#### Scenario: Enum field serialization — clean value
- **WHEN** any YML-emitting artifact calls `save_file()`
- **THEN** enum fields in the output file contain only the enum's string value (e.g. `chain_artifact_type: newspaper_article`)
- **AND** the file contains no `!!python/object` or `!!python/object/apply` tags

#### Scenario: Round-trip safety
- **WHEN** a chain artifact YAML file is read back with `yaml.safe_load()`
- **THEN** it loads successfully without a `yaml.constructor.ConstructorError`

#### Scenario: Value unchanged
- **WHEN** a chain artifact YAML file is written with the fixed serializer
- **THEN** the enum's human-readable value (e.g. `"yml"`, `"symbolic_object"`, `"circular_time"`) is preserved unchanged
