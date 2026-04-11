## MODIFIED Requirements
### Requirement: Production Plan Schema
The `PlanSection` dataclass SHALL include an `arc` field: a float 0.0–1.0
representing intended emotional intensity for that section (0.0 = near-silence,
1.0 = peak). The field SHALL default to 0.0 and be serialised as a plain float
in YAML. It SHALL be populated by `_infer_arc_from_label` when `generate_plan`
creates sections, and preserved by `refresh_plan` (human overrides survive refresh).

`_infer_arc_from_label(label: str) → float` SHALL return:
- `intro`, `outro` → 0.15
- `verse` → 0.35
- `pre_chorus` → 0.55
- `chorus`, `refrain`, `hook` → 0.75
- `bridge` → 0.20
- `climax`, `peak` → 0.90
- anything else → 0.40

#### Scenario: Arc field round-trips through YAML
- **WHEN** a plan is saved and reloaded
- **THEN** `arc` values are preserved as floats

#### Scenario: Arc auto-seeded from label
- **WHEN** `generate_plan` creates sections
- **THEN** chorus sections have arc > 0.6 and bridge sections have arc < 0.3

#### Scenario: Human override survives refresh
- **WHEN** a human sets arc=0.9 on a verse and `refresh_plan` is called
- **THEN** the verse arc remains 0.9 after refresh
