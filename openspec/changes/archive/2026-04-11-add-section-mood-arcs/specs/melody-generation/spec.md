## MODIFIED Requirements
### Requirement: Composite Scoring
The melody pipeline SHALL read the section `arc` value from `production_plan.yml`
when available, and use it to adjust template selection: low arc (< 0.3) favours
sparse/lamentful-tagged templates; high arc (> 0.65) favours dense/arpeggiated templates.
The arc influence SHALL be a score adjustment applied before ranking.

#### Scenario: Low arc boosts lamentful/sparse melody templates
- **WHEN** section arc=0.15 and lamentful-tagged patterns compete with dense patterns
- **THEN** lamentful-tagged patterns receive a positive score adjustment

#### Scenario: High arc favours dense patterns
- **WHEN** section arc=0.80 and dense-tagged patterns compete with sparse patterns
- **THEN** dense-tagged patterns receive a positive score adjustment
