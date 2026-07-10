## ADDED Requirements

### Requirement: Album Template Diversity Scoring
The melody and bass pipelines SHALL apply an album-wide diversity multiplier to each
candidate's composite score, based on how many previously-completed songs in the same
album (`shrink_wrapped/<album>/used_templates.json`) have used that candidate's
template.

Multiplier by prior-use count:
- 0 prior uses → 1.15× (novelty bonus)
- 1 prior use → 1.0× (neutral)
- 2+ prior uses → `max(0.35, 0.6 - 0.1 * (uses - 2))` (penalty, decreasing further with
  each additional use, floored at 0.35×)

#### Scenario: Fresh template gets a bonus
- **WHEN** a template has 0 recorded uses in `used_templates.json`
- **THEN** its composite score is multiplied by 1.15

#### Scenario: Single prior use is neutral
- **WHEN** a template has exactly 1 recorded use
- **THEN** its composite score is unchanged (1.0×)

#### Scenario: Repeated use is penalised starting at 2
- **WHEN** a template has 2 recorded uses
- **THEN** its composite score is multiplied by 0.6

#### Scenario: Penalty deepens with further reuse
- **WHEN** a template has 5 recorded uses
- **THEN** its composite score is multiplied by 0.35 (the floor)

#### Scenario: Registry missing — no penalty
- **WHEN** no `used_templates.json` exists yet for the album
- **THEN** every template is treated as having 0 prior uses
