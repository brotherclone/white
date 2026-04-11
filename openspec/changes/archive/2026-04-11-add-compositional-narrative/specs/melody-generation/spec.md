## MODIFIED Requirements

### Requirement: Composite Scoring
The melody pipeline composite scoring SHALL incorporate narrative lead_voice and
register constraints when `composition_narrative.yml` is present.

- `lead_voice: none` → skip melody generation for this section (return empty)
- `lead_voice: melody` → standard generation (no adjustment)
- `register: low` or `low_mid` → prefer templates with descent contours; lower starting pitch
- `register: mid_high` or `high` → prefer templates with wide interval or arpeggiated contours

The narrative constraint adjustment SHALL be a tag-weighted score delta applied
alongside the arc-based adjustment.

#### Scenario: Low register shifts starting pitch and contour
- **WHEN** section narrative declares `register: low`
- **AND** both descent and arpeggiated templates are candidates
- **THEN** descent-tagged templates score higher

#### Scenario: No lead voice skips melody section
- **WHEN** section narrative declares `lead_voice: none`
- **THEN** no melody candidates are generated for the section
- **AND** the section is skipped in the melody pipeline output
