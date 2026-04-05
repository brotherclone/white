## ADDED Requirements

### Requirement: Cross-Section Melodic Continuity
The melody pipeline SHALL apply a continuity penalty to candidates whose opening note
creates a large interval leap from the closing note of the preceding approved melody
section, as ordered by `production_plan.yml`.

The penalty SHALL be a score multiplier of 0.85× applied when the interval exceeds
`melodic_continuity_semitones` (default: 4, configurable in `song_proposal.yml`).

#### Scenario: Smooth transition preferred
- **WHEN** two candidate templates for a section start within 4 semitones of the
  preceding section's last note
- **THEN** neither receives the continuity penalty and they are ranked by other factors

#### Scenario: Leap penalised
- **WHEN** a candidate's first note is more than `melodic_continuity_semitones` away
  from the preceding approved section's last note
- **THEN** the candidate's composite score is multiplied by 0.85

#### Scenario: No preceding section — no penalty
- **WHEN** the section being generated is the first approved melody section in the plan
- **THEN** no continuity penalty is applied to any candidate

#### Scenario: Custom threshold from proposal
- **WHEN** `song_proposal.yml` contains `melodic_continuity_semitones: 6`
- **THEN** the 0.85× penalty applies only when the interval exceeds 6 semitones
