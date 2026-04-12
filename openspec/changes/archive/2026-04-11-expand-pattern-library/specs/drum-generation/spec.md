## MODIFIED Requirements
### Requirement: Drum Pattern Templates
Each `DrumPattern` in the template library SHALL carry an optional `tags: list[str]`
field drawn from a controlled vocabulary: `sparse`, `dense`, `half_time`, `brushed`,
`motorik`, `ambient`, `ghost_only`, `electronic`. Existing patterns without tags
behave identically to current behaviour.

The library SHALL include the following additional sparse/atmospheric templates:
- `half_time_sparse` — kick on beat 1, snare on beat 3, open hat on the off-beat
- `ghost_verse` — ghost snare only, no kick, whisper hats
- `brushed_folk` — brush swells on 2 and 4, light kick, no hi-hat grid
- `ambient_pulse` — single low kick every 2 bars, crash swell on bar 4
- `kosmische_slow` — motorik feel at half tempo

All new templates SHALL carry `sparse` and/or `ambient` tags.

#### Scenario: Tag field present on all patterns
- **WHEN** the pattern library is loaded
- **THEN** every `DrumPattern` has a `tags` attribute (empty list if none assigned)

#### Scenario: Sparse templates available
- **WHEN** the library is filtered for patterns tagged `sparse`
- **THEN** at least 5 patterns are returned

## ADDED Requirements
### Requirement: Aesthetic Tag-Weighted Selection
Pipeline phases SHALL read an optional `aesthetic_hints` dict from `song_context.yml`.
When present, phases SHALL apply a score bonus of +0.1 to candidates whose pattern
tags match the hints (keys: `density` — `sparse` | `moderate` | `dense`; `texture` —
`hazy` | `clean` | `rhythmic`). A mis-matched density tag SHALL apply a penalty of
−0.05. The Refractor chromatic score remains dominant; tag weighting is a soft prior.

#### Scenario: Sparse density hint boosts sparse-tagged patterns
- **WHEN** `aesthetic_hints.density == "sparse"` and a candidate uses a `sparse`-tagged pattern
- **THEN** the candidate's composite score is increased by 0.1 relative to an untagged candidate

#### Scenario: No aesthetic_hints — behaviour unchanged
- **WHEN** `aesthetic_hints` is absent from song context
- **THEN** selection behaviour is identical to the pre-tags baseline
