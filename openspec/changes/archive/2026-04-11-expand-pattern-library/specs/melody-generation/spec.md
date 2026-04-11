## MODIFIED Requirements
### Requirement: Melody Contour Templates
Each `MelodyPattern` in the template library SHALL carry an optional `tags: list[str]`
field drawn from a controlled vocabulary: `stepwise`, `arpeggiated`, `descent`,
`wide_interval`, `sparse`, `dense`, `lamentful`. Existing patterns without tags
behave identically.

The library SHALL include the following additional lamentful/sparse templates:
- `slow_descent` — stepwise downward motion, quarter notes, phrase every 2 bars
- `breath_phrase` — 3-note phrase, long rest, 3-note phrase
- `pentatonic_lament` — minor pentatonic, descending, held notes
- `floating_repeat` — same 2-3 note motif repeated at slightly different rhythmic positions
- `single_line` — one note per bar, whole-note or dotted half

All new templates SHALL carry `lamentful`, `sparse`, or `stepwise` tags as appropriate.

#### Scenario: Tag field present on all patterns
- **WHEN** the melody pattern library is loaded
- **THEN** every `MelodyPattern` has a `tags` attribute (empty list if none assigned)

#### Scenario: Lamentful templates available
- **WHEN** the library is filtered for patterns tagged `lamentful`
- **THEN** at least 4 patterns are returned
