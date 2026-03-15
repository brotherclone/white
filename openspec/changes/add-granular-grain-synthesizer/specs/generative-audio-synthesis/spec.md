## MODIFIED Requirements

### Requirement: Granular Chromatic Texture Synthesis
The system SHALL provide a standalone grain synthesizer that produces continuous audio
textures from corpus segments scored by Refractor for a target color.

#### Scenario: texture generated for a valid color
- **WHEN** `grain_synthesizer.py --color Red --duration 30` is run
- **THEN** a 30-second WAV file is written alongside a `grain_map.yml` listing each
  grain's source segment, time offset, and Refractor chromatic_match score

#### Scenario: grain pool uses Refractor scoring
- **WHEN** the grain pool is assembled for a target color
- **THEN** segments are ranked by `retrieve_by_color()` (chromatic_match via Refractor),
  not by random selection from the color-labeled corpus

#### Scenario: crossfade produces no audible clicks
- **WHEN** grains are joined with Hann-windowed crossfades
- **THEN** the RMS energy at each join boundary does not exceed the RMS of adjacent
  grain tails (no discontinuity spike)

#### Scenario: mixed mono/stereo pool handled
- **WHEN** the grain pool contains a mix of mono and stereo source files
- **THEN** all grains are normalized to stereo before crossfade; output is always stereo
