## ADDED Requirements

### Requirement: MIDI Style Reference Extraction
The system SHALL extract statistical features from locally available MIDI files
for each `sounds_like` artist and write an aggregate `StyleProfile` to
`song_context.yml` under `style_reference_profile`. If no local MIDI files exist
for any artist, this block SHALL be omitted and pipeline behaviour SHALL be unchanged.

The local directory structure SHALL be:
```
style_references/<artist_slug>/*.mid
```

Where artist_slug is the artist name lowercased with spaces replaced by underscores.

Extracted features SHALL include: note_density, mean_duration_beats,
velocity_mean, velocity_variance, interval_histogram, rest_ratio,
harmonic_rhythm, phrase_length_mean. All are averaged across all available MIDI
files for the artist, then averaged across all artists.

Profiles SHALL be cached as `style_references/<artist_slug>/profile.yml` and only
recomputed when source MIDI files change.

#### Scenario: Local MIDI files present — profile written
- **WHEN** `init_production` runs for a song with `sounds_like: [Grouper]`
- **AND** `style_references/grouper/` contains one or more MIDI files
- **THEN** `song_context.yml` contains a `style_reference_profile` block
- **AND** the profile reflects the statistical features of the MIDI files

#### Scenario: No local MIDI files — no profile written
- **WHEN** no MIDI files exist in `style_references/` for any `sounds_like` artist
- **THEN** `song_context.yml` does NOT contain a `style_reference_profile` block
- **AND** all pipelines run with existing behaviour

#### Scenario: Partial coverage — profile from available artists only
- **WHEN** `sounds_like` lists two artists but only one has local MIDI files
- **THEN** the profile is derived from only the artist with files
- **AND** a warning is logged for the missing artist
