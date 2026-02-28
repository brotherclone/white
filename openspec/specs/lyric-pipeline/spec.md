# lyric-pipeline Specification

## Purpose
TBD - created by archiving change update-lyric-loop-headers. Update Purpose after archive.
## Requirements
### Requirement: Loop-Label Section Headers

The lyric pipeline SHALL use approved melody loop labels as section headers in
generated lyrics, so each header corresponds directly to a MIDI file in
`melody/approved/`.

The pipeline SHALL derive vocal sections from `arrangement.txt` (track 4 clips)
rather than from `production_plan.yml`. A clip's presence on track 4 is the
sole signal that it is a vocal section — no `vocals: true/false` flag is needed.

Song metadata (title, BPM, time_sig, key, color, concept, sounds_like) SHALL be
read from the song proposal YAML.

#### Scenario: One block per unique melody clip in arrangement

- **WHEN** the lyric pipeline runs
- **AND** `arrangement.txt` exists in the production directory
- **THEN** it SHALL read all track-4 clips from the arrangement
- **AND** produce one lyric block per unique clip name, in first-seen order
- **AND** `bars` SHALL be derived from the clip duration and the song BPM/time_sig
- **AND** `repeat` SHALL equal the number of times that clip appears in the arrangement

#### Scenario: No production_plan.yml required

- **WHEN** the lyric pipeline runs
- **AND** `production_plan.yml` is absent
- **THEN** the pipeline SHALL proceed without error
- **AND** all song metadata SHALL come from the song proposal YAML

#### Scenario: Missing arrangement

- **WHEN** `arrangement.txt` does not exist
- **THEN** the pipeline SHALL exit with an error:
  `"arrangement.txt not found — export from Logic before generating lyrics"`

#### Scenario: Syllable fitting per loop

- **WHEN** syllable fitting is computed for a candidate
- **THEN** fitting is calculated per approved melody label
- **AND** note count is read from `melody/approved/<label>.mid`
- **AND** if the MIDI is absent, total_notes defaults to 0 with no error

