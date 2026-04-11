## MODIFIED Requirements

### Requirement: Loop-Label Section Headers

The lyric pipeline SHALL use approved melody loop labels as section headers in
generated lyrics, so each header corresponds directly to a MIDI file in
`melody/approved/`.

The pipeline SHALL derive vocal sections from `arrangement.txt` (track 4 clips)
rather than from `production_plan.yml`. A clip's presence on track 4 is the
sole signal that it is a vocal section — no `vocals: true/false` flag is needed.

Song metadata (title, BPM, time_sig, key, color, concept, sounds_like) SHALL be
read from the song proposal YAML.

Each section instance SHALL carry a `lyric_repeat_type` field (`exact`, `variation`,
or `fresh`) loaded from `production_plan.yml` if present, or auto-inferred from the
loop label otherwise.

#### Scenario: One block per unique melody clip in arrangement (exact sections)

- **WHEN** the lyric pipeline runs
- **AND** a loop labelled `melody_chorus` appears three times in `arrangement.txt`
- **THEN** the prompt and output file contain exactly one `[melody_chorus]` block
- **AND** subsequent appearances are tracked internally as `exact_repeat` instances
  and receive the same lyric text

#### Scenario: One block per instance for variation sections

- **WHEN** a loop labelled `melody_verse` appears twice in `arrangement.txt`
- **THEN** the prompt instructs Claude to write `[melody_verse]` and `[melody_verse_2]`
- **AND** both blocks appear in the output lyrics file

#### Scenario: No production_plan.yml required

- **WHEN** the lyric pipeline runs
- **AND** `production_plan.yml` is absent
- **THEN** the pipeline SHALL proceed without error
- **AND** all song metadata SHALL come from the song proposal YAML
- **AND** `lyric_repeat_type` SHALL be inferred from loop labels

#### Scenario: Missing arrangement

- **WHEN** `arrangement.txt` does not exist
- **THEN** the pipeline SHALL exit with an error:
  `"arrangement.txt not found — export from Logic before generating lyrics"`

#### Scenario: Syllable fitting per loop

- **WHEN** syllable fitting is computed for a candidate
- **THEN** fitting is calculated per approved melody label
- **AND** note count is read from `melody/approved/<label>.mid` (stripping any `_N` instance suffix)
- **AND** if the MIDI is absent, total_notes defaults to 0 with no error
- **AND** `exact_repeat` instances copy their fitting result from the first instance
