## ADDED Requirements

### Requirement: Loop-Label Section Headers

The lyric pipeline SHALL use approved melody loop labels as section headers in
generated lyrics, so each header corresponds directly to a MIDI file in
`melody/approved/`.

#### Scenario: One block per unique approved melody label

- **WHEN** the lyric pipeline reads vocal sections
- **THEN** it SHALL produce one lyric block per unique approved melody label
  (e.g. `melody_verse_alternate`, `melody_verse_alternate_2`, `melody_bridge`)
- **AND** if a production plan section contains multiple approved melody labels,
  each label produces a separate block
- **AND** section names (e.g. `Verse`, `Bridge`) SHALL NOT appear as headers

#### Scenario: Prompt uses loop labels

- **WHEN** the Claude prompt is built
- **THEN** section headers in the prompt SHALL use the approved melody loop label
  (e.g. `[melody_verse_alternate]`) rather than the plan section name

#### Scenario: Syllable fitting per loop

- **WHEN** syllable fitting is computed for a candidate
- **THEN** fitting is calculated per approved melody label
- **AND** note count is read from `melody/approved/<label>.mid`

#### Scenario: lyrics_review.yml vocal_sections field

- **WHEN** `lyrics_review.yml` is initialised
- **THEN** the `vocal_sections` field SHALL list approved melody loop labels
  in the order they appear in the production plan

#### Scenario: lyrics.txt headers match MIDI filenames

- **WHEN** a candidate is promoted to `melody/lyrics.txt`
- **THEN** each `[header]` in the file SHALL match the stem of an approved
  MIDI file in `melody/approved/`
- **AND** comments MAY be used to annotate pass numbers or section context
