# Spec: drum-breakbeat-library

Adds a `breakbeat` genre family and associated templates to the drum pattern library,
drawing from the classic-breakbeats reference spreadsheet.

---

## ADDED Requirements

### Requirement: breakbeat genre family registered in GENRE_FAMILY_KEYWORDS

`GENRE_FAMILY_KEYWORDS` MUST include a `"breakbeat"` key whose value is a list
containing at minimum: `"breakbeat"`, `"hip-hop"`, `"boom bap"`, `"break"`,
`"funk"`, `"soul"`, `"r&b"`, `"groove"`.

#### Scenario: hip-hop genre tag maps to breakbeat family

Given a song proposal with genre tags `["hip-hop", "art-pop"]`
When `map_genres_to_families` is called
Then `"breakbeat"` is included in the returned families

#### Scenario: funk genre tag maps to breakbeat family

Given a song proposal with genre tags `["neo-soul", "funk"]`
When `map_genres_to_families` is called
Then `"breakbeat"` is included in the returned families

#### Scenario: unrelated tag does not map to breakbeat

Given a song proposal with genre tags `["ambient", "drone"]`
When `map_genres_to_families` is called
Then `"breakbeat"` is NOT in the returned families

---

### Requirement: new GM percussion voices added

`GM_PERCUSSION` MUST include the following additional entries:
- `"tambourine"`: 54
- `"conga_high"`: 63
- `"conga_low"`: 64
- `"cowbell"`: 56

These SHALL be available for use in any template voice definition and MUST pass
the `test_all_templates_use_valid_gm_voices` validation.

#### Scenario: tambourine resolves to MIDI note 54

Given `GM_PERCUSSION["tambourine"]`
Then the value is `54`

#### Scenario: conga voices resolve to correct MIDI notes

Given `GM_PERCUSSION["conga_high"]` and `GM_PERCUSSION["conga_low"]`
Then the values are `63` and `64` respectively

---

### Requirement: TEMPLATES_4_4_BREAKBEAT covers all three energy levels

`TEMPLATES_4_4_BREAKBEAT` MUST contain at least one template at each of
`"low"`, `"medium"`, and `"high"` energy. The list MUST contain at minimum 15
templates total (9 named classic breaks at medium energy + low/high variants).

All templates MUST have `genre_family="breakbeat"` and `time_sig=(4, 4)`.
All beat positions MUST be in the range `[0.0, 3.75]` using 0.25 increments for
16th-note precision. All velocity levels MUST be valid keys in `VELOCITY`.
All voice names MUST be valid keys in `GM_PERCUSSION`.

#### Scenario: breakbeat family has all three energy levels

Given `ALL_TEMPLATES` after this change
When filtering for `genre_family="breakbeat"` and `time_sig=(4, 4)`
Then at least one template has `energy="low"`
And at least one template has `energy="medium"`
And at least one template has `energy="high"`

#### Scenario: minimum 15 breakbeat templates present

Given `ALL_TEMPLATES` after this change
When filtering for `genre_family="breakbeat"`
Then the count is at least 15

#### Scenario: 9 named classic breaks present as medium-energy templates

Given `ALL_TEMPLATES` after this change
Then templates with the following names exist, each with `energy="medium"`:
- `billie_jean`
- `funky_drummer`
- `impeach_the_president`
- `when_the_levee_breaks`
- `walk_this_way`
- `its_a_new_day`
- `papa_was_too`
- `the_big_beat`
- `ashleys_roachclip`

#### Scenario: all breakbeat templates use valid beat positions

Given any breakbeat template in `ALL_TEMPLATES`
For every voice, for every `(position, velocity)` onset
Then `0.0 <= position <= 3.75`
And `position % 0.25 == 0`

#### Scenario: funky_drummer encodes ghost notes

Given the `funky_drummer` template
Then its `snare` or `kick` voice contains at least one onset with velocity `"ghost"`

#### Scenario: papa_was_too includes tambourine

Given the `papa_was_too` template
Then `"tambourine"` is a key in its `voices` dict

#### Scenario: ashleys_roachclip includes congas

Given the `ashleys_roachclip` template
Then `"conga_high"` or `"conga_low"` is a key in its `voices` dict

---

### Requirement: TEMPLATES_4_4_BREAKBEAT registered in ALL_TEMPLATES

`ALL_TEMPLATES` MUST include all templates from `TEMPLATES_4_4_BREAKBEAT`.
Template names MUST remain globally unique across the entire registry.

#### Scenario: breakbeat templates appear in ALL_TEMPLATES

Given `ALL_TEMPLATES`
When filtering by `genre_family="breakbeat"`
Then the count matches `len(TEMPLATES_4_4_BREAKBEAT)`

#### Scenario: no duplicate template names after addition

Given `ALL_TEMPLATES` after this change
Then all template names are unique

---

### Requirement: existing drum template tests continue to pass

All new templates SHALL satisfy every structural invariant enforced by the existing
test suite. The `GM_PERCUSSION` additions SHALL be backward-compatible: existing
voice names MUST retain their MIDI note values unchanged.

#### Scenario: test_all_templates_use_valid_gm_voices passes with new voices

Given the new `tambourine`, `conga_high`, `conga_low`, `cowbell` entries in `GM_PERCUSSION`
When `test_all_templates_use_valid_gm_voices` runs against `ALL_TEMPLATES`
Then no template fails the voice name validation
