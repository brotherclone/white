# chord-generation Specification Delta

## MODIFIED Requirements

### Requirement: Diatonic Workhorse Candidates

After Markov candidates are scored and ranked, `chord_pipeline.py` SHALL
generate an additional set of diatonic workhorse candidates from the chord
bank and append them to `review.yml`.

Diatonic candidates are assembled using the existing
`ChordProgressionGenerator.get_chord_by_function()` method — no new MIDI
synthesis or Markov traversal is required. They are intended as grounded,
guitar-playable contrast options, annotated so the reviewer knows to assign
them to verse sections. (Section labels are assigned by the human reviewer
after generation; they are not known at pipeline time.)

The following patterns SHALL be attempted. Major-key songs use the Major set;
minor-key songs use the Minor set:

**Major patterns**
| pattern_name | degrees |
|---|---|
| `I_V_vi_IV` | I – V – vi – IV |
| `I_IV_V` | I – IV – V |
| `I_vi_IV_V` | I – vi – IV – V |
| `ii_V_I` | II – V – I |

**Minor patterns**
| pattern_name | degrees |
|---|---|
| `i_VII_VI_VII` | i – VII – VI – VII |
| `i_VI_III_VII` | i – VI – III – VII |
| `i_iv_v` | i – iv – v |
| `i_VI_VII_i` | i – VI – VII – i |

For each pattern, one chord per degree SHALL be selected from the bank. If no
chord is found for a given degree the pattern SHALL be skipped silently.

Each diatonic candidate SHALL be written to `candidates/` as a MIDI file and
added to `review.yml` with:
- `id`: `diatonic_{pattern_name}`
- `source`: `diatonic`
- `scores`: `null`
- `label`: `null`
- `status`: `pending`
- `notes`: `"Diatonic workhorse — assign to verse sections"`
- `rank`: `null` (listed after all scored Markov candidates)

#### Scenario: A minor song gets diatonic candidates appended

- **GIVEN** a song in A minor
- **WHEN** chord_pipeline runs
- **THEN** `review.yml` contains Markov candidates followed by up to 4
  diatonic candidates with `source: diatonic` and `scores: null`
- **AND** their IDs follow the `diatonic_{pattern_name}` convention

#### Scenario: Pattern degree missing from bank is skipped

- **GIVEN** a pattern degree that has no matching chord in the bank for the
  current key
- **WHEN** diatonic candidate assembly runs
- **THEN** that pattern is silently skipped
- **AND** remaining patterns are still added

#### Scenario: White cut-up mode is unaffected

- **GIVEN** a song using White donor cut-up mode
- **WHEN** chord_pipeline runs
- **THEN** no diatonic candidates are added (White mode has its own candidate logic)
