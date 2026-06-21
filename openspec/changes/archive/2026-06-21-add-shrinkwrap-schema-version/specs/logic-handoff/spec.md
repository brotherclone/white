## ADDED Requirements

### Requirement: Stage Regression Info
`white_composition.logic_handoff` SHALL expose a `regression_info(logic_song_dir, current, target)` function that computes the set of files that would be deleted if the MixStage were moved backward from `current` to `target`, and whether any such files actually exist.

The function SHALL raise `ValueError` when `target` is not strictly earlier than `current` in `_STAGE_ORDER`, or when either stage name is not a valid `MixStage` value.

`REGRESSION_FILE_MAP` SHALL be a module-level constant mapping each `MixStage` value to a list of glob patterns (relative to the Logic song dir) that are deleted when moving backward past that stage:

```
"lyrics":             ["lyrics*.txt", "*.lrc"]
"vocal_placeholders": ["MIDI/melody/vocal_placeholder*.mid", "MIDI/melody/assembled*.mid"]
"recording":          ["Recordings/*"]
"augmentation":       ["Augmented/*"]
"cleaning":           ["Cleaned/*"]
```

Stages not in the map (`rough_mix`, `mix_candidate`, `final_mix`) produce no file deletions.

The returned dict has the shape `{ "destructive": bool, "files_to_delete": list[str] }` where `files_to_delete` contains the resolved absolute paths of matching files that actually exist.

#### Scenario: Destructive regression identifies files
- **WHEN** `regression_info(song_dir, "vocal_placeholders", "lyrics")` is called
- **AND** `song_dir/MIDI/melody/vocal_placeholder_verse.mid` exists
- **THEN** returns `{ "destructive": True, "files_to_delete": ["<song_dir>/MIDI/melody/vocal_placeholder_verse.mid"] }`

#### Scenario: Non-destructive regression
- **WHEN** `regression_info(song_dir, "mix_candidate", "rough_mix")` is called
- **THEN** returns `{ "destructive": False, "files_to_delete": [] }`
  because neither stage is in `REGRESSION_FILE_MAP`

#### Scenario: Multi-stage regression collects all passed-through patterns
- **WHEN** `regression_info(song_dir, "recording", "lyrics")` is called
- **THEN** patterns for both `vocal_placeholders` and `recording` are resolved
- **AND** `files_to_delete` contains all matching files for both stages

#### Scenario: Forward movement raises ValueError
- **WHEN** `regression_info(song_dir, "lyrics", "vocal_placeholders")` is called
  (target is later than current)
- **THEN** `ValueError` is raised with a descriptive message

#### Scenario: Invalid stage name raises ValueError
- **WHEN** either `current` or `target` is not a valid `MixStage` value
- **THEN** `ValueError` is raised
