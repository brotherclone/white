## ADDED Requirements

### Requirement: Lyric Repeat Type
The lyric pipeline SHALL classify each vocal section instance with a `lyric_repeat_type`
indicating whether repeated plays of the same melody loop should receive identical,
structurally-related, or fully independent lyric content.

Three types are defined:
- **`exact`** — the section's lyrics are written once and repeated verbatim across all
  plays (chorus, refrain, hook). Only the first instance appears in the generation prompt;
  subsequent plays reuse the same lyric block.
- **`variation`** — each play of the loop receives its own lyric lines, but Claude is
  instructed to preserve rhyme scheme, meter, and core imagery across instances (verse 2
  vs verse 1). Each instance appears in the prompt with a numbered variation note.
- **`fresh`** — each instance is treated as fully independent with no structural
  relationship to other plays of the same loop (bridge, outro, climax). This is the
  default when no type is inferred or specified.

The `lyric_repeat_type` SHALL be:
1. Auto-detected from the loop label when no override exists:
   - Label contains `chorus`, `refrain`, or `hook` (case-insensitive) → `exact`
   - Label contains `verse` or `pre_chorus`/`pre-chorus` (case-insensitive) → `variation`
   - All other labels → `fresh`
2. Overridable via `lyric_repeat_type` in the corresponding `production_plan.yml` section.

#### Scenario: Chorus repeats verbatim
- **WHEN** a loop labelled `melody_chorus` appears three times in `arrangement.txt`
- **THEN** the generation prompt contains exactly one `[melody_chorus]` block instruction
- **AND** that block carries a note that it repeats verbatim
- **AND** the output lyrics file has one `[melody_chorus]` block
- **AND** ACE Studio receives the same lyric content for all three arrangement instances

#### Scenario: Verse instances vary
- **WHEN** a loop labelled `melody_verse` appears twice in `arrangement.txt`
- **THEN** the generation prompt contains a `[melody_verse]` block and a `[melody_verse_2]` block
- **AND** the second block carries a variation note referencing the first
- **AND** Claude writes distinct content for each, sharing rhyme scheme and meter

#### Scenario: Fresh instance generated independently
- **WHEN** a loop labelled `melody_bridge` appears once
- **THEN** the generation prompt contains one `[melody_bridge]` block with no repeat notes
- **AND** behaviour is identical to the current pipeline

#### Scenario: Manual override in production plan
- **WHEN** a section in `production_plan.yml` has `lyric_repeat_type: exact`
- **AND** the loop label would otherwise infer `variation` (e.g., `melody_verse`)
- **THEN** the pipeline uses the explicit value from the plan, not the inferred value

#### Scenario: Syllable fitting for exact repeats
- **WHEN** fitting is computed for a section with `exact_repeat` instances
- **THEN** the fitting result for each `exact_repeat` instance matches the first
  instance (same MIDI, same lyrics block)
