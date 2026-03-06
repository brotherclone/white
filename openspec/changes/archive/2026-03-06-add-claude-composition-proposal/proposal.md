# Change: Claude Composition Proposal

## Why
After all loops are approved, the human currently goes straight into Logic and
arranges the song freehand. Claude has read the song concept, knows the color
target, and can see the full approved loop inventory — but contributes nothing
to the arrangement decision. That's a missed creative collaboration opportunity.

The proposal step gives Claude a voice before Logic opens: "here is how I think
this song should move, and why." The human may follow it closely, diverge wildly,
or use it as a foil. Either way, the delta between Claude's proposal and the actual
Logic arrangement becomes meaningful creative data captured by the drift report.

This is distinct from the deprecated `production_plan.yml`, which was a structural
manifest generated mechanically from approved chord labels. The composition proposal
is a genuine creative act: Claude reasoning about energy arc, section sequencing,
narrative shape, and why those choices serve the song's concept and color target.

## What Changes
- New module `app/generators/midi/composition_proposal.py`
- Reads the song proposal YAML and the approved loop inventory (chord, drum, bass,
  melody review files) to build a context prompt
- Calls the Claude API (`claude-sonnet-4-6`) and asks for a structured arrangement
  proposal with rationale
- Writes `composition_proposal.yml` to the production directory
- Schema includes: `proposed_sections` (ordered list with repeat, energy, transition
  notes), `rationale` (Claude's compositional reasoning in prose), `proposed_by: claude`,
  `generated` timestamp, `loop_inventory` (what was available), `color_target`
- The file is advisory — no downstream pipeline reads it as a dependency
- Drift against `arrangement.txt` is captured by the existing `compute_drift` in
  `assembly_manifest.py` (extended to accept a proposal path as the baseline)

## Impact
- Affected specs: new capability `composition-proposal`
- Affected code:
  - New: `app/generators/midi/composition_proposal.py`
  - Modified: `app/generators/midi/assembly_manifest.py` — extend `compute_drift`
    to optionally diff against `composition_proposal.yml` instead of
    `production_plan.yml`
  - New tests: `tests/generators/midi/test_composition_proposal.py`
- No breaking changes — purely additive
- Depends on: Anthropic SDK (`anthropic` already in deps)
