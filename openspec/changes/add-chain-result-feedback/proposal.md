# Change: Add Chain Result Feedback (Negative Constraints)

## Why

About a third of chain proposals converge on the same output: "I am a mirror" in C major at 96 BPM. The White agent (Prism) is reflexive by design — it examines its own examination — but without awareness of prior outputs, it keeps rediscovering the same attractor state. The pipeline has no memory of what it already produced.

This is a concrete problem for the final White album: we need diverse compositions across all chromatic colors, not 20 variations of "I am a mirror."

## What Changes

- Shrink-wrapped chain artifacts (from `add-shrinkwrap-chain-artifacts`) provide structured metadata per thread
- New "negative constraints" file generated from prior results: a list of BPM/key/title/concept combinations to avoid
- White agent ingests negative constraints at proposal initiation time
- Constraints are soft (influence, not hard-block) to preserve creative latitude

## Impact

- **Prerequisite**: `add-shrinkwrap-chain-artifacts` (needs structured metadata to build constraints)
- Affected code: White agent proposal generation, chain workflow initiation
- Affected files: `app/agents/white_agent.py`, workflow entry points
- New file: `chain_artifacts/negative_constraints.yml`
