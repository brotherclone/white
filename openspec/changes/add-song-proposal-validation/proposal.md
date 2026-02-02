# Change: Add Song Proposal Validation Mode

## Why
Concept generation is time-consuming for validation runs on RunPod. Song proposals in `chain_artifacts` already contain rich concept text with ground-truth rainbow color labels (temporal/spatial/ontological modes), making them ideal for faster validation cycles. Each chain run produces 10-20 iterations.

## What Changes
- Add `--proposals-dir` input option to `training/validate_concepts.py` for validating song proposals
- Add `--thread-proposals` input option to load from a specific thread's `all_song_proposals_*.yml`
- Parse song proposal format (extract `concept` field and `rainbow_color` metadata)
- Compare predicted labels against ground-truth rainbow_color modes for accuracy reporting
- Output validation summary with agreement statistics between model predictions and chain labels

## Impact
- Affected code: `training/validate_concepts.py`
- New capability: Validate using existing song proposals instead of generating new concepts
- Benefit: Dramatically faster validation cycles for RunPod deployments
