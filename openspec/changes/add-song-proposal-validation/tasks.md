## 1. Implementation

- [ ] 1.1 Add `load_song_proposal_from_file()` function to parse song proposal YAML format
- [ ] 1.2 Add `load_all_song_proposals()` function to parse aggregated proposals file
- [ ] 1.3 Add `--proposals-dir` CLI argument for directory of individual proposal files
- [ ] 1.4 Add `--thread-proposals` CLI argument for loading from `all_song_proposals_*.yml`
- [ ] 1.5 Extract ground-truth labels from `rainbow_color` field (temporal_mode, ontological_mode, objectional_mode)
- [ ] 1.6 Add `GroundTruthComparison` dataclass for tracking prediction vs ground truth
- [ ] 1.7 Update `validate_batch()` to compute agreement statistics when ground truth available
- [ ] 1.8 Add accuracy summary output (per-dimension and overall agreement)
- [ ] 1.9 Test with existing chain_artifacts proposals
