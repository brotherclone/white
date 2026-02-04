## 1. Implementation

- [x] 1.1 Add `load_song_proposal_from_file()` function to parse song proposal YAML format
- [x] 1.2 Add `load_all_song_proposals()` function to parse aggregated proposals file
- [x] 1.3 Add `--proposals-dir` CLI argument for directory of individual proposal files
- [x] 1.4 Add `--thread-proposals` CLI argument for loading from `all_song_proposals_*.yml`
- [x] 1.5 Extract ground-truth labels from `rainbow_color` field (temporal_mode, ontological_mode, objectional_mode)
- [x] 1.6 Add `GroundTruthComparison` dataclass for tracking prediction vs ground truth
- [x] 1.7 Update `validate_batch()` to compute agreement statistics when ground truth available
- [x] 1.8 Add accuracy summary output (per-dimension and overall agreement)
- [ ] 1.9 Test with existing chain_artifacts proposals (requires torch/RunPod environment)

## Validation

```bash
# Test with a specific thread's proposals (requires torch)
python training/validate_concepts.py --thread-proposals /path/to/all_song_proposals_*.yml

# Test with a directory of proposals
python training/validate_concepts.py --proposals-dir /chain_artifacts/

# Example output:
# GROUND TRUTH ACCURACY
# Album Accuracy:       75.0% (15/20)
# Temporal Accuracy:    80.0% (16/20)
# Spatial Accuracy:     70.0% (14/20)
# Ontological Accuracy: 65.0% (13/20)
```
