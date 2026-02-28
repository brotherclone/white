# cleanup-dead-code

## Summary

Remove confirmed dead code and archive completed one-off scripts from the
`app/generators/midi/prototype/`, `scripts/`, and `training/` directories.
No behaviour changes to any active pipeline.

## Motivation

Two sessions of pipeline rewrites (chord primitive collapse, lyric pipeline
remove-plan) left behind a demo script and a broken stub. The training
directory also accumulated one-off Modal/RunPod scripts and notebooks whose
jobs are finished and whose outputs are already committed or published to
HuggingFace. Keeping them in the top-level risks confusion about what is
canonical.

## Scope

### Remove (confirmed dead, safe to delete)

| File | Reason |
|------|--------|
| `app/generators/midi/prototype/main.py` | Demo script. Only imports within `prototype/`; not imported by any other module. `chord_pipeline.py` imports `generator.py` directly, not `main.py`. |
| `scripts/batch_validate_concepts.py` | Broken stub. Top-level import `from your_white_agent import create_white_agent_graph` references a module that does not exist. Never completed. |

### Archive (one-off jobs — done, outputs committed/published)

Move to `training/archive/` (scripts) and `training/notebooks/archive/` (notebooks).

**Scripts:**

| File | Job | Status |
|------|-----|--------|
| `training/hf_dataset_prep.py` | Prep + push dataset to HuggingFace | Complete — v0.2.0 published |
| `training/push_to_hub_fixed.py` | One-off Hub upload fix | Complete — superseded by `hf_dataset_prep.py` |

**Notebooks:**

| File | Purpose | Status |
|------|---------|--------|
| `training/notebooks/runpod_embedding_extraction.ipynb` | RunPod DeBERTa extraction (pre-Modal) | Superseded by Modal script |
| `training/notebooks/runpod_multitask_comparison.ipynb` | Phase 2 multi-task experiments | Complete |
| `training/notebooks/runpod_regression_validation.ipynb` | Phase 4 regression validation | Complete |
| `training/notebooks/runpod_training.ipynb` | Phase 1/2 RunPod training | Superseded by train.py |
| `training/notebooks/annotation_interface.ipynb` | Human annotation UI prototype | Superseded by `tools/annotation_cli.py` |

### Keep (confirmed active or future-relevant)

- `app/generators/midi/prototype/generator.py` — imported by `chord_pipeline.py`
- `app/generators/midi/prototype/midi_parser.py` — used by `build_database.py`
- `app/generators/midi/prototype/build_database.py` — standalone utility
- `training/models/regression_head.py` — imported by `models/__init__.py`, used in tests and `train_phase_four.py`
- `training/models/uncertainty.py` — imported by `models/__init__.py`, used by `multitask_model.py` and `concept_validator.py`
- `training/models/multitask_model.py` — used by `concept_validator.py`
- `app/generators/midi/production_plan.py` — despite deprecated spec, still imported by `assembly_manifest.py`, `drum_pipeline.py`, `song_evaluator.py`, and `lyric_pipeline.py`
- `training/train.py`, `training/train_phase_four.py` — re-training entrypoints, keep accessible
- `training/modal_embedding_extraction.py`, `training/modal_clap_extraction.py`, `training/modal_midi_fusion.py` — Modal compute scripts, keep at root for re-runs
- `training/export_onnx.py` — ONNX export script, keep at root for re-runs
- `training/notebooks/interpretability_analysis.ipynb` — analysis tool, keep

## Out of scope

- Any refactor of `production_plan.py` (separate change)
- Any changes to `add-lyric-feedback-loop` tasks
- Moving `training/train.py` or `training/train_phase_four.py` (still useful as re-training entrypoints)
