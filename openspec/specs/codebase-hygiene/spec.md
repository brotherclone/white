# codebase-hygiene Specification

## Purpose
TBD - created by archiving change cleanup-dead-code. Update Purpose after archive.
## Requirements
### Requirement: Completed one-off training scripts archived

Completed Modal/RunPod training and data-pipeline scripts SHALL live under
`training/archive/` rather than `training/`, so that the active training
surface (`train.py`, `train_phase_four.py`, `refractor.py`) is clearly
distinguished from finished work.

#### Scenario: Archive directory contains completed scripts

- **GIVEN** the training phase jobs are complete (refractor.pt committed,
  HuggingFace dataset v0.2.0 published)
- **THEN** `training/archive/` SHALL contain:
  `hf_dataset_prep.py`, `push_to_hub_fixed.py`
- **AND** none of those files SHALL remain at `training/` root level

#### Scenario: Active training scripts remain at training/ root

- **WHEN** `training/` root is listed
- **THEN** `train.py`, `train_phase_four.py`, `refractor.py`,
  `validate_concepts.py`, `verify_extraction.py` SHALL still be present
  at root level

### Requirement: Completed RunPod/annotation notebooks archived

The system SHALL maintain completed RunPod training notebooks and the
superseded annotation interface notebook under `training/notebooks/archive/`
rather than `training/notebooks/` root.

#### Scenario: Archive directory contains completed notebooks

- **GIVEN** RunPod training is complete and Modal scripts supersede RunPod notebooks
- **THEN** `training/notebooks/archive/` SHALL contain:
  `runpod_embedding_extraction.ipynb`, `runpod_multitask_comparison.ipynb`,
  `runpod_regression_validation.ipynb`, `runpod_training.ipynb`,
  `annotation_interface.ipynb`

#### Scenario: Active notebook remains at notebooks/ root

- **WHEN** `training/notebooks/` root is listed
- **THEN** `interpretability_analysis.ipynb` SHALL remain at root level

