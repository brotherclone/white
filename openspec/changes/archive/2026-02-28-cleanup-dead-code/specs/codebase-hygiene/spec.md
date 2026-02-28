## REMOVED Requirements

### Requirement: Prototype demo script removed

`app/generators/midi/prototype/main.py` SHALL NOT exist in the repository.
The active chord generation entry point is `chord_pipeline.py`, which imports
`prototype/generator.py` directly.

#### Scenario: No orphaned demo in prototype

- **GIVEN** the prototype package exists
- **THEN** `prototype/main.py` SHALL NOT be present
- **AND** `chord_pipeline.py` SHALL continue to import `ChordProgressionGenerator`
  from `prototype/generator.py` without error

### Requirement: Incomplete batch validation stub removed

`scripts/batch_validate_concepts.py` SHALL NOT exist. It referenced a
non-existent module (`your_white_agent`) and was never completed.

#### Scenario: No broken import stub in scripts/

- **GIVEN** the scripts/ directory is inspected
- **THEN** `batch_validate_concepts.py` SHALL NOT be present
- **AND** no file in `scripts/` SHALL import from a module that does not exist
  in the codebase

## ADDED Requirements

### Requirement: Completed one-off training scripts archived

Completed Modal/RunPod training and data-pipeline scripts SHALL live under
`training/archive/` rather than `training/`, so that the active training
surface (`train.py`, `train_phase_four.py`, `chromatic_scorer.py`) is clearly
distinguished from finished work.

#### Scenario: Archive directory contains completed scripts

- **GIVEN** the training phase jobs are complete (fusion_model.pt committed,
  HuggingFace dataset v0.2.0 published)
- **THEN** `training/archive/` SHALL contain:
  `hf_dataset_prep.py`, `push_to_hub_fixed.py`
- **AND** none of those files SHALL remain at `training/` root level

#### Scenario: Active training scripts remain at training/ root

- **WHEN** `training/` root is listed
- **THEN** `train.py`, `train_phase_four.py`, `chromatic_scorer.py`,
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
