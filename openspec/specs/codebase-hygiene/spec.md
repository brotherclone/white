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

### Requirement: Single virtual environment
The project SHALL maintain exactly one virtual environment (`.venv`) for all development,
testing, and pipeline execution. A second environment (`.venv312`) SHALL NOT be required
for any task. This is achieved by pinning `transformers>=4.47,<5` in `pyproject.toml`.

#### Scenario: Full test suite runs under .venv
- **WHEN** `python -m pytest tests/` is run under `.venv`
- **THEN** all tests pass without any subprocess re-dispatch to an alternate interpreter

#### Scenario: Refractor imports cleanly under .venv
- **WHEN** `from training.refractor import Refractor` is executed under `.venv`
- **THEN** no ImportError and no venv-mismatch warning is emitted

#### Scenario: Pipeline commands use .venv
- **WHEN** any pipeline step from SONG_GENERATION_PROCESS.md is run
- **THEN** the command SHALL use `.venv/bin/python` not `.venv312/bin/python`

### Requirement: transformers version pinned below 5.x
`pyproject.toml` SHALL pin `transformers>=4.47,<5` to prevent uv from resolving to a
breaking release and to document the constraint explicitly.

#### Scenario: uv resolves transformers to 4.x
- **WHEN** `uv sync` is run
- **THEN** `uv.lock` SHALL record a transformers version in the `4.x` series

### Requirement: Python 3.12 minimum declared
`pyproject.toml` SHALL declare `requires-python = ">=3.12"` to match the `.python-version`
pin and make the project's actual Python requirement explicit.

#### Scenario: requires-python matches .python-version
- **GIVEN** `.python-version` contains `3.12`
- **THEN** `pyproject.toml` `requires-python` SHALL be `">=3.12"`

### Requirement: venv312 test infrastructure removed
The codebase SHALL contain no venv312 pytest marker, no conftest subprocess re-runner,
and no test decorators referencing venv312. The marker SHALL NOT appear in pytest.ini
or tests/conftest.py.

#### Scenario: No venv312 marker in test suite
- **WHEN** `grep -r "venv312" tests/` is run
- **THEN** no matches SHALL be returned

