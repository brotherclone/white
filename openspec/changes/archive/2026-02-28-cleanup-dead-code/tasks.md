## 1. Remove dead code

- [ ] 1.1 Delete `app/generators/midi/prototype/main.py`
- [ ] 1.2 Delete `scripts/batch_validate_concepts.py`

## 2. Archive completed training scripts

- [ ] 2.1 Create `training/archive/` directory
- [ ] 2.2 Move `training/hf_dataset_prep.py` → `training/archive/`
- [ ] 2.3 Move `training/push_to_hub_fixed.py` → `training/archive/`

## 3. Archive completed notebooks

- [ ] 3.1 Create `training/notebooks/archive/` directory
- [ ] 3.2 Move `training/notebooks/runpod_embedding_extraction.ipynb` → `training/notebooks/archive/`
- [ ] 3.3 Move `training/notebooks/runpod_multitask_comparison.ipynb` → `training/notebooks/archive/`
- [ ] 3.4 Move `training/notebooks/runpod_regression_validation.ipynb` → `training/notebooks/archive/`
- [ ] 3.5 Move `training/notebooks/runpod_training.ipynb` → `training/notebooks/archive/`
- [ ] 3.6 Move `training/notebooks/annotation_interface.ipynb` → `training/notebooks/archive/`

## 4. Verify nothing broke

- [ ] 4.1 Run `python -m pytest tests/generators/midi/ -q` — all passing
- [ ] 4.2 Confirm `chord_pipeline.py` still imports cleanly: `python -c "from app.generators.midi.chord_pipeline import run_chord_pipeline"`
- [ ] 4.3 Confirm no remaining import of deleted files: `grep -r "prototype.main\|batch_validate" --include="*.py" .`
