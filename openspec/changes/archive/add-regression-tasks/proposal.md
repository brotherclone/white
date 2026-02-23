# Change: Add Regression Tasks for Continuous Rebracketing Metrics

## Why
While classification identifies rebracketing types, continuous metrics like intensity, boundary fluidity, and temporal complexity provide nuanced quantitative measures. Rainbow Table ontological modes (temporal, spatial, ontological) require continuous probability distributions to capture hybrid states, liminal concepts, and transmigration distances. These regression outputs form the core of the chromatic fitness function — the Evolutionary Music Generator uses predicted mode distributions to score how well a MIDI/audio candidate matches a target chromatic mode.

## What Changes
- Add `RegressionHead` module for predicting continuous rebracketing metrics
- Add Rainbow Table ontological regression heads (temporal, spatial, ontological mode distributions + confidence)
- Implement multi-task learning framework combining classification and regression
- Add loss functions for regression: MSE, Huber loss, and weighted combinations
- Add evaluation metrics: MAE, RMSE, R², correlation coefficients
- Extend dataset to include continuous target variables (intensity scores, fluidity scores)
- Add soft target derivation from discrete Rainbow Table labels with label smoothing
- Implement uncertainty estimation via ensemble predictions or evidential deep learning
- Add hybrid state detection for liminal concepts spanning multiple modes
- Implement transmigration distance computation for mode-to-mode transitions
- Add album assignment prediction from ontological score distributions
- Add configuration for multi-task loss weighting and regression targets

## Impact
- Affected specs: regression-tasks (new capability + Rainbow Table extensions)
- Affected code:
  - `training/models/regression_head.py` (new)
  - `training/models/rainbow_table_regression_head.py` (new - 10 outputs)
  - `training/models/multitask_model.py` (new)
  - `training/core/trainer.py` - multi-task loss computation
  - `training/core/pipeline.py` - continuous target loading, soft target generation
  - `training/evaluation/` - regression metrics, hybrid detection
  - `training/validation/transmigration.py` (new - distance computation)
- Dependencies:
  - scikit-learn for evaluation metrics
  - pydantic for result dataclasses
- Training complexity: Multi-task learning adds optimization challenges
- Data requirements:
  - Continuous annotations for rebracketing metrics needed
  - Soft target generation from discrete Rainbow Table labels
  - Human-in-the-loop refinement for ambiguous segments
- Downstream: Regression outputs (mode distributions, confidence, transmigration distance) are consumed by the ChromaticScorer in the Evolutionary Music Generator
