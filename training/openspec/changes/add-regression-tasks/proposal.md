# Change: Add Regression Tasks for Continuous Rebracketing Metrics

## Why
While classification identifies rebracketing types, continuous metrics like intensity, boundary fluidity, and temporal complexity provide nuanced quantitative measures. Regression enables the model to predict degree rather than just category, supporting richer analytical and generative capabilities.

## What Changes
- Add `RegressionHead` module for predicting continuous rebracketing metrics
- Implement multi-task learning framework combining classification and regression
- Add loss functions for regression: MSE, Huber loss, and weighted combinations
- Add evaluation metrics: MAE, RMSE, R², correlation coefficients
- Extend dataset to include continuous target variables (intensity scores, fluidity scores)
- Implement uncertainty estimation via ensemble predictions or evidential deep learning
- Add configuration for multi-task loss weighting and regression targets

## Impact
- Affected specs: regression-tasks (new capability)
- Affected code:
  - `training/models/regression_head.py` (new)
  - `training/models/multitask_model.py` (new)
  - `training/core/trainer.py` - multi-task loss computation
  - `training/core/pipeline.py` - continuous target loading
  - `training/evaluation/` - regression metrics
- Dependencies: scikit-learn for evaluation metrics
- Training complexity: Multi-task learning adds optimization challenges
- Data requirements: Continuous annotations for rebracketing metrics needed
