# Change: Add Temporal Sequence Modeling for Rebracketing Evolution

## Why
Rebracketing is not static - it evolves across segments within albums, creating narrative arcs and ontological transitions. Modeling temporal dependencies enables the system to understand how rebracketing patterns develop, predict transitions, and capture the sequential nature of conceptual transformation.

## What Changes
- Add `TemporalDataset` that loads segments with temporal context windows (previous and next segments)
- Implement `SegmentLSTM` for learning temporal dependencies across segment sequences
- Implement `TemporalTransformer` with temporal positional encoding using actual time distances
- Implement `TemporalConvNet` (TCN) with dilated causal convolutions for long-range dependencies
- Add `TransitionPredictor` module for predicting rebracketing changes between segments
- Extend dataset to include segment ordering and temporal metadata (timestamps, album context)
- Add evaluation metrics for sequence prediction: next-segment accuracy, transition prediction

## Impact
- Affected specs: temporal-sequence-modeling (new capability)
- Affected code:
  - `training/models/temporal/` (new directory with LSTM, Transformer, TCN, Transition models)
  - `training/core/pipeline.py` - temporal dataset with context windows
  - `training/core/trainer.py` - sequence-aware batching and loss computation
- Dependencies: None beyond PyTorch
- Training complexity: Increased due to sequence modeling
- Data requirements: Segment ordering and temporal metadata needed
