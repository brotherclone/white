# Change: Add Multi-Class Rebracketing Type Classifier

## Why
Phase 1 successfully validated that rebracketing presence is learnable (binary classification). Now we need to predict the specific rebracketing type taxonomy (spatial, temporal, causal, perceptual, memory, etc.) to enable more nuanced understanding of conceptual transformations.

## What Changes
- Add `MultiClassRebracketingClassifier` model with softmax output for rebracketing type taxonomy
- Replace `BCEWithLogitsLoss` with `CrossEntropyLoss` for multi-class prediction
- Add class weighting mechanism for rare rebracketing types
- Implement per-class F1 scores and confusion matrix evaluation
- Add multi-label support for segments with multiple rebracketing types
- Extend configuration schema to support multiclass parameters (num_classes, class_weights)
- Add evaluation metrics: macro/micro F1, per-class precision/recall, confusion matrices

## Impact
- Affected specs: multiclass-rebracketing-classifier (new capability)
- Affected code:
  - `training/models/` - new multiclass model architecture
  - `training/core/pipeline.py` - label preprocessing for multi-class
  - `training/core/trainer.py` - loss function and metric updates
  - `training/config/` - new configuration schema sections
- Dependencies: Existing text-only classifier foundation from Phase 1
- Training time: Expected to increase due to additional classes
