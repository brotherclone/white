# Implementation Tasks

## 1. Model Architecture
- [ ] 1.1 Create `MultiClassRebracketingClassifier` class in `training/models/multiclass_classifier.py`
- [ ] 1.2 Implement MLP with softmax output layer
- [ ] 1.3 Add support for multi-label mode (sigmoid + BCELoss if needed)
- [ ] 1.4 Add class weight initialization mechanism

## 2. Data Pipeline
- [ ] 2.1 Update dataset to return rebracketing type labels (not just binary)
- [ ] 2.2 Implement label encoding for rebracketing taxonomy
- [ ] 2.3 Add class distribution analysis utilities
- [ ] 2.4 Implement balanced sampling for rare classes

## 3. Training Loop
- [ ] 3.1 Replace loss function with `CrossEntropyLoss`
- [ ] 3.2 Implement class weight calculation (balanced mode)
- [ ] 3.3 Add support for multi-task learning (optional: simultaneous binary + multiclass)
- [ ] 3.4 Update optimizer and scheduler for new model size

## 4. Evaluation Metrics
- [ ] 4.1 Implement per-class F1 score calculation
- [ ] 4.2 Add confusion matrix generation and visualization
- [ ] 4.3 Add macro and micro averaging metrics
- [ ] 4.4 Implement top-k accuracy for multi-label scenarios
- [ ] 4.5 Add metric logging to WandB/TensorBoard

## 5. Configuration
- [ ] 5.1 Add `model.classifier.type` config option (binary vs multiclass)
- [ ] 5.2 Add `model.classifier.num_classes` parameter
- [ ] 5.3 Add `model.classifier.class_weights` config (balanced, manual, or none)
- [ ] 5.4 Add `model.classifier.multi_label` boolean flag

## 6. Testing & Validation
- [ ] 6.1 Write unit tests for multiclass model forward pass
- [ ] 6.2 Test class weight calculation
- [ ] 6.3 Validate metric calculations with known examples
- [ ] 6.4 Run training on sample data and verify convergence
- [ ] 6.5 Compare results with Phase 1 binary classifier baseline

## 7. Documentation
- [ ] 7.1 Document rebracketing type taxonomy mapping
- [ ] 7.2 Add training guide for multiclass mode
- [ ] 7.3 Document interpretation of confusion matrices
- [ ] 7.4 Add example config files for multiclass training
