# Implementation Tasks

## 1. Model Architecture
- [x] 1.1 Create `MultiClassRebracketingClassifier` class in `training/models/multiclass_classifier.py`
- [x] 1.2 Implement MLP with softmax output layer
- [x] 1.3 Add support for multi-label mode (sigmoid + BCELoss if needed)
- [x] 1.4 Add class weight initialization mechanism

## 2. Data Pipeline
- [x] 2.1 Update dataset to return rebracketing type labels (not just binary)
- [x] 2.2 Implement label encoding for rebracketing taxonomy
- [x] 2.3 Add class distribution analysis utilities
- [x] 2.4 Implement balanced sampling for rare classes (via stratified split + class weights)

## 3. Training Loop
- [x] 3.1 Replace loss function with `CrossEntropyLoss`
- [x] 3.2 Implement class weight calculation (balanced mode)
- [ ] 3.3 Add support for multi-task learning (optional: simultaneous binary + multiclass)
- [x] 3.4 Update optimizer and scheduler for new model size

## 4. Evaluation Metrics
- [x] 4.1 Implement per-class F1 score calculation
- [x] 4.2 Add confusion matrix generation and visualization
- [x] 4.3 Add macro and micro averaging metrics
- [x] 4.4 Implement top-k accuracy for multi-label scenarios
- [x] 4.5 Add metric logging to WandB/TensorBoard

## 5. Configuration
- [x] 5.1 Add `data.target_type` config option (binary, multiclass, multilabel)
- [x] 5.2 Add `data.class_mapping` for num_classes derivation
- [x] 5.3 Add class weights config (balanced mode via stats)
- [x] 5.4 Add `data.multi_label` boolean via target_type=multilabel

## 6. Testing & Validation
- [x] 6.1 Write unit tests for multiclass model forward pass
- [x] 6.2 Test class weight calculation
- [x] 6.3 Validate metric calculations with known examples
- [ ] 6.4 Run training on sample data and verify convergence
- [ ] 6.5 Compare results with Phase 1 binary classifier baseline

## 7. Documentation
- [ ] 7.1 Document rebracketing type taxonomy mapping
- [ ] 7.2 Add training guide for multiclass mode
- [ ] 7.3 Document interpretation of confusion matrices
- [ ] 7.4 Add example config files for multiclass training
