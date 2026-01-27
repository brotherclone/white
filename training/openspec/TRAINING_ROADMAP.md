# Training Pipeline Roadmap

This document provides an overview of the training pipeline improvements and extensions documented as OpenSpec changes. These changes correspond to the recommendations from the training additions outline created in desktop mode.

**Last Updated**: 2026-01-26

## Overview

The training pipeline spans 10 major phases, progressing from basic classification to advanced generative models and production deployment. Each phase has been documented as an OpenSpec change with full requirements, scenarios, and implementation tasks.

## Phase Sequence

### Phase 1: Binary Classification ✓ Complete
**Status**: Implemented and deployed
- Text-only binary classifier for `has_rebracketing_markers`
- DeBERTa-v3-base + MLP architecture
- Training loop with warmup, cosine annealing, mixed precision
- Validates that rebracketing taxonomy is learnable

### Phase 2: Multi-Class Classification ✓ Substantially Complete
**Change**: `add-multiclass-rebracketing-classifier`
**Status**: 85% complete - model trained, achieves 100% accuracy

Extends binary classification to predict specific rebracketing types (spatial, temporal, causal, perceptual, memory, etc.). Includes:
- MultiClassRebracketingClassifier with softmax output ✓
- CrossEntropyLoss for multi-class prediction ✓
- Class weighting for rare types ✓
- Per-class F1 scores and confusion matrices ✓
- Multi-label support for segments with multiple types ✓

**Remaining**:
- 3.3 Multi-task learning (classification + regression simultaneous)
- 6.4/6.5 Compare with Phase 1 baseline
- 7.x Documentation

**Key Files**:
- proposal.md:10 lines
- tasks.md:30 tasks across 7 sections
- spec.md:9 requirements, 27 scenarios

### Phase 3: Multimodal Fusion
**Change**: `add-multimodal-fusion`
**Priority**: Medium (requires Phase 2 validation first)

Combines text, audio, and MIDI representations for richer rebracketing detection. Includes:
- AudioEncoder (Wav2Vec2, CLAP, custom CNN options)
- MIDIEncoder (piano roll, event-based, Music Transformer)
- MultimodalFusion (early, late, cross-attention, gated strategies)
- Audio/MIDI preprocessing pipelines
- Cross-modal attention mechanisms

**Key Files**:
- proposal.md:Breaking change (dataset format expansion)
- tasks.md:55 tasks across 10 sections
- spec.md:14 requirements, 48 scenarios

### Phase 4: Regression Tasks ✓ Substantially Complete
**Change**: `add-regression-tasks`
**Status**: 97% complete (86/89 tasks) - CODE COMPLETE, awaiting training run

Predicts continuous Rainbow Table ontological modes (temporal, spatial, ontological distributions) plus chromatic confidence. Includes:
- RegressionHead and RainbowTableRegressionHead ✓
- Multi-task learning (classification + regression) ✓
- MSE, Huber, Smooth L1, KL divergence losses ✓
- Uncertainty estimation (ensemble, MC dropout, evidential) ✓
- MAE, RMSE, R², correlation metrics ✓
- Soft target generation from discrete labels ✓
- Concept validation API with accept/reject gates ✓
- Human-in-the-loop annotation interface ✓
- Transmigration distance computation ✓
- Album prediction from ontological scores ✓

**Remaining**:
- 8.5/8.6 Run training and compare performance (requires GPU)
- 10.12 LangGraph integration tests

**CRITICAL FIX NEEDED**: Training scripts use placeholder random embeddings instead of loading from the 69GB embedded parquet. Must implement embedding loading before training produces meaningful results.

**Key Files**:
- proposal.md:44 lines (expanded)
- tasks.md:89 tasks across 15 sections
- spec.md:12 requirements, 36 scenarios

### Phase 5: Temporal Sequence Modeling
**Change**: `add-temporal-sequence-modeling`
**Priority**: Medium (adds sequential understanding)

Models rebracketing evolution across time and segment sequences. Includes:
- TemporalDataset with context windows
- SegmentLSTM, TemporalTransformer, TemporalConvNet architectures
- Temporal positional encoding using actual time distances
- TransitionPredictor for delta prediction between segments
- Sequence-level evaluation metrics

**Key Files**:
- proposal.md:14 lines
- tasks.md:41 tasks across 9 sections
- spec.md:13 requirements, 45 scenarios

### Phase 6: Chromatic Style Transfer
**Change**: `add-chromatic-style-transfer`
**Priority**: Medium-High (enables cross-mode generation)

Generates segments in different chromatic modes while preserving content. Includes:
- ChromaticStyleEncoder for extracting mode essence
- DisentangledEncoder for content-style separation
- ChromaticDecoder for generation
- Style reconstruction, transfer, preservation losses
- Adversarial training for realism

**Key Files**:
- proposal.md:16 lines
- tasks.md:37 tasks across 9 sections
- spec.md:13 requirements, 37 scenarios

### Phase 7: Generative Models
**Change**: `add-generative-models`
**Priority**: Medium (most complex, enables synthesis)

Generates entirely new segments using VAE, Diffusion, and GPT-style models. Includes:
- RebracketingVAE conditioned on chromatic mode
- RebracketingDiffusion for high-quality generation
- SegmentGenerator (autoregressive transformer)
- Latent space manipulation
- Audio tokenization (EnCodec) and MIDI tokenization (REMI)
- FID, diversity, chromatic consistency metrics

**Key Files**:
- proposal.md:18 lines
- tasks.md:45 tasks across 10 sections
- spec.md:15 requirements, 42 scenarios

### Phase 8: Model Interpretability ~ Partially Complete
**Change**: `add-model-interpretability`
**Status**: 40% complete via notebook implementation

Analyzes and visualizes what models learn about rebracketing. Includes:
- Attention visualization (text, audio, cross-modal) - NOT STARTED
- Embedding space analysis (TSNE, UMAP) ✓ (via notebook)
- Feature attribution (Integrated Gradients, SHAP) - NOT STARTED
- Counterfactual explanations - NOT STARTED
- Chromatic geometry analysis - NOT STARTED
- Confusion matrix and misclassification analysis ✓ (via notebook)
- Confidence distribution analysis ✓ (via notebook)

**Implemented in**: `notebooks/interpretability_analysis.ipynb`

**Key Files**:
- proposal.md:13 lines
- tasks.md:31 tasks across 8 sections
- spec.md:9 requirements, 27 scenarios

### Phase 9: Data Augmentation
**Change**: `add-data-augmentation`
**Priority**: High (improves all phases)

Increases training data quality and quantity through augmentation and synthesis. Includes:
- AudioAugmenter (time stretch, pitch shift, noise, reverb)
- TextAugmenter (back-translation, synonyms, paraphrase)
- MIDIAugmenter (transpose, velocity, quantization)
- SyntheticGenerator using White Agent
- Label preservation validation

**Key Files**:
- proposal.md:12 lines
- tasks.md:28 tasks across 7 sections
- spec.md:7 requirements, 20 scenarios

### Phase 10: Production Deployment
**Change**: `add-production-deployment`
**Priority**: High (enables agent integration)

Deploys trained models for use by White Agent and other components. Includes:
- ONNX export for CPU optimization
- RebracketingInferenceAPI (FastAPI)
- RebracketingAnalyzerTool for LangGraph agents
- Streaming analysis for real-time processing
- Model versioning and registry
- Monitoring and logging

**Key Files**:
- proposal.md:17 lines
- tasks.md:36 tasks across 9 sections
- spec.md:11 requirements, 33 scenarios

### Infrastructure: Experiment Tracking & Optimization
**Change**: `add-infrastructure-improvements`
**Priority**: High (supports all phases)

Provides robust infrastructure for training at scale. Includes:
- Weights & Biases integration
- Optuna/Ray Tune hyperparameter optimization
- PyTorch DDP for multi-GPU training
- MLflow model versioning and registry
- Checkpoint management and resumption
- Training profiling and monitoring

**Key Files**:
- proposal.md:16 lines
- tasks.md:41 tasks across 9 sections
- spec.md:12 requirements, 39 scenarios

## Current Status Summary

| Phase | Status | Completion |
|-------|--------|------------|
| Phase 1 (Binary) | ✓ Complete | 100% |
| Phase 2 (Multi-Class) | ✓ Substantially Complete | 85% |
| Phase 4 (Regression) | ✓ Code Complete | 97% |
| Phase 8 (Interpretability) | ~ Partial (notebook) | 40% |
| Infrastructure | Not Started | 0% |
| Phase 3, 5, 6, 7, 9, 10 | Not Started | 0% |

## Critical Path to Production

**Immediate Priority** (to unblock White Agent validation):
1. **Fix embedding loading** in Phase 4 training scripts
2. **Run Phase 4 training** on RunPod with real embeddings
3. **Test validate_concepts.py** with trained model

**Recommended Implementation Order** (remaining work):

1. ~~**Phase 2** (Multi-Class) → Natural extension of Phase 1~~ ✓ DONE
2. ~~**Phase 8** (Interpretability) → Understand what's working~~ PARTIAL (notebook exists)
3. ~~**Phase 4** (Regression) → Ontological mode prediction~~ ✓ CODE COMPLETE
4. **Infrastructure** → Enable efficient experimentation for later phases
5. **Phase 9** (Augmentation) → Improve data before complex models
6. **Phase 5** (Temporal) → Add sequence modeling
7. **Phase 3** (Multimodal) → Big architectural change
8. **Phase 6** (Style Transfer) → Useful for White Agent generation
9. **Phase 7** (Generative) → Most complex, enables full synthesis
10. **Phase 10** (Production) → Deploy for agent integration

## Key Architectural Decisions

### Training Strategy
- **Curriculum learning**: Start with easier examples, progress to complex
- **Multi-task vs specialized**: One model for everything vs separate models per task
- **Transfer learning**: Fine-tune pretrained models vs train from scratch

### Data Strategy
- **Streaming vs batch**: Load all 14k segments or stream from disk
- **Preprocessing**: Compute features once (fast training) or on-the-fly (flexible)
- **Validation split**: Hold out entire albums or random segments

### Evaluation Strategy
- **Metrics**: Accuracy, F1, AUC-ROC for classification; MAE, RMSE for regression
- **Test set**: Hold out White Album for final evaluation
- **Human evaluation**: Validate generated segments match chromatic modes

### Computational Strategy
- **GPU budget**: RunPod costs vs local vs Colab
- **Training time**: Quick iterations (hours) vs overnight runs (days)
- **Model size**: Small fast models vs large capable models

## Using These Specs

Each OpenSpec change can be worked on independently or in combination:

```bash
# View a specific change
openspec show add-multiclass-rebracketing-classifier

# Validate all changes
openspec validate --strict

# When ready to implement a phase
openspec show add-multiclass-rebracketing-classifier
# Read proposal.md for context
# Read tasks.md for implementation checklist
# Read spec.md for detailed requirements
# Mark tasks as completed in tasks.md as you work

# After deployment
openspec archive add-multiclass-rebracketing-classifier
```

## Integration with White Album Project

These training improvements directly support White Album creation:

1. **Chromatic Understanding**: Models learn distinctions between chromatic modes (BLACK, RED, etc.)
2. **Rebracketing Detection**: Identify and classify rebracketing patterns in concepts
3. **Generative Synthesis**: Create new White Album content that transmigrates from INFORMATION to SPACE
4. **Agent Integration**: White Agent can query models for rebracketing analysis
5. **Style Transfer**: Generate concepts in different chromatic modes

## Required Fixes Before Production

### Critical: Embedding Loading (Phase 4)

All Phase 4 training/validation scripts use placeholder random embeddings:

```python
# CURRENT (broken)
embedding = torch.randn(768)  # Random noise - model learns nothing useful

# REQUIRED
embedding = load_embedding_from_parquet(segment_id)  # Real embeddings
```

**Files requiring fix**:
- `train_phase_four.py:231`
- `validate_concepts.py:164`
- `core/regression_training.py:156`

**Fix approach**:
1. Load `training_data_embedded.parquet` (69GB) at dataset init
2. Index embeddings by segment ID
3. Return real embeddings in `__getitem__`

### Medium: Incomplete Album Mappings

Missing album mappings in multiple files:

| Album | Mode | Files Missing |
|-------|------|---------------|
| Violet | Past_Person_Known | `validate_concepts.py`, `core/regression_training.py` |
| Indigo | Present_Person_Forgotten | `validate_concepts.py`, `core/regression_training.py` |

### Low: Hardcoded Paths

- `validate_concepts.py:538` hardcodes `/chain_artifacts` - should be configurable
- `core/regression_training.py:24` hardcodes parquet path

## Next Steps

**Immediate** (unblock White Agent validation):
1. Fix embedding loading in Phase 4 scripts
2. Run Phase 4 training on RunPod
3. Test `validate_concepts.py` with trained model
4. Integrate with White Agent workflow

**Ongoing**:
1. Choose next phase to implement
2. Read the corresponding OpenSpec change documentation
3. Follow the tasks.md checklist for implementation
4. Validate against spec.md requirements and scenarios
5. Deploy and integrate with White Agent workflows

## Resources

- Training additions outline: `/Volumes/LucidNonsense/White/claude_working_area/training_additions_outline.md`
- OpenSpec changes: `/Volumes/LucidNonsense/White/training/openspec/changes/`
- Project context: `/Volumes/LucidNonsense/White/training/openspec/project.md`
- Training README: `/Volumes/LucidNonsense/White/training/README.md`

---

*Last Updated: 2026-01-26*

**Status**: Phases 1, 2, and 4 code complete. Fix embedding loading, then train Phase 4 to enable White Agent validation.
