# Training Pipeline Roadmap

This document provides an overview of the training pipeline improvements and extensions documented as OpenSpec changes. These changes correspond to the recommendations from the training additions outline created in desktop mode.

**Last Updated**: 2026-01-26

## Overview

The training pipeline spans 10 major phases, progressing from basic classification to advanced generative models and production deployment. Each phase has been documented as an OpenSpec change with full requirements, scenarios, and implementation tasks.

## CRITICAL ARCHITECTURAL CLARIFICATION (2026-02-06)

**The ML models do NOT integrate with White Concept Agent.**

The White Concept Agent generates concepts through philosophical transmigration (INFORMATION → SPACE). It operates based on chromatic taxonomy and rebracketing theory. It does not need ML validation to function - it already works.

**The ML models ARE for a future Music Production Agent** (to be built):

### The Actual Use Case: Evolutionary Music Composition
```
1. White Agent generates concept (text) 
   ↓
2. Music Production Agent begins composition:
   - Generate 50 chord progression variations
   - ML model scores each for chromatic consistency
   - Keep top 3
   ↓
3. For each top chord progression:
   - Generate 50 drum pattern variations
   - ML model scores each
   - Keep top 3
   ↓
4. Repeat for bass, melody, harmony, etc.
   ↓
5. Final candidates → human evaluation
```

The ML model is a **fitness function** scoring: "How well does this audio/MIDI match the target chromatic mode (GREEN/RED/VIOLET/etc.)?"

This requires:
- **Audio features**: What does GREEN *sound* like?
- **MIDI features**: What chord voicings, rhythms, melodic contours are GREEN?
- **Lyric-melody alignment**: How do vocals sit in the mix for GREEN vs RED?

### Impact on Phase Priorities

**CRITICAL PATH** (implement these):
- ✅ Phase 1, 2, 4: Foundation (COMPLETE)
- 🔥 **Phase 3 (Multimodal)**: THE BLOCKER - must add audio/MIDI/lyric encoding
- Phase 10 (Production): API for music generator (NOT White Agent)
- Infrastructure: Experiment tracking, multi-GPU training

**DEPRECATED** (skip for now - were aimed at text concept validation):
- Phase 5 (Temporal Sequence): Was for concept evolution over time
- Phase 6 (Chromatic Style Transfer): Was for text-to-text style transfer
- Phase 7 (Generative Models): Was for generating text concepts
- Phase 9 (Data Augmentation): Lower priority, can add later

**NEW REQUIRED** (not yet spec'd):
- Evolutionary Music Generator (`app/generator` expansion)
- Multi-stage pruning/scoring orchestration
- Integration with Logic Pro / DAW export

**KEPT AS-IS**:
- Phase 8 (Interpretability): Still useful for understanding what model learned

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

### Phase 4: Regression Tasks ✓ COMPLETE
**Change**: `add-regression-tasks`
**Status**: 100% complete - TRAINED AND VALIDATED (2026-01-27)

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

**Training Results**:
| Dimension | Mode Accuracy | Notes |
|-----------|---------------|-------|
| Temporal | **94.9%** | Excellent |
| Ontological | **92.9%** | Excellent |
| Spatial | 61.6% | Limited by instrumental tracks |
| Album (all 3) | 57.2% | Spatial is bottleneck |

**Key Findings**:
- Single-task models outperform multi-task (task interference observed)
- Spatial mode limited by "Place" albums (Yellow/Green) being instrumental - no lyrics = no text embeddings
- Early stopping on accuracy (not loss) critical for stable training

**Remaining**:
- 10.12 LangGraph integration tests

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

| Phase                      | Status               | Priority    | Completion |
|----------------------------|----------------------|-------------|------------|
| Phase 1 (Binary)           | ✅ Complete          | ✅ Critical | 100%       |
| Phase 2 (Multi-Class)      | ✅ Complete          | ✅ Critical | 100%       |
| Phase 4 (Regression)       | ✅ Complete          | ✅ Critical | 100%       |
| **Phase 3 (Multimodal)**   | **Not Started**      | 🔥 BLOCKER  | 0%         |
| Phase 10 (Production)      | Not Started          | ✅ Critical | 0%         |
| Infrastructure             | Not Started          | ✅ Critical | 0%         |
| Phase 8 (Interpretability) | ~ Partial (notebook) | Medium      | 40%        |
| Phase 5 (Temporal)         | Not Started          | DEPRECATED  | -          |
| Phase 6 (Style Transfer)   | Not Started          | DEPRECATED  | -          |
| Phase 7 (Generative)       | Not Started          | DEPRECATED  | -          |
| Phase 9 (Augmentation)     | Not Started          | Low         | 0%         |

## Critical Path to Music Production Agent

**DONE** ✅:
1. Phase 1, 2, 4 trained and validated
2. Text-only classification: 100% accuracy
3. Regression: 95% temporal, 93% ontological, 62% spatial

**IMMEDIATE** (unblock music generator):
1. **Implement Phase 3 (Multimodal Fusion)** - THE BLOCKER
   - Add audio encoder (Wav2Vec2 or CLAP)
   - Add MIDI encoder (piano roll representation)
   - Add three-pronged lyric encoder (semantic + prosodic + structural)
   - Retrain Phase 4 with multimodal inputs
   - This will fix spatial mode accuracy (currently bottlenecked by instrumental tracks)

2. **Build Evolutionary Music Generator** (new system in `app/generator/`)
   - Multi-stage generation (chords → drums → bass → melody)
   - ML model scoring at each stage
   - Pruning strategy (top-k selection)
   - Integration with existing chord progression generator

3. **Phase 10 (Production Deployment)**
   - FastAPI endpoint for music generator
   - ONNX export for inference speed
   - Batch scoring for 50+ candidates per stage

**LATER** (nice-to-have):
- Phase 8: Interpretability analysis (what makes GREEN vs RED?)
- Phase 9: Data augmentation (more training data)
- Infrastructure: Experiment tracking, hyperparameter tuning

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

### ✅ RESOLVED: Embedding Loading (Phase 4)

Implemented via `core/embedding_loader.py`:

- **PrecomputedEmbeddingLoader**: Loads pre-computed embeddings from parquet
- **DeBERTaEmbeddingEncoder**: Computes embeddings on-the-fly for new concepts
- **find_embedding_file()**: Auto-discovers embedding files in data directories

Fixed files:
- ✅ `train_phase_four.py` - Uses PrecomputedEmbeddingLoader
- ✅ `validate_concepts.py` - Uses DeBERTaEmbeddingEncoder
- ✅ `core/regression_training.py` - Uses PrecomputedEmbeddingLoader

### ✅ RESOLVED: Album Mappings

All 27 mode combinations now mapped to 8 albums (Orange, Red, Violet, Yellow, Indigo, Green, Blue, Black):
- ✅ `validate_concepts.py` - Full mapping
- ✅ `core/regression_training.py` - Complete index mapping

### Low: Hardcoded Paths

- `validate_concepts.py:538` hardcodes `/chain_artifacts` - should be configurable
- `core/regression_training.py:24` hardcodes parquet path (configurable via CONFIG dict)

## Next Steps

**Immediate** (unblock White Agent validation):
1. ~~Fix embedding loading in Phase 4 scripts~~ ✅ DONE
2. ~~Run Phase 4 training on RunPod~~ ✅ DONE (2026-01-27)
3. **Test `validate_concepts.py` with trained model** ← NEXT
4. Integrate with White Agent workflow

**To improve spatial mode accuracy**:
- Implement Phase 3 (Multimodal Fusion) to add audio embeddings
- Instrumental tracks (Yellow/Green = "Place" albums) need audio features, not text

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

*Last Updated: 2026-01-27*

**Status**: Phases 1, 2, and 4 complete. Classification achieves 100%, regression achieves 95% temporal / 93% ontological / 62% spatial. Spatial limited by instrumental tracks (no lyrics). Next: integrate with White Agent, then Phase 3 (multimodal) for audio embeddings.
