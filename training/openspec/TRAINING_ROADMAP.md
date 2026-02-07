# Training Pipeline Roadmap

**Last Updated**: 2026-02-07

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

## Execution Plan (Complete Order of Events)

Two parallel tracks converge at the Evolutionary Music Generator.

### Track A: Training Pipeline (RunPod GPU)

```
LOCAL PREP (before RunPod)
══════════════════════════════════

 ✅ Pipeline bug fixes (2026-02-07):
    • Structure-based segmentation fallback for instrumentals
      (15 songs with no LRC now produce ~170 segments via manifest structure)
    • Verified: rebuild manifest DB → 1,327 tracks across all 8 colors
    • Tests: 36/36 passing

 ① add-runpod-deployment-guide          ← READ before touching RunPod
    • Region selection (US-KS-2 or US-CA-2 recommended over US-MD-1)
    • Network volume gotchas (region-locked, 64KB min alloc, no S3 in MD-1)
    • File upload strategy, execution order

 ✅ add-training-data-verification (COMPLETE 2026-02-07)
    • training/verify_extraction.py: --extract, --fidelity, --all, --color, --song, --random N
    • Coverage report by album color (audio %, MIDI %, text %)
    • Fidelity checks: 10/10 audio, 10/10 MIDI passing
    • 28 tests passing

 ✅ MIDI segmentation bug fix (2026-02-07):
    • build_training_segments_db.py was storing full MIDI files, not segment slices
    • Now uses segment_midi_file() to trim MIDI to segment time window
    • Audio and MIDI segments now aligned to same start_seconds:end_seconds


LOCAL EXTRACTION (no GPU needed)
══════════════════════════════════

 ✅ Rebuild base_manifest_db.parquet (COMPLETE 2026-02-07)
    python -m app.extractors.manifest_extractor.build_base_manifest_db
    → 1,327 tracks across all 8 colors (Indigo + Violet now included)
    → Verified: 8 unique rainbow_color values
                │
 ④ Re-run segment extraction ← IN PROGRESS
    python -m app.extractors.segment_extractor.build_training_segments_db
    → Structure fallback extracts Green, Yellow, Red instrumentals
    → Indigo/Violet segments now get rainbow_color from manifest join
    → MIDI now segmented to time window (not full file)
    → Expected: ~12,000+ segments (up from 10,544)
    → Verify: Green segments > 0, no UNLABELED segments
                │
 ⑤ Verify extraction
    python -m training.verify_extraction --all
    → Spot-check Green audio — can you hear it?
    → Spot-check MIDI segments — notes line up with audio?
    → Coverage report: all 8 colors, MIDI %, audio %
    → GATE: do NOT proceed if data looks wrong


RUNPOD EXECUTION (GPU needed)
══════════════════════════════════

 ⑥ prepare-multimodal-data (Phase 3.0)
    → DeBERTa embedding pass on all segments
    → Verify: text embeddings populated for vocal segments
    → Verify: instrumental segments have text mask = False
                │
 ⑦ add-multimodal-fusion (Phases 3.1 + 3.2) — THE BLOCKER
    → Precompute CLAP audio embeddings (512-dim, resample 44.1kHz → 48kHz)
    → Precompute piano roll MIDI embeddings (128 pitch x 256 time → CNN → 512-dim)
    → Train fusion model: [audio 512 + MIDI 512 + text 768] → MLP → regression heads
    → Target: spatial mode 62% → >85%
```

### Track B: Agent Pipeline (Local, No GPU)

```
 ✅ add-shrinkwrap-chain-artifacts (COMPLETE 2026-02-07)
    → 20 threads shrinkwrapped to shrinkwrapped/ directory
    → EVP intermediates stripped (135 files, 210MB freed from chain_artifacts)
    → manifest.yml per thread + index.yml generated
    → 47 tests passing
                │
 ✅ add-chain-result-feedback (COMPLETE 2026-02-07)
    → Key entropy: 1.67 bits (12/20 = 60% C major!)
    → BPM std dev: 8.14 (cluster around 91-96)
    → Overused phrases: "seven chromatic methodologies" (45%), "transmigration" (35%)
    → White Agent now loads constraints at workflow start, injects into all prompts
    → 27 tests passing
```

### Convergence

```
POST-TRAINING
══════════════════════════════════

 ⑧ Build Evolutionary Music Generator (not yet spec'd)
    → Uses multimodal model (Track A ⑦) as fitness function
    → Uses negative constraints (Track B ②) for diversity
    → Multi-stage: concept → chords → drums → bass → melody → human eval
                │
 ⑨ Phase 10: Production Deployment
    → FastAPI endpoint for scoring
    → ONNX export for inference speed
    → Batch scoring for 50+ candidates per stage
```

### What Changed Since Last RunPod Run

| Item | Before (2026-01-27) | After (2026-02-07) |
|------|---------------------|---------------------|
| Manifest DB | 892 tracks, 6 colors (01-06 only) | 1,327 tracks, 8 colors (01-08) |
| Green segments | 0 (LRC hard-bail) | ~102 (structure fallback) |
| Yellow instrumental segments | Missing 4 songs | ~45 new segments |
| Red instrumental segments | Missing 3 songs | ~23 new segments |
| MIDI coverage | 0% (bug) | 43.3% (fixed 2026-02-06) |
| Indigo/Violet labels | UNLABELED (3,506 segments) | Labeled (manifest DB rebuilt) |
| Total expected segments | 10,544 | ~12,000+ |

## Phase Details

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

### Phase 3: Multimodal Fusion (split into 3 changes)

#### Phase 3.0: Data Prerequisites
**Change**: `prepare-multimodal-data`
**Priority**: 🔥 IMMEDIATE — blocks Phase 3.1
**Status**: Not Started

Prepares training data for multimodal model training:
- Run DeBERTa embedding pass on new extraction (~12,000 segments, currently 0% embedded)
- Verify all 8 album colors present (Green, Indigo, Violet now in extraction)
- Verify audio/MIDI binary coverage and flag consistency
- Document remaining coverage gaps (Blue MIDI at 12%)

#### Phase 3.1 + 3.2: Audio + MIDI + Text Fusion
**Change**: `add-multimodal-fusion`
**Priority**: 🔥 THE BLOCKER — enables chromatic fitness function
**Status**: Not Started
**Design**: `design.md` complete — CLAP audio encoder, piano roll CNN, learned null embeddings, precompute-then-fuse, late concatenation

Core multimodal model combining audio, MIDI, and text:
- CLAP audio encoder (`laion/larger_clap_music`) → [batch, 512]
- Piano roll CNN MIDI encoder → [batch, 512]
- Existing DeBERTa-v3 text encoder → [batch, 768]
- Late fusion [1792] → MLP → [512] → regression heads
- Learned null embeddings for missing modalities (57% MIDI-absent)
- Modality dropout (p=0.15) during training

#### Phase 3.3 + 3.4: Prosodic & Structural Lyric Encoding
**Change**: `add-prosodic-lyric-encoding`
**Priority**: Medium — deferred until Phase 3.1/3.2 results validate approach
**Status**: Not Started
**Prerequisites**: `add-multimodal-fusion` must ship first

### Phase 4: Regression Tasks ✓ COMPLETE
**Change**: `add-regression-tasks`
**Status**: 100% complete - TRAINED AND VALIDATED (2026-01-27)

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

### Phase 5-7: DEPRECATED
- Phase 5 (Temporal Sequence): Was for concept evolution over time
- Phase 6 (Chromatic Style Transfer): Was for text-to-text style transfer
- Phase 7 (Generative Models): Was for generating text concepts

### Phase 8: Model Interpretability ~ Partially Complete
**Change**: `add-model-interpretability`
**Status**: 40% complete via notebook implementation

### Phase 9: Data Augmentation
**Change**: `add-data-augmentation`
**Priority**: Low

### Phase 10: Production Deployment
**Change**: `add-production-deployment`
**Priority**: Critical (after Phase 3)

### Infrastructure: Experiment Tracking & Optimization
**Change**: `add-infrastructure-improvements`
**Priority**: High (supports all phases)

## Current Status Summary

| Phase | Change | Status | Priority |
|-------|--------|--------|----------|
| Phase 1 (Binary) | - | ✅ Complete | Done |
| Phase 2 (Multi-Class) | - | ✅ Complete | Done |
| Phase 4 (Regression) | - | ✅ Complete | Done |
| **Pipeline Fixes** | *(this branch)* | **✅ Complete** | Done |
| **MIDI Segmentation Fix** | *(this branch)* | **✅ Complete** | Done |
| **RunPod Guide** | `add-runpod-deployment-guide` | **Spec'd** | 🔥 Read before RunPod |
| **Data Verification** | `add-training-data-verification` | **✅ Complete** | Done |
| **Phase 3.0 (Data Prep)** | `prepare-multimodal-data` | Not Started | 🔥 IMMEDIATE |
| **Phase 3.1+3.2 (Fusion)** | `add-multimodal-fusion` | Design complete | 🔥 BLOCKER |
| **Shrink-Wrap** | `add-shrinkwrap-chain-artifacts` | ✅ Complete | High (parallel track) |
| **Result Feedback** | `add-chain-result-feedback` | ✅ Complete | High (parallel track) |
| Phase 3.3+3.4 (Lyrics) | `add-prosodic-lyric-encoding` | Not Started | Medium |
| Phase 10 (Production) | `add-production-deployment` | Not Started | After Phase 3 |
| Infrastructure | `add-infrastructure-improvements` | Not Started | High |
| Phase 8 (Interpretability) | `add-model-interpretability` | ~ Partial | Medium |
| Phase 9 (Augmentation) | `add-data-augmentation` | Not Started | Low |

## Pipeline Bug Fixes (2026-02-07)

### ✅ RESOLVED: Instrumental Track Extraction
**Root cause**: `build_training_segments_db.py:218-220` hard-returned empty when no LRC file existed. All instrumental tracks (no lyrics = no LRC) produced 0 segments.

**Fix**: Added structure-based segmentation fallback. When no LRC file exists but `manifest.structure` has sections, segments are created from structure boundaries with the same max-length splitting logic.

**Impact**: 15 previously-skipped songs now produce segments:
- 8 Green songs → ~102 segments
- 4 Yellow songs (04_03, 04_06, 04_09, 04_10) → ~45 segments
- 3 Red songs (02_06, 02_10, 02_12) → ~23 segments

**Files changed**:
- `app/util/timestamp_audio_extractor.py` — Added `create_segment_specs_from_structure()`
- `app/extractors/segment_extractor/build_training_segments_db.py` — LRC → structure fallback
- `tests/util/test_timestamp_pipeline.py` — 6 new tests (36/36 passing)

### ✅ RESOLVED: Missing Album Labels (Indigo/Violet)
**Root cause**: `base_manifest_db.parquet` was built before Indigo (07) and Violet (08) albums were staged in `staged_raw_material/`. Not a code bug — just needs a rebuild.

**Fix**: Re-run `build_base_manifest_db.py` before next extraction.

**Impact**: 892 → 1,327 tracks. All 8 colors present. 3,506 previously-UNLABELED segments (Indigo + Violet) will get `rainbow_color` from the manifest join.

### ✅ RESOLVED: MIDI Segment Detection (2026-02-06)
`absolute_tick` was accumulating across tracks in Type 1 MIDI files instead of resetting per track. Fixed in `build_training_segments_db.py:124`. MIDI coverage went from 0% to 43.3%.

### ✅ RESOLVED: MIDI File Path (2026-02-06)
Was reconstructing path from `staged_material_dir / song_id / midi_file` but `midi_file` is already an absolute path. Changed to `Path(row["midi_file"])`.

## Required Fixes Before Production

### ✅ RESOLVED: Embedding Loading (Phase 4)
Implemented via `core/embedding_loader.py`.

### ✅ RESOLVED: Album Mappings
All 27 mode combinations now mapped to 8 albums.

### Low: Hardcoded Paths
- `validate_concepts.py:538` hardcodes `/chain_artifacts` - should be configurable
- `core/regression_training.py:24` hardcodes parquet path (configurable via CONFIG dict)

## Resources

- OpenSpec changes (training): `training/openspec/changes/`
- OpenSpec changes (agent): `openspec/changes/`
- Design decisions: `training/openspec/changes/add-multimodal-fusion/design.md`
- Project context: `training/openspec/project.md`

---

*Last Updated: 2026-02-07*

**Status**: Phases 1, 2, 4 complete. Pipeline bug fixes ready to commit (structure fallback for instrumentals, manifest DB rebuild for Indigo/Violet). Four new specs created: RunPod deployment guide, training data verification, chain artifact shrink-wrap, chain result feedback. Next: commit fixes, implement verification tool, then RunPod run for Phase 3.0 → 3.1/3.2.
