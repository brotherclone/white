# Training Pipeline Roadmap

**Last Updated**: 2026-02-12

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

 ✅ Rebuild base_manifest_db.parquet (COMPLETE 2026-02-10)
    python -m app.extractors.manifest_extractor.build_base_manifest_db
    → 1,327 tracks across all 8 colors (Indigo + Violet now included)
    → Verified: 8 unique rainbow_color values
    → Rebuilt 2026-02-10 after YAML edits introduced 4 duplicate track IDs
                │
 ✅ Re-run segment extraction (COMPLETE 2026-02-10)
    python -m app.extractors.segment_extractor.build_training_segments_db
    → 11,605 segments across 83 songs (up from 10,544)
    → All 8 colors: Black 1748, Red 1474, Orange 1731, Yellow 656,
      Green 393, Blue 2097, Indigo 1406, Violet 2100
    → Audio: 85.4%, MIDI: 44.3%
    → Zero UNLABELED segments, zero metadata duplicates
                │
 ✅ Verify extraction (COMPLETE 2026-02-10)
    python -m training.verify_extraction --all
    → Audio fidelity: 10/10 passed
    → MIDI fidelity: 10/10 passed
    → RESULT: PASS — All checks passed
                │
 ✅ Publish to HuggingFace (COMPLETE 2026-02-10)
    python training/hf_dataset_prep.py --push --public --include-embeddings
    → earthlyframes/white-training-data v0.2.0
    → 3 configs: base_manifest, training_full, training_segments
    → Media parquet uploaded (15.3 GB): audio waveforms + MIDI binaries
    → Dataset card with coverage tables, usage docs, CI License
    → Public: https://huggingface.co/datasets/earthlyframes/white-training-data


MODAL GPU EXECUTION (migrated from RunPod 2026-02-12)
══════════════════════════════════

 ✅ prepare-multimodal-data (Phase 3.0) — COMPLETE 2026-02-12
    → DeBERTa-v3-base embedding pass: 11,605 concept + 10,764 lyric embeddings (768-dim)
    → 841 instrumental segments → zero vectors + has_lyric_embedding=False
    → Output: training_data_with_embeddings.parquet (4.5 MB)
    → Executed on Modal (A10G), ~3 min
                │
 ✅ CLAP audio embedding precomputation (Phase 3.1 partial) — COMPLETE 2026-02-12
    → CLAP (laion/larger_clap_music) audio embeddings: 9,981/11,692 segments (512-dim)
    → Manual resample 44.1kHz → 48kHz via librosa
    → 1,711 segments without audio → zero vectors + has_audio_embedding=False
    → Output: training_data_clap_embeddings.parquet (20.5 MB)
    → Executed on Modal (A10G), ~30 min. Media parquet cached in Modal Volume.
                │
 ✅ add-multimodal-fusion (Phases 3.1 + 3.2) — COMPLETE 2026-02-13
    → Piano roll preprocessing: 11,692 segments → [128,256] matrices (5,231 with MIDI)
    → PianoRollEncoder CNN (1.1M params, unfrozen) + fusion MLP (3.2M params)
    → Input: [audio 512 + MIDI 512 + concept 768 + lyric 768] = 2560-dim
    → Learned null embeddings + modality dropout (p=0.15)
    → Results: temporal 90%, spatial 93%, ontological 91%
    → Spatial mode: 62% → 93% (target was >85%) ✓
    → Model: training/data/fusion_model.pt (16.4 MB)
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

 ✅ ONNX Export + ChromaticScorer (COMPLETE 2026-02-14)
    → ONNX export: training/data/fusion_model.onnx (16 MB, CPU inference)
    → ChromaticScorer class: training/chromatic_scorer.py
    → score(midi_bytes, concept_emb=...) → {temporal, spatial, ontological, confidence}
    → score_batch(candidates, concept_emb=...) → ranked list for 50+ candidates
    → Lazy-loaded DeBERTa + CLAP encoders (MIDI-only scoring never loads CLAP)
    → Model definition extracted: training/models/multimodal_fusion.py
                │
 ✅ Music Production Pipeline — Chord Phase (COMPLETE 2026-02-14)
    → OpenSpec: openspec/changes/add-music-production-pipeline/
    → Pipeline: app/generators/midi/chord_pipeline.py
    → Reads shrinkwrapped song proposal → Markov chord generation → ChromaticScorer scoring
    → Composite scoring (30% theory + 70% chromatic) → top-N MIDI files + review.yml
    → Human labels candidates (verse/chorus/bridge) in review.yml → promote to approved/
    → Promotion tool: app/generators/midi/promote_chords.py
                │
 ⑧ Next phases: drums, bass, melody+lyrics, assembly (not yet spec'd)
    → Each follows same pattern: generate → score → human gate → approve
```

### What Changed Since Last RunPod Run

| Item                 | Before (2026-01-27)                  | After (2026-02-10)                    |
|----------------------|--------------------------------------|---------------------------------------|
| Manifest DB          | 892 tracks, 6 colors (01-06 only)    | 1,327 tracks, 8 colors (01-08)        |
| Green segments       | 0 (LRC hard-bail)                    | 393 (structure fallback)              |
| Yellow segments      | Missing 4 instrumental songs         | 656 (structure fallback)              |
| Red segments         | Missing 3 instrumental songs         | 1,474 (structure fallback)            |
| MIDI coverage        | 0% (bug)                             | 44.3% (5,145/11,605)                  |
| Audio coverage       | ~85%                                 | 85.4% (9,907/11,605)                  |
| Indigo/Violet labels | UNLABELED (3,506 segments)           | Indigo 1,406 + Violet 2,100           |
| Total segments       | 10,544                               | 11,605                                |
| Metadata duplicates  | 87 (stale manifest after YAML edits) | 0 (rebuilt 2026-02-10)                |
| HuggingFace          | Not published                        | v0.2.0 public, 15.3 GB media included |

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
**Priority**: ✅ COMPLETE
**Status**: Complete (2026-02-12, Modal GPU)

Prepares training data for multimodal model training:
- ✅ All 8 album colors present and verified (11,605 segments, 2026-02-10)
- ✅ Audio/MIDI binary coverage verified (85.4% audio, 44.3% MIDI)
- ✅ Published to HuggingFace (earthlyframes/white-training-data v0.2.0)
- ✅ DeBERTa embedding pass: 11,605 concept + 10,764 lyric embeddings (768-dim)
- ✅ CLAP audio embedding pass: 9,981 audio embeddings (512-dim)
- Known gap: Blue MIDI at 12%, Yellow/Green instrumental (no lyrics)

#### Phase 3.1 + 3.2: Audio + MIDI + Text Fusion
**Change**: `add-multimodal-fusion`
**Priority**: ✅ COMPLETE (2026-02-13)
**Status**: Complete — spatial mode 62% → 93%
**Design**: `design.md` — CLAP audio encoder, piano roll CNN, learned null embeddings, joint CNN training (Option A), late concatenation

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

### Phase 10: ONNX Export + ChromaticScorer
**Change**: `add-production-deployment` (revised — no API, direct import)
**Priority**: ✅ COMPLETE (2026-02-14)
**Status**: Complete

Revised scope (2026-02-13): No FastAPI endpoint needed. The Evolutionary Music Generator
calls the scorer directly in-process. Scope is now:
- ONNX export of `MultimodalFusionModel` (4.3M params → fast CPU inference)
- `ChromaticScorer` class that loads ONNX model + handles piano roll conversion
- Interface: `score(midi_bytes, audio_waveform, concept_text) → {temporal, spatial, ontological, confidence}`
- Batch interface: `score_batch(candidates) → ranked list` for 50+ candidates per stage
- DeBERTa + CLAP encoders still needed at inference for new text/audio — either precompute or lazy-load

### Infrastructure: Experiment Tracking & Optimization
**Change**: `add-infrastructure-improvements`
**Priority**: High (supports all phases)

## Current Status Summary

| Phase                         | Change                            | Status                           | Priority              |
|-------------------------------|-----------------------------------|----------------------------------|-----------------------|
| Phase 1 (Binary)              | -                                 | ✅ Complete                       | Done                  |
| Phase 2 (Multi-Class)         | -                                 | ✅ Complete                       | Done                  |
| Phase 4 (Regression)          | -                                 | ✅ Complete                       | Done                  |
| **Pipeline Fixes**            | *(this branch)*                   | **✅ Complete**                   | Done                  |
| **MIDI Segmentation Fix**     | *(this branch)*                   | **✅ Complete**                   | Done                  |
| **Extraction + Verification** | -                                 | **✅ Complete** (2026-02-10)      | Done                  |
| **HuggingFace Publish**       | -                                 | **✅ v0.2.0 public** (2026-02-10) | Done                  |
| **RunPod Guide**              | `add-runpod-deployment-guide`     | **Spec'd**                       | 🔥 Read before RunPod |
| **Data Verification**         | `add-training-data-verification`  | **✅ Complete**                   | Done                  |
| **Phase 3.0 (Data Prep)**     | `prepare-multimodal-data`         | **✅ Complete** (2026-02-12)      | Done                  |
| **Phase 3.1+3.2 (Fusion)**    | `add-multimodal-fusion`           | **✅ Complete** (2026-02-13)      | Done                  |
| **Shrink-Wrap**               | `add-shrinkwrap-chain-artifacts`  | ✅ Complete                       | Done                  |
| **Result Feedback**           | `add-chain-result-feedback`       | ✅ Complete                       | Done                  |
| Phase 3.3+3.4 (Lyrics)        | `add-prosodic-lyric-encoding`     | Not Started                      | Medium                |
| Phase 10 (ONNX+Scorer)        | `add-production-deployment`       | **✅ Complete** (2026-02-14)     | Done                  |
| Infrastructure                | `add-infrastructure-improvements` | Not Started                      | High                  |
| Phase 8 (Interpretability)    | `add-model-interpretability`      | ~ Partial                        | Medium                |
| Phase 9 (Augmentation)        | `add-data-augmentation`           | Not Started                      | Low                   |

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

### ✅ RESOLVED: Metadata Duplicates from Stale Manifest (2026-02-10)
**Root cause**: YAML manifests for 3 songs (01_01, 03_03, 08_08) were edited after the manifest DB was built (YAMLs modified Feb 9, manifest built Feb 7). The stale manifest had incorrect `track_id` values for 4 tracks, producing 87 duplicate rows when segments joined to metadata.

**Fix**: Rebuild `base_manifest_db.parquet` after YAML edits. Zero duplicate composite keys after rebuild.

**Impact**: Metadata rows now exactly match segment rows (11,605 = 11,605). Previously 11,692 metadata vs 11,605 segments.

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

*Last Updated: 2026-02-12*

**Status**: Phases 1, 2, 3, 4, 10 complete. Extraction pipeline fully operational: 11,605 segments, all 8 colors, 85.4% audio, 44.3% MIDI. Published to HuggingFace as `earthlyframes/white-training-data` v0.2.0 (public, 15.3 GB media included). **Phase 3 complete** (2026-02-13): Multimodal fusion model achieves 90% temporal, 93% spatial, 91% ontological. **Phase 10 complete** (2026-02-14): ChromaticScorer class with ONNX inference, batch scoring for 50+ candidates, lazy-loaded DeBERTa/CLAP encoders. **Next: Step 8 (Evolutionary Music Generator).** GPU execution on Modal (serverless).
