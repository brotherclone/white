## Context

Validation of the chunked mix scoring (`update-mix-scoring-chunked`) showed that confidence
remained ~0.10 across all 78 labeled songs regardless of chunking. The CLAP embeddings of
finished productions are out-of-distribution for the base Refractor model, which was trained
on short, isolated catalog segments. A calibration layer trained on the `_main.wav` corpus
bridges this gap without requiring a full model retrain.

Training set: 78 `_main.wav` files, 8 colors (R/O/Y/G/B/I/V/Z), 6 missing main.wav (skipped).
With ~43 chunks per song at 30s / 5s stride, augmented training sees ~3,354 chunk-level
examples per epoch.

## Goals / Non-Goals

- **Goals**: confident, accurate top-1 color prediction (≥70%) on full-mix audio;
  model small enough to train in minutes on CPU; ONNX export for runtime consistency;
  graceful fallback when model file absent
- **Non-Goals**: fine-tuning or retraining CLAP; scoring stems individually; real-time
  scoring; handling completely unknown songs at inference time without a concept embedding

## Decisions

- **Decision**: Calibration head over CLAP (frozen), not end-to-end fine-tune
  - 78 songs is too few to fine-tune CLAP (330M params) without catastrophic forgetting
  - CLAP already extracts rich audio features; we just need to remap them to our color space
  - Training time: minutes on CPU vs. hours on GPU

- **Decision**: Three independent softmax regression heads, one per axis
  - Matches the base Refractor output contract (temporal/spatial/ontological dicts)
  - Loss: MSE against `CHROMATIC_TARGETS` soft distributions (e.g. Red temporal = [0.8, 0.1, 0.1])
  - Independent heads avoid task interference (observed in Phase 4 training history)

- **Decision**: Random-chunk augmentation during training, not mean-pooled song embeddings
  - Mean-pooling gives 78 training samples — too few, high variance
  - Random-chunk sampling gives ~3,300 samples per epoch with natural variation
  - All chunks of a song share the same color label (weak supervision is fine here)
  - At inference time we still mean-pool chunks for a stable embedding

- **Decision**: Concept embedding as optional second input (concatenated before first hidden layer)
  - All 78 songs have concept text; the concept signal measurably shifts distributions
    (confirmed in Phase 5 sounds-like training)
  - At inference: concept always available from `score_mix`'s proposal load
  - If absent (future unknown songs): zero-pad the concept slot

- **Decision**: Architecture `(512 [+ 768]) → 256 → 128 → 3 × Linear(3) + softmax`
  - Input dim: 512 (CLAP) or 1280 (CLAP + concept)
  - Two hidden layers with ReLU + Dropout(0.3) to regularize on small dataset
  - Three separate output heads to avoid cross-task gradient interference
  - ONNX export: three outputs `temporal_logits`, `spatial_logits`, `ontological_logits`

- **Decision**: Training split — stratified 80/20 by color, not leave-one-out
  - Leave-one-out is expensive (78 training runs) and hard to early-stop
  - Stratified split ensures all 8 colors appear in both train and val
  - With only ~10 songs per color, val set is 1–2 songs per color

- **Decision**: `refractor_cdm.onnx` is opt-in at runtime (fallback to base model)
  - `score_mix` checks whether the file exists before loading it
  - Allows shipping the code before the model is trained and committed
  - `--cdm-onnx-path ""` explicitly disables it for debugging

## Risks / Trade-offs

- **78-song dataset is tiny**: 1–2 songs per color in validation set; per-color accuracy
  numbers will have high variance. Mitigated by reporting per-color N alongside accuracy.
- **Label quality**: `CHROMATIC_TARGETS` soft labels assume one dominant mode per axis;
  some songs may genuinely be "between" colors. Mitigated by using soft [0.8/0.1/0.1]
  targets rather than one-hot.
- **Data leakage risk**: if concept embedding encodes too much "I am Violet" signal,
  the model could shortcut audio entirely. Monitor by evaluating audio-only vs.
  audio+concept accuracy separately.

## Open Questions

- Should we include `sounds_like` embeddings as a third input (as in Phase 5)?
  Low priority — sounds_like already lives in the base Refractor; the calibration head
  focuses on the audio gap.
- Should the training script also produce a `refractor_cdm_deberta.onnx` (concept-only
  path as a sanity check)?
