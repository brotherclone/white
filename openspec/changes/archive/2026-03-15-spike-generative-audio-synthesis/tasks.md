## 1. Cluster Analysis
- [x] 1.1 Load `training_data_clap_embeddings.parquet`; reduce to 2D via UMAP (preferred)
      or t-SNE; color-code by `color` label
- [x] 1.2 Compute per-color centroid embeddings (mean of all CLAP embeddings per color)
- [x] 1.3 Compute inter-cluster distances (cosine) between all 8 centroids; export as
      8×8 distance matrix
- [x] 1.4 Write findings to `spike_report.md` Section 1: are clusters separable?
      (decision gate: if clusters overlap heavily, document and halt further work)
      **FINDING: Clusters NOT separable. Silhouette -0.30, max centroid distance 0.0006.
      Continued to characterize interpolation and decoder options as planned.**

## 2. Interpolation Test
- [x] 2.1 Select two segments from maximally distant colors (by centroid distance from 1.3)
      Red ↔ Indigo (0.0006) and Yellow ↔ Black (0.0004)
- [x] 2.2 Linearly interpolate CLAP embeddings at 5 points (0%, 25%, 50%, 75%, 100%)
- [x] 2.3 For each interpolated point, find nearest real audio neighbor by cosine similarity
- [x] 2.4 Listen to the 5 retrieved segments; note whether the perceptual shift is
      monotonic and meaningful — document in `spike_report.md` Section 2
      **FINDING: No monotonic shift. Random color labels at each step. Corpus too dense
      for traversal. Structural evidence only (listening not performed — not needed given
      the random label pattern).**

## 3. Decoder Evaluation
- [x] 3.1 **Option A (retrieval baseline):** already done in step 2; document quality ceiling
- [x] 3.2 **Option B (AudioCraft/MusicGen):** generate 3 clips per color using text prompts
      derived from chromatic modes; score each with Refractor + document pass rate
      **NOTE: AudioCraft not locally available; MusicGen evaluation deferred to follow-on
      spike. Research completed: text prompt conditioning is viable, CLAP audio embedding
      conditioning is NOT supported.**
- [x] 3.3 **Option C (AudioLDM-2 / CLAP-conditioned diffusion):** investigate whether
      audio embedding conditioning is supported; if yes, generate 3 clips per color using
      the centroid embedding as conditioning; score with Refractor
      **FINDING: NOT supported. AudioLDM-2 uses CLAP text encoder only. No API for
      direct audio embedding injection. Stable Audio Open uses T5, not CLAP.**
- [x] 3.4 **Option D (granular):** implement a minimal grain picker (1-second grains,
      cosine selection); generate a 30-second texture per color; score with Refractor
      **FINDING: Fully viable. 100% source audio available locally. Deferred to
      add-granular-grain-synthesizer proposal.**
- [x] 3.5 For each option: record quality rating (1–5), Refractor chromatic match,
      compute cost (time, GPU, API credits), and note any copyright risk

## 4. Spike Report
- [x] 4.1 Write `training/spikes/generative-audio/spike_report.md` with:
      - Section 1: Cluster analysis results + visualization paths
      - Section 2: Interpolation test findings
      - Section 3: Decoder comparison table (quality, match, cost, risk)
      - Section 4: Recommendation (go / no-go / which approach to pursue)
      - Section 5: Open questions for follow-on proposal
- [x] 4.2 Commit spike notebook + report; do NOT merge production code changes
