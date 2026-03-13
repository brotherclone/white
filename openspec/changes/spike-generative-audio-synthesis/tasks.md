## 1. Cluster Analysis
- [ ] 1.1 Load `training_data_clap_embeddings.parquet`; reduce to 2D via UMAP (preferred)
      or t-SNE; color-code by `color` label
- [ ] 1.2 Compute per-color centroid embeddings (mean of all CLAP embeddings per color)
- [ ] 1.3 Compute inter-cluster distances (cosine) between all 8 centroids; export as
      8×8 distance matrix
- [ ] 1.4 Write findings to `spike_report.md` Section 1: are clusters separable?
      (decision gate: if clusters overlap heavily, document and halt further work)

## 2. Interpolation Test
- [ ] 2.1 Select two segments from maximally distant colors (by centroid distance from 1.3)
- [ ] 2.2 Linearly interpolate CLAP embeddings at 5 points (0%, 25%, 50%, 75%, 100%)
- [ ] 2.3 For each interpolated point, find nearest real audio neighbor by cosine similarity
- [ ] 2.4 Listen to the 5 retrieved segments; note whether the perceptual shift is
      monotonic and meaningful — document in `spike_report.md` Section 2

## 3. Decoder Evaluation
- [ ] 3.1 **Option A (retrieval baseline):** already done in step 2; document quality ceiling
- [ ] 3.2 **Option B (AudioCraft/MusicGen):** generate 3 clips per color using text prompts
      derived from chromatic modes; score each with Refractor + document pass rate
- [ ] 3.3 **Option C (AudioLDM-2 / CLAP-conditioned diffusion):** investigate whether
      audio embedding conditioning is supported; if yes, generate 3 clips per color using
      the centroid embedding as conditioning; score with Refractor
- [ ] 3.4 **Option D (granular):** implement a minimal grain picker (1-second grains,
      cosine selection); generate a 30-second texture per color; score with Refractor
- [ ] 3.5 For each option: record quality rating (1–5), Refractor chromatic match,
      compute cost (time, GPU, API credits), and note any copyright risk

## 4. Spike Report
- [ ] 4.1 Write `training/spikes/generative-audio/spike_report.md` with:
      - Section 1: Cluster analysis results + visualization paths
      - Section 2: Interpolation test findings
      - Section 3: Decoder comparison table (quality, match, cost, risk)
      - Section 4: Recommendation (go / no-go / which approach to pursue)
      - Section 5: Open questions for follow-on proposal
- [ ] 4.2 Commit spike notebook + report; do NOT merge production code changes
