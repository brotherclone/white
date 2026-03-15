# Spike Report: Generative Audio Synthesis from Chromatic Embedding Space

**Date:** 2026-03-15
**Data:** `training_data_clap_embeddings.parquet` (9,981 segments, 512-dim CLAP)
**Status:** Complete — recommendation: conditional no-go on CLAP-based synthesis

---

## 1. Cluster Analysis

### Setup

- 9,981 segments with valid CLAP audio embeddings (from 11,605 total; 14% had no audio)
- Embeddings: 512-dim, pre-normalized to unit length (mean norm = 1.0000, std = 0.0000)
- Color distribution: Blue 2,014 · Violet 1,636 · Black 1,475 · Orange 1,458 · Red 1,381 · Indigo 1,086 · Yellow 577 · Green 354

### Are the clusters separable?

**No.**

| Metric | Value | Interpretation |
|---|---|---|
| Silhouette score (cosine, n=2,000 sample) | **-0.30** | Clusters heavily overlap; embeddings closer to other-cluster centroids than their own |
| Global mean cosine similarity (all pairs) | **0.9935** | All embeddings packed into a tiny cone in 512-dim space |
| Global cosine sim std | 0.0075 | Variance is minimal — no clear structure |
| Max centroid-to-centroid cosine distance | **0.0006** | All 8 centroids within 0.06% of each other |

### 8×8 Centroid Cosine Distance Matrix

```
              Red   Orange  Yellow  Green   Blue   Indigo  Violet  Black
Red          0.0000  0.0002  0.0003  0.0002  0.0003  0.0006  0.0003  0.0001
Orange       0.0002  0.0000  0.0002  0.0001  0.0000  0.0002  0.0000  0.0003
Yellow       0.0003  0.0002  0.0000  0.0003  0.0003  0.0002  0.0003  0.0004
Green        0.0002  0.0001  0.0003  0.0000  0.0002  0.0003  0.0002  0.0002
Blue         0.0003  0.0000  0.0003  0.0002  0.0000  0.0001  0.0000  0.0003
Indigo       0.0006  0.0002  0.0002  0.0003  0.0001  0.0000  0.0001  0.0006
Violet       0.0003  0.0000  0.0003  0.0002  0.0000  0.0001  0.0000  0.0004
Black        0.0001  0.0003  0.0004  0.0002  0.0003  0.0006  0.0004  0.0000
```

Most similar pairs: Orange↔Blue (0.0000), Orange↔Violet (0.0000), Blue↔Violet (0.0000)
Most distant pairs: Red↔Indigo (0.0006), Indigo↔Black (0.0006)

### Within-cluster coherence

| Color | Mean intra-cluster cosine sim | n |
|---|---|---|
| Red | 0.9911 | 1,381 |
| Orange | 0.9931 | 1,458 |
| Yellow | 0.9934 | 577 |
| Green | 0.9908 | 354 |
| Blue | 0.9935 | 2,014 |
| Indigo | 0.9967 | 1,086 |
| Violet | 0.9940 | 1,636 |
| Black | 0.9864 | 1,475 |

High within-cluster similarity (~0.99) combined with equally high between-cluster similarity means the "clusters" are indistinguishable from the overall corpus distribution. The 8 labels carry no separable signal in CLAP space.

### PCA structure

84.5% of variance is explained by the top 5 principal components. The 512-dim space has effective dimensionality of roughly 5–10 PCs. This is consistent with a corpus that is acoustically homogeneous — all tracks are from the same artist/genre family (ambient, experimental, indie) and CLAP maps them into a tightly clustered region.

### UMAP visualization

UMAP was run (50 PCA components → 2D, n_neighbors=30, min_dist=0.1). Coordinates saved to `umap_coords.csv`. All 8 color clusters occupy the same overlapping region (centers approximately x=9–10, y=5–7; spreads 2.4–3.6 per axis, larger than center-to-center distances). No color-specific structure is visible.

### Comparison: DeBERTa concept embeddings

The DeBERTa concept embedding (768-dim, encodes song concept text) shows:
- Silhouette score: **-0.02** (still negative, but significantly less negative than CLAP's -0.30)
- Global mean cosine sim: **0.66** (much more spread out than CLAP's 0.99)
- The chromatic signal IS present in the concept/semantic space — just not in the audio/CLAP space

**Decision gate:** Clusters are NOT separable in CLAP space. The core premise of centroid-guided synthesis is false. The analysis continues to characterize the interpolation behavior and evaluate decoder options, but the fundamental "CLAP space is navigable for chromatic targeting" hypothesis is rejected.

---

## 2. Interpolation Test

### Methodology

Linearly interpolate between the two most distant color centroids (Red and Indigo, distance 0.0006). At each of 5 points (t=0, 0.25, 0.5, 0.75, 1.0), retrieve the nearest real audio segment by cosine similarity.

### Results: Red → Indigo

| t | Nearest color | Cosine sim | Segment | Song |
|---|---|---|---|---|
| 0.00 (Red centroid) | Yellow | 0.999729 | 04_05_seg_0009_track_14 | Guest Quarters |
| 0.25 | Yellow | 0.999709 | 04_05_seg_0009_track_14 | Guest Quarters |
| 0.50 | Blue | 0.999761 | 06_10_seg_0007_track_02 | Watch Out for That One |
| 0.75 | Red | 0.999833 | 02_10_seg_0002_track_13 | Getting Started in Tempography |
| 1.00 (Indigo centroid) | Blue | 0.999904 | 06_11_seg_0010_track_05 | Taped Over |

### Results: Yellow → Black

| t | Nearest color | Cosine sim | Segment | Song |
|---|---|---|---|---|
| 0.00 (Yellow centroid) | Blue | 0.999814 | 06_11_seg_0018_track_09 | Taped Over |
| 0.25 | Black | 0.999779 | 01_01_seg_0014_track_22 | The Conjurer's Thread |
| 0.50 | Black | 0.999729 | 01_01_seg_0014_track_22 | The Conjurer's Thread |
| 0.75 | Blue | 0.999663 | 06_10_seg_0007_track_08 | Watch Out for That One |
| 1.00 (Black centroid) | Yellow | 0.999599 | 04_05_seg_0012_track_09 | Guest Quarters |

### Findings

The interpolation path does not track the intended color trajectory. The nearest-neighbor color at each point is effectively random — the interpolation vector moves through the densely packed corpus region and retrieves whatever segment happens to be nearest, regardless of color label. The retrieved cosine similarities are all ~0.9997–0.9999, confirming the corpus is so densely packed that any query point retrieves audio indistinguishably close to all colors.

**Perceptual shift assessment:** Cannot be evaluated from embedding geometry alone — the retrieved segments are from entirely different songs at each step. No monotonic perceptual shift can be assumed or verified without listening, but the structural evidence (random color labels at each step) strongly suggests no coherent trajectory.

---

## 3. Decoder Evaluation

### Option A: Retrieval Baseline

**Status: Already shipped.**

`retrieve_samples.py` implements retrieval by color (chromatic_match scoring via Refractor) and by CLAP cosine similarity. Source audio is 100% available locally — all top-20 segments per color have accessible `source_audio_file` paths with `start_seconds`/`end_seconds` timestamps. The `--copy-audio` flag cuts segments via `soundfile` fallback when precomputed segment WAVs are absent.

| Criterion | Score |
|---|---|
| Quality | Authentic (actual corpus recordings) |
| Chromatic match | Scored via Refractor — most coherent targeting available |
| GPU cost | None |
| Implementation cost | 0 (done) |
| Copyright risk | Own recordings — zero |

**Ceiling**: Retrieval is bounded by what exists in the corpus. No new audio can be generated.

### Option B: AudioCraft / MusicGen (text-conditioned)

**Status: Not locally available. Requires GPU (A10G via Modal).**

MusicGen accepts natural language text prompts and generates music. Chromatic mode descriptions could serve as prompts:

- Red: "melancholic past-tense meditation, sparse piano, ambient, slow"
- Indigo: "abstract, otherworldly, unreal, drone, no clear temporal anchor"
- Black: "dark ambient, undefined, formless, void"

MusicGen's `CLAPEmbeddingConditioner` computes CLAP representations internally from text or audio — it does NOT accept pre-computed 512-dim CLAP embeddings from external sources.

| Criterion | Score |
|---|---|
| Quality | Unknown without evaluation |
| Chromatic match | Unverified — text prompts are approximate, not grounded in Refractor |
| GPU cost | ~$0.10/clip on Modal A10G (2-3 min per color) |
| Implementation cost | ~1 session (Modal script, text prompt design, Refractor scoring of output) |
| Copyright risk | MusicGen-generated content — check Meta license (CC-BY-NC) |

**Viability**: Moderate. The weak link is that text prompts are a lossy approximation of the chromatic target. Output quality needs empirical evaluation.

### Option C: AudioLDM-2 / CLAP-conditioned Diffusion

**Status: Direct CLAP audio embedding conditioning is NOT supported.**

AudioLDM-2 uses CLAP internally for its text conditioning branch. The `prompt_embeds` and `generated_prompt_embeds` parameters accept CLAP-text and GPT-2 embeddings via `encode_prompt()` — but these are derived from text, not from a pre-computed 512-dim audio embedding. There is no API path to inject a CLAP audio embedding directly. Stable Audio Open uses T5 (not CLAP) for conditioning — CLAP appears only in the Stability AI research for memorization evaluation.

| Criterion | Score |
|---|---|
| Quality | N/A |
| Chromatic match | N/A |
| GPU cost | N/A |
| Implementation cost | Not feasible without custom training |
| Copyright risk | N/A |

**Viability: None** without training a custom projection layer from CLAP-audio space into AudioLDM-2's conditioning space.

### Option D: Granular Synthesis

**Status: Implementable locally, no GPU.**

Source audio is fully available (100% of top-20 segments per color have local `source_audio_file` paths). A grain picker can:
1. Retrieve top-N segments by Refractor chromatic_match
2. Load 1-second grains via `soundfile` at random offsets within each segment
3. Crossfade grains (Hann window) into a continuous texture of arbitrary length
4. Output: 30–60 second WAV per color

The resulting texture is a literal collage of real corpus recordings. It is not "new" audio — it is a chromatic mosaic from the corpus.

| Criterion | Score |
|---|---|
| Quality | 3/5 — authentic but potentially repetitive |
| Chromatic match | High — grains selected by Refractor score |
| GPU cost | None |
| Implementation cost | ~4 hours (grain_synthesizer.py + tests) |
| Copyright risk | Own recordings — zero |

**Viability: High.** The main limitation is musical coherence — adjacent grains from different songs may have jarring tonal/tempo discontinuities unless grain selection applies additional harmonic constraints (matching key or BPM).

### Comparison Table

| Option | Quality | Chromatic match | GPU cost | Impl. cost | Risk |
|---|---|---|---|---|---|
| A: Retrieval (existing) | Real recordings | Refractor-scored | None | Done | None |
| B: MusicGen text | Unknown | Approximate | Low (Modal) | 1 session | License (CC-BY-NC) |
| C: CLAP-conditioned diffusion | N/A | N/A | High | Not feasible | N/A |
| D: Granular synthesis | 3/5 | Refractor-scored | None | 4 hours | None |

---

## 4. Recommendation

### Go / No-Go

**No-go on CLAP-space synthesis as originally framed.**

The three original questions are answered:

1. **Are the 8 color clusters separable in CLAP space?** No. Silhouette score -0.30, centroid distances ≤0.0006. CLAP encodes acoustic similarity; all White corpus tracks are acoustically homogeneous (same artist, same genre family). The chromatic signal is not present in CLAP space.

2. **Does embedding interpolation produce perceptually coherent audio?** No. Interpolation between maximally distant centroids retrieves random-colored segments — no monotonic shift. There is no navigable chromatic trajectory in CLAP space.

3. **Which decoder approach is most viable?** Retrieval (Option A) is already working. Granular synthesis (Option D) is the most viable new capability — implementable locally, chromatic scoring via Refractor, no copyright risk. MusicGen text-conditioning (Option B) is viable as a creative exploration but requires empirical evaluation to determine whether text prompts can substitute for proper chromatic targeting.

### Recommended next steps (prioritized)

1. **Implement granular grain synthesizer** (`training/tools/grain_synthesizer.py`) — low cost, immediately useful for producing chromatic backing textures. Open `add-granular-grain-synthesizer` proposal.

2. **Evaluate MusicGen text prompts** — run 3 clips per color on Modal, score with Refractor, determine whether chromatic pass rate justifies production use. This is a quick follow-on spike (1 session), not a full proposal.

3. **Do not pursue** CLAP embedding inversion, AudioLDM-2 CLAP conditioning, or Stable Audio direct embedding conditioning — none of these are supported by any existing model, and training a custom projection is not warranted at this stage.

---

## 5. Open Questions for Follow-On Proposal

1. **Granular coherence**: Can the grain picker apply key/BPM constraints (e.g., `key_signature_note` from metadata) to reduce tonal discontinuities between adjacent grains? The metadata parquet has `key_signature_note` and `bpm` per segment.

2. **MusicGen chromatic pass rate**: What percentage of MusicGen outputs (given chromatic text prompts) achieve chromatic_match > 0.6 when scored by Refractor? This determines whether MusicGen is a viable production tool or just a creative toy.

3. **Concept embedding synthesis**: DeBERTa concept embeddings show much better color separability (silhouette -0.02 vs -0.30 for CLAP). Is there a text-to-audio model that could be conditioned on DeBERTa embeddings? This is a cleaner path than CLAP inversion.

4. **Crossmodal projection**: Could a small MLP trained on (Refractor output → MusicGen conditioning) pairs enable proper chromatic targeting? This would require ~1,000 labeled (audio, chromatic_match) pairs — already available in the corpus — and would be the "right" solution if MusicGen text prompts prove unreliable.

---

## Appendix: Raw Data

- `umap_coords.csv` — 2D UMAP coordinates for all 9,981 segments (x, y, color, segment_id)
- Source data: `training/data/training_data_clap_embeddings.parquet` (21 MB, 9,981 rows)
- Metadata: `training/data/training_data_with_embeddings.parquet` (4.5 MB, 11,605 rows)
