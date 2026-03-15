## Context

The CLAP model (trained by LAION) maps audio → 512-dim embedding space. The White training
corpus has 9,907 segments across 8 colors with precomputed CLAP embeddings. The Refractor
model is already trained to classify and regress chromatic properties from these embeddings.

The hypothesis: if color clusters are separable in CLAP space, we can navigate the space
toward a target color and either retrieve the nearest real audio or decode a new audio
signal from the target embedding.

**Constraints:**
- All audio is copyrighted source material — generated synthesis must not reproduce
  identifiable passages
- Local inference is strongly preferred (GPU-optional for prototyping)
- Any approach must integrate with the existing `score_mix.py` pipeline (close the loop:
  synthesize → score → iterate)

---

## Goals / Non-Goals

Goals:
- Determine whether CLAP embeddings for the 8 colors form visually separable clusters
- Test linear interpolation as a navigation mechanism
- Evaluate 2–3 candidate decoders with honest quality/cost assessment
- Produce a written recommendation for or against building a production feature

Non-Goals:
- Production-ready audio output (this spike may produce noisy/artifact-heavy audio)
- Fine-tuning any model
- Integration with Logic Pro or ACE Studio

---

## Candidate Approaches

### A. Nearest-Neighbor Retrieval (no decoder needed)
Navigate to a target position in CLAP space by constructing a synthetic target embedding
(weighted centroid of color cluster), then retrieve the real segment with highest cosine
similarity. Essentially what `retrieve_samples.py` does.

**Pro:** No decoder, uses existing infrastructure, always produces real audio.
**Con:** Not generative — limited to what's in the corpus. Cannot synthesize novel textures.

### B. AudioCraft (Meta, open-weights)
MusicGen / AudioGen can be conditioned on text descriptions. Could compose prompts from
the chromatic target (e.g., "ambient texture, future tense, person, imagined") and score
outputs with Refractor. Not embedding-to-audio but prompt-to-audio.

**Pro:** High quality, controllable, runs locally on GPU.
**Con:** Text prompting is indirect — no direct CLAP embedding conditioning. Requires A10G
or better for inference.

### C. Stable Audio / AudioLDM-2 (latent diffusion)
Some latent diffusion audio models accept CLAP embeddings as conditioning directly
(AudioLDM-2 uses CLAP text + audio conditioning). Could condition on a target color's
CLAP centroid.

**Pro:** Direct CLAP conditioning possible; most faithful to the "navigate embedding space"
hypothesis.
**Con:** Requires investigation of which models support audio embedding (not just text
embedding) as input. API costs if not self-hosted.

### D. Granular Synthesis + Chromatic Navigation
Slice real segments into grains, parameterize grain selection by cosine proximity to a
target CLAP embedding, reassemble. Stays within the corpus; no external model needed.

**Pro:** Fully local, uses real audio material, produces genuinely novel textures.
**Con:** CLAP inference at grain level is expensive; output may be incoherent without
careful grain size tuning.

---

## Risks / Trade-offs

- **Risk:** Color clusters are not separable in CLAP space (all 8 colors map to overlapping
  regions). → Mitigation: check this first (Step 1) before investing in decoder evaluation.
- **Risk:** CLAP inversion approaches (C) produce low-quality audio without fine-tuning.
  → Mitigation: evaluate with explicit quality criteria (spectral coherence, no obvious
  artifacts, passes Refractor scoring).
- **Risk:** Copyright exposure if synthesis produces passages identifiable as source.
  → Mitigation: avoid granular approaches that reconstruct recognizable melodic lines;
  prefer approaches that generate novel textures.

---

## Open Questions

1. Does AudioLDM-2 support conditioning on a raw 512-dim CLAP embedding (not just text)?
2. Is the CLAP model version used in Refractor training the same as the one in AudioLDM-2
   (laion/clap-htsat-unfused vs. laion/larger_clap_music_and_speech)?
3. What GPU is available for the spike? (A10G via Modal, or local M-series?)
4. Is there a useful role for the MIDI data here (e.g., guide rhythm/pitch of generated
   audio via ControlNet-style conditioning)?
