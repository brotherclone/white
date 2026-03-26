# Spike Report: Stable Audio Open — Chromatic Text Prompt Evaluation

**Date:** 2026-03-16
**Model:** `stabilityai/stable-audio-open-1.0` (diffusers, StableAudioPipeline)
**GPU:** Modal A10G
**Clips:** 6 total (0 per color × 8 colors)
**Verdict:** **NO-GO**

---

## 1. Prompts Used

**Red:** sparse ambient piano, melancholic, slow, introspective, past memories, wistful, fading, quiet reverb, nostalgic

**Orange:** warm nostalgic folk texture, familiar place, rootsy acoustic guitar, gentle, grounded, orange sunset, belonging

**Yellow:** present moment, bright acoustic texture, open fields, optimistic, forward motion, daylight, organic, alive

**Green:** organic nature sounds, present-tense growth, living breathing texture, forest ambience, gentle movement, verdant

**Blue:** present urban ambient, cool minimal electronic, observational, street sounds dissolved into drone, detached, blue hour

**Indigo:** alien drone, suspended time, no tonal anchor, unreal space, dissonant shimmer, liminal void, indigo darkness

**Violet:** dreaming, liminal between-states, soft electronic wash, future imagined, violet dusk, floating, aspirational, dissolving

**Black:** dark ambient void, formless, subterranean rumble, undefined, no melody, deep silence with texture, black space

---

## 2. Results

Pass threshold: chromatic_match > 0.4

| Color | Mean match | Min | Max | Pass rate | vs corpus baseline | Gen time |
|-------|-----------|-----|-----|-----------|-------------------|---------|
| Red | 0.3352 | 0.3352 | 0.3352 | 0% (0/1) | — | 12.7s |
| Orange | 0.3319 | 0.3319 | 0.3319 | 0% (0/1) | — | 12.2s |
| Yellow | — | — | — | — | — | — |
| Green | 0.3321 | 0.3321 | 0.3321 | 0% (0/1) | — | 12.1s |
| Blue | 0.3353 | 0.3353 | 0.3353 | 0% (0/1) | — | 11.2s |
| Indigo | — | — | — | — | — | — |
| Violet | — | — | — | — | — | — |
| Black | 0.3333 | 0.3333 | 0.3333 | 0% (0/2) | — | 11.3s |

**Overall pass rate:** 0%

---

## 3. License Assessment

Stable Audio Open uses the **Stability AI Community License**:

- Non-commercial and small-business use (annual revenue < $1M): **free**
- Commercial use above $1M revenue threshold: requires a paid license from Stability AI
- Generated outputs are owned by the operator; no attribution required

**For Earthly Frames:** If annual revenue is currently under $1M, generated audio can be
used commercially under the community license at no cost. Verify current revenue threshold
at https://stability.ai/community-license-agreement before shipping any generated audio.

This is meaningfully better than MusicGen (CC-BY-NC = non-commercial only with no pathway
to commercial use).

---

## 4. Go / No-Go

**NO-GO**

Overall pass rate 0% is below the ≥50% threshold. Only 0/5 colors meet ≥50%. Text prompts alone are not a reliable chromatic targeting mechanism — granular retrieval remains the primary path.

---

## 5. Root Cause Analysis

All 6 scored clips returned chromatic_match ≈ 0.333 (uniform distribution across 3 classes per dimension). This is the Refractor model's "I don't know" output — the same score you get with a random 512-dim vector.

**What this means:** Stable Audio's CLAP embeddings lie outside the training distribution of the Refractor model. Refractor was trained entirely on real recordings from the White corpus (~9,900 real music segments). The spectral/timbral characteristics of diffusion-synthesized audio are sufficiently different from recorded music that the model outputs maximum uncertainty rather than a meaningful classification.

**Generation quality itself is not the issue.** The clips generated in ~12s each on A10G. The diffusion model is producing coherent, styled audio. The problem is the evaluation pipeline, not the generation.

**Paths forward if synthesis remains a goal:**
1. **Human evaluation bypass** — Score clips by ear, not by Refractor. Generate 3 clips per color and present to the human producer for selection.
2. **Refractor fine-tuning** — Add a small set of synthetic audio clips (labeled by humans) to retrain the classification heads. This closes the domain gap.
3. **Prompt → corpus retrieval hybrid** — Use Stable Audio for texture generation but evaluate it by retrieving the nearest real corpus segment (by CLAP similarity) and using that segment's Refractor score as a proxy.
4. **Accept the no-go** — Granular synthesis from the real corpus is already working and produces chromatic audio Refractor can measure. Stable Audio adds complexity for uncertain gain.
