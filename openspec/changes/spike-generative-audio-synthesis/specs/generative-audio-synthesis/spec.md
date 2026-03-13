## ADDED Requirements

### Requirement: CLAP Color Cluster Analysis
The spike SHALL produce a 2D visualization of all 8-color CLAP embeddings, per-color
centroids, and an 8×8 inter-cluster cosine distance matrix.

#### Scenario: Clusters are separable
- **WHEN** UMAP projection shows clear 8-cluster structure with minimal overlap
- **THEN** the spike report documents this as a go signal and proceeds to interpolation

#### Scenario: Clusters overlap significantly
- **WHEN** UMAP projection shows heavy inter-color overlap in CLAP space
- **THEN** the spike report documents this as a limiting factor, records which color
  pairs overlap most, and recommends whether further work is warranted

---

### Requirement: Embedding Interpolation Test
The spike SHALL test linear interpolation between two audio CLAP embeddings and retrieve
the nearest real audio neighbor at each interpolation step, with human evaluation notes
on perceptual coherence of the sequence.

#### Scenario: Monotonic perceptual shift observed
- **WHEN** the 5 retrieved neighbors are listened to in order
- **THEN** the evaluator notes whether the perceptual quality shifts gradually and
  meaningfully between the two endpoints; findings are documented in the spike report

#### Scenario: Interpolation retrieves duplicates or noise
- **WHEN** multiple interpolation points map to the same neighbor or clearly incoherent
  segments
- **THEN** the spike report documents this as a signal that the embedding space is not
  smoothly navigable at this granularity

---

### Requirement: Decoder Options Evaluation
The spike SHALL evaluate at least two candidate decoder approaches for generating audio
from a target CLAP embedding or chromatic description, and SHALL score each generated
output using Refractor to measure chromatic alignment.

#### Scenario: AudioCraft / text-prompt approach evaluated
- **WHEN** MusicGen is prompted with chromatic mode descriptions and outputs are scored
- **THEN** the spike report records pass rate (% outputs with chromatic_match > 0.3),
  generation time, and GPU cost per clip

#### Scenario: CLAP-conditioned diffusion model evaluated
- **WHEN** a diffusion model supporting audio embedding conditioning is identified and
  tested with color centroid embeddings as input
- **THEN** the spike report records whether the model accepts 512-dim CLAP embeddings,
  output quality rating, and Refractor chromatic match

#### Scenario: No viable decoder found
- **WHEN** all evaluated approaches fail to produce Refractor chromatic_match > 0.2
  reliably
- **THEN** the spike report documents this as a no-go with explanation; the granular
  retrieval fallback (Option A) is documented as the current best alternative

---

### Requirement: Spike Report
The spike SHALL produce a written report `training/spikes/generative-audio/spike_report.md`
with cluster analysis results, interpolation findings, a decoder comparison table, and a
clear go/no-go recommendation for a follow-on production feature.

#### Scenario: Report contains all required sections
- **WHEN** the spike is complete
- **THEN** `spike_report.md` contains sections for cluster analysis, interpolation test,
  decoder comparison (with quality/match/cost/risk columns), recommendation, and open
  questions for a potential follow-on proposal

#### Scenario: Go recommendation
- **WHEN** at least one decoder produces chromatic_match > 0.3 with acceptable cost
- **THEN** the recommendation section names the preferred approach and sketches the
  interface for a follow-on `add-generative-audio-synthesis` proposal

#### Scenario: No-go recommendation
- **WHEN** no decoder meets the quality/cost bar
- **THEN** the recommendation section documents why and suggests revisiting when better
  open-weight audio generation models become available
