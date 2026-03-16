## MODIFIED Requirements

### Requirement: MusicGen Chromatic Text Prompt Evaluation
The spike SHALL evaluate whether MusicGen-Medium (text-conditioned) produces audio that
achieves meaningful chromatic alignment when prompted with chromatic mode descriptions,
using Refractor as the scoring oracle.

#### Scenario: pass rate meets threshold
- **WHEN** 3 clips per color are generated and scored by Refractor
- **THEN** if ≥50% of clips per color achieve chromatic_match > 0.4, the report
  recommends opening `add-musicgen-chromatic-synthesis` as a follow-on proposal

#### Scenario: pass rate below threshold
- **WHEN** mean chromatic_match across colors is < 0.3
- **THEN** the report concludes text prompts are insufficient for chromatic targeting
  and recommends against pursuing MusicGen synthesis without better prompt engineering
  or fine-tuning

#### Scenario: results compared to corpus baseline
- **WHEN** MusicGen scores are tabulated
- **THEN** each color's mean chromatic_match is compared against the corpus retrieval
  baseline (mean score of top-20 segments from retrieve_by_color) to determine whether
  synthesis is competitive with retrieval
