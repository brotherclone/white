# Implementation Tasks

## 1. Constraint Generation
- [ ] 1.1 Read shrink-wrapped `chain_artifacts/index.yml` to collect all prior results
- [ ] 1.2 Extract key fields: title, bpm, key, concept (first 200 chars), rainbow_color
- [ ] 1.3 Detect clusters: group by (bpm +/- 5, key, similar concept via simple text matching)
- [ ] 1.4 Generate `chain_artifacts/negative_constraints.yml` with:
  - List of specific (bpm, key) pairs to deprioritize
  - List of concept keywords/phrases to avoid (e.g., "mirror", "consciousness examining itself")
  - List of titles already used
  - Cluster warnings (e.g., "12 of 20 proposals are in C major — strongly avoid C major")

## 2. Constraint Format
- [ ] 2.1 Define YAML schema for negative constraints
- [ ] 2.2 Include severity levels: `avoid` (soft), `exclude` (hard)
- [ ] 2.3 Include provenance: which thread(s) generated each constraint
- [ ] 2.4 Support manual overrides (user can add/remove constraints by hand)

## 3. White Agent Integration
- [ ] 3.1 Load negative constraints at workflow start (`start_workflow()`)
- [ ] 3.2 Inject constraints into the initial proposal prompt as "avoid" guidance
- [ ] 3.3 Add constraint-awareness to the Prism's facet selection (prefer facets that haven't been used)
- [ ] 3.4 Log which constraints influenced the proposal (for debugging convergence)

## 4. Diversity Metrics
- [ ] 4.1 After each new run, report diversity score across all proposals:
  - Key distribution (entropy across 12 keys)
  - BPM distribution (spread)
  - Concept uniqueness (pairwise similarity)
  - Color coverage (which rainbow colors have proposals)
- [ ] 4.2 Flag if diversity drops below threshold
- [ ] 4.3 Add diversity score to shrink-wrap manifest

## 5. Testing
- [ ] 5.1 Test constraint generation from mock shrink-wrapped artifacts
- [ ] 5.2 Test that White agent prompt includes constraints when file exists
- [ ] 5.3 Test that constraint file is created correctly from index
- [ ] 5.4 Test diversity metrics calculation
