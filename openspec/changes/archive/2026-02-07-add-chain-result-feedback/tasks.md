# Implementation Tasks

## 1. Constraint Generation
- [x] 1.1 Read shrink-wrapped `shrinkwrapped/index.yml` to collect all prior results
- [x] 1.2 Extract key fields: title, bpm, key, concept (first 200 chars), rainbow_color
- [x] 1.3 Detect clusters: group by (bpm +/- 5, key, similar concept via marker phrase matching)
- [x] 1.4 Generate `shrinkwrapped/negative_constraints.yml` with:
  - List of specific (bpm, key) pairs to deprioritize
  - List of concept keywords/phrases to avoid (e.g., "transmigration", "seven chromatic methodologies")
  - List of titles already used
  - Cluster warnings (e.g., "12 of 20 proposals are in C major — strongly avoid C major")

## 2. Constraint Format
- [x] 2.1 Define YAML schema for negative constraints
- [x] 2.2 Include severity levels: `avoid` (soft >30%), `exclude` (hard >50%)
- [x] 2.3 Include provenance: reason string documenting count/fraction per constraint
- [x] 2.4 Support manual overrides (preserved on regeneration via `manual_overrides` key)

## 3. White Agent Integration
- [x] 3.1 Load negative constraints at workflow start (`start_workflow()`) from `shrinkwrapped/index.yml`
- [x] 3.2 Inject constraints into the initial proposal prompt (`initiate_song_proposal()`)
- [x] 3.3 Inject constraints into rewrite prompt (`rewrite_proposal_with_synthesis()`)
- [x] 3.4 Log when constraints are loaded and injected

## 4. Diversity Metrics
- [x] 4.1 Key distribution entropy (bits) across all proposals
- [x] 4.2 BPM standard deviation and mean
- [x] 4.3 Concept phrase frequency counts
- [x] 4.4 Warnings when key entropy < 2.0 bits or BPM std dev < 10
- [x] 4.5 Add diversity score to shrink-wrap manifest (deferred to future iteration — metrics available via CLI)

## 5. Testing
- [x] 5.1 Test constraint generation from mock shrink-wrapped artifacts (27 tests)
- [x] 5.2 Test key normalization (handles LLM mangled keys like "C hromatic Complete")
- [x] 5.3 Test constraint file write with manual override preservation
- [x] 5.4 Test diversity metrics calculation (entropy, std dev)
- [x] 5.5 Test prompt formatting (format_for_prompt produces valid prompt block)
