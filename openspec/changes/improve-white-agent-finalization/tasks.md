# Tasks: Improve White Agent Finalization

## Phase 1: Artifact Integration

- [x] **1.1** Create `_gather_artifact_summaries(state)` method that extracts key content from artifacts by type
- [x] **1.2** Add artifact summaries to `_perform_meta_rebracketing` prompt
- [x] **1.3** Add artifact summaries to `_generate_chromatic_synthesis` prompt
- [x] **1.4** Test with full spectrum run to verify artifacts appear in output

## Phase 2: Dynamic Agent Context

- [ ] **2.1** Modify meta-rebracketing prompt to only list agents present in `transformation_traces`
- [ ] **2.2** Include each agent's `document_synthesis` in the aggregation
- [ ] **2.3** Replace hardcoded boundary descriptions with actual `boundaries_shifted` from traces
- [ ] **2.4** Add agent-specific artifact counts to context

## Phase 3: Enrich Transformation Traces

- [ ] **3.1** Add `content_excerpt: Optional[str]` field to `TransformationTrace`
- [ ] **3.2** Add `artifact_count: int` field to `TransformationTrace`
- [ ] **3.3** Update each `process_*_agent_work` method to populate new fields
- [ ] **3.4** Update `_format_transformation_traces` to include excerpts

## Phase 4: Synthesis Prompt Variation

- [ ] **4.1** Create `SynthesisPromptTemplate` enum or similar structure
- [ ] **4.2** Implement 3-4 distinct synthesis templates (temporal, ontological, emotional, structural)
- [ ] **4.3** Add logic to select template based on dominant patterns in traces
- [ ] **4.4** Test each template produces meaningfully different output

## Phase 5: Per-Agent Synthesis Sections

- [ ] **5.1** Create `_generate_agent_mini_synthesis(agent_name, traces, artifacts)` method
- [ ] **5.2** Call for each agent that ran, preserving their unique lens
- [ ] **5.3** Aggregate mini-syntheses with clear section headers
- [ ] **5.4** Ensure final document maintains coherence while preserving diversity

## Validation

```bash
# Run full spectrum and compare outputs
MOCK_MODE=false python run_white_agent.py start --mode full_spectrum

# Check that artifacts appear in synthesis
grep -r "extinction\|timeline\|newspaper" chain_artifacts/*/md/*chromatic_synthesis*

# Compare multiple runs for variety
diff chain_artifacts/run1/md/*synthesis* chain_artifacts/run2/md/*synthesis*
```

## Dependencies

- Phase 1 can start immediately
- Phase 2 depends on understanding current document_synthesis flow
- Phase 3 requires updating TransformationTrace model
- Phase 4 can run parallel to Phase 3
- Phase 5 depends on Phases 1-3

## Success Criteria

1. Final synthesis references specific artifact content (extinction species names, timeline divergence points, newspaper headlines)
2. Different runs produce structurally varied outputs
3. Stopping after different agents produces appropriately scoped synthesis
4. Clear attribution of which lens contributed which insight
