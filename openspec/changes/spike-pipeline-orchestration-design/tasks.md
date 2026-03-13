## 1. Data Flow Map
- [ ] 1.1 For each pipeline phase, document: input files + fields read, output files +
      fields written, which fields are derived vs. passed through
- [ ] 1.2 Trace `sounds_like` specifically: every file it appears in, every function that
      reads or writes it, whether it is the same data or derived/transformed
- [ ] 1.3 Trace `concept` and `color`: same treatment as sounds_like — both are re-read
      per phase and sometimes inconsistently (manifest vs. proposal vs. review.yml)
- [ ] 1.4 Produce a dependency graph (ASCII or Mermaid) showing phase → file → phase
      relationships

## 2. Load_song_proposal Audit
- [ ] 2.1 List all implementations: `chord_pipeline.py:161`, `production_plan.py:228`,
      `lyric_pipeline.py._find_and_load_proposal()`, and any others found via grep
- [ ] 2.2 For each: document the field set returned, the file(s) read, and which callers
      depend on it
- [ ] 2.3 Determine whether one canonical loader can satisfy all callers; if not, document
      the minimum divergence needed

## 3. Phase Review Schema Audit
- [ ] 3.1 Document the schema of each phase review YAML (keys, types, required vs.
      optional fields)
- [ ] 3.2 Identify shared structure (candidates list, approved_label, scoring_weights,
      color, bpm) vs. phase-specific fields
- [ ] 3.3 Evaluate whether a shared base schema would help or over-constrain; note any
      fields that `promote_part.py` or other cross-phase readers depend on

## 4. Production Context Design
- [ ] 4.1 Define the proposed `song_context.yml` schema: field names, types, required
      vs. optional, source of truth for each field
- [ ] 4.2 Evaluate backward compatibility: can existing production dirs (violet, blue)
      have `song_context.yml` generated from their existing files without re-running any
      MIDI phase?
- [ ] 4.3 Evaluate relationship to `production_plan.yml` and `composition_proposal.yml`:
      which fields overlap, which are superseded, whether those files remain or merge

## 5. Spike Report
- [ ] 5.1 Write `training/spikes/pipeline-orchestration/design_report.md` with:
      - Section 1: Data flow map (dependency graph)
      - Section 2: load_song_proposal audit findings
      - Section 3: Phase review schema comparison
      - Section 4: Proposed song_context.yml schema
      - Section 5: Migration strategy for existing production dirs
      - Section 6: Recommended scope for follow-on `refactor-pipeline-orchestration`
        (what to change, what to leave alone, in what order)
- [ ] 5.2 Decision gate: if the proposed refactor scope exceeds ~2 sessions of work,
      recommend a phased approach (incremental changes vs. big-bang rewrite)
