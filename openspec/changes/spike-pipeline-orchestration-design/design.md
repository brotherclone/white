## Context

The pipeline phases and their primary inputs/outputs as they exist today:

```
Song proposal YAML  (human/agent authored, lives in thread/yml/)
        │
        ▼
[chord_pipeline.py]  reads: song proposal YAML, thread manifest
        │             writes: chords/candidates/, chords/review.yml
        ▼
[drum_pipeline.py]   reads: chords/review.yml, production_plan.yml (optional)
        │             writes: drums/candidates/, drums/review.yml
        ▼
[bass_pipeline.py]   reads: chords/review.yml, drums/review.yml
        │             writes: bass/candidates/, bass/review.yml
        ▼
[melody_pipeline.py] reads: chords/review.yml, bass/review.yml
        │             writes: melody/candidates/, melody/review.yml
        │
  [Logic Pro assembly — human step]
        │
        ▼
[composition_proposal.py]  reads: all review.ymls, song proposal YAML
        │                  writes: composition_proposal.yml
        │                  (first time sounds_like appears in machine-readable form)
        ▼
[lyric_pipeline.py]  reads: chords/review.yml → song proposal YAML, arrangement.txt
        │             writes: melody/lyrics_review.yml, melody/candidates/*.txt
        ▼
[score_mix.py]       reads: mix bounce audio
                     writes: melody/mix_score.yml
```

Key observation: `composition_proposal.yml` is produced **between** melody and lyrics, but
carries data (sounds_like, rationale, section intent) that would be useful to ALL earlier
phases.

---

## Goals / Non-Goals

Goals:
- Map every cross-phase data dependency explicitly (what file, what field, which phase
  reads it, which phase writes it)
- Identify the minimum viable "production context" — the set of fields that every phase
  needs and that should be stable from initialization onward
- Evaluate whether a shared context file (distinct from composition_proposal.yml) would
  reduce duplication without adding unnecessary complexity
- Assess whether phase review files should share a common schema

Non-Goals:
- Rewriting the pipeline (that is a follow-on implementation task)
- Changing the human workflow (Logic Pro assembly remains a manual step)
- Adding new generation capabilities (that is for other specs)

---

## Known Pain Points to Investigate

### A. Dual load_song_proposal implementations
`chord_pipeline.py:161` and `production_plan.py:228` both parse the song proposal YAML
but return different field sets. `lyric_pipeline.py` has a third variant via
`_find_and_load_proposal()`. Determine: should there be one canonical loader? Where should
it live?

### B. sounds_like data flow gap
Song proposal → chord pipeline: `sounds_like` is often absent (pre-composition-proposal
songs) or not read. `add-initial-sounds-like-proposal` patches this, but the patch
introduces a fourth place that reads/writes sounds_like. Document the full sounds_like
lifecycle and determine the canonical source of truth.

### C. composition_proposal.yml as dual-purpose artifact
The file serves:
1. Human creative review (rationale, section intent, transition notes)
2. Machine data source (loop inventory, sounds_like, proposed_by)

Is this healthy? Should the machine-readable fields be split into a separate `song_context.yml`
that persists from init through all phases? Or is composition_proposal.yml the right
long-term home if it runs earlier?

### D. Phase review file schema divergence
`chords/review.yml`, `drums/review.yml`, `bass/review.yml`, `melody/review.yml`,
`melody/lyrics_review.yml` each have different structures. Evaluate whether a shared
`approved_label` + `candidates` skeleton would simplify downstream readers (e.g.
`promote_part.py` already abstracts over this somewhat).

### E. No pipeline status / run log
There is no file recording which phases have run, when, and with what model/parameters.
Evaluate whether a `pipeline_log.yml` or similar would help with debugging and resuming
interrupted pipelines.

---

## Proposed Target Architecture (hypothesis — to validate or refute)

```
[init_production.py]  ← new (see add-initial-sounds-like-proposal)
        │  writes: song_context.yml  (canonical, stable, validated)
        │  fields: title, color, concept, key, bpm, time_sig, singer,
        │           sounds_like, genres, mood, source_proposal_path
        ▼
[chord/drum/bass/melody/lyric pipelines]
        │  ALL read: song_context.yml for metadata
        │  write:    phase review YAMLs (unchanged schema)
        ▼
[composition_proposal.py]
        │  reads: song_context.yml + all phase review YAMLs
        │  writes: composition_proposal.yml (creative artifact, human-facing)
        │  may update: song_context.yml.sounds_like (if Claude proposes changes)
        ▼
[score_mix.py, drift_report, etc.]
        │  reads: song_context.yml for target
```

The key claim: `song_context.yml` is the production-dir-scoped source of truth for song
metadata. It is written once at init, read by every phase, and optionally updated by
composition_proposal. It is NOT a creative artifact — it has a validated schema.

---

## Open Questions

1. Does moving metadata to `song_context.yml` create a migration problem for existing
   production directories (violet, blue, etc.)? Can a fallback reader reconstruct it from
   `chords/review.yml` for legacy dirs?
2. Should `composition_proposal.yml` continue to duplicate metadata fields (color, bpm,
   etc.) or reference `song_context.yml` as canonical and omit them?
3. Is `promote_part.py` already generic enough, or does it need to be aware of
   `song_context.yml`?
4. Are the 5 phase review YAML schemas worth unifying? What would break?
5. Where does `production_plan.yml` fit? It overlaps with `song_context.yml` and
   `composition_proposal.yml` — is it still needed?
