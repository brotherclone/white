## Context
User feedback after ~3 hours of generated material: lyrics never rhyme and read as
word-lists rather than lines, syllable counts aren't converging closer to target across
repeated generations, and melodies stay near-identical across generations for the same
song/color despite templates and `/evolve`.

Prior investigation (this session) traced the causes:
- `lyric_pipeline.py:1172` (`_call_api`) is called exactly once per candidate — no
  system prompt, no rhyme instruction, no post-hoc verification or revision.
- `extract_phrases()` (`lyric_pipeline.py:128`) already segments the melody MIDI into
  phrases by rest gaps (≥0.5 beat), and each phrase already maps 1:1 to a lyric line —
  this data is real and precise, just never used for anything beyond syllable-target
  width.
- `melody_pipeline.py`'s `--evolve` flag defaults to `False` (argparse `store_true`), so
  a normal run never breeds at all.
- `pattern_evolution.py`'s `_MUTATION_PROB = 0.15` mutates one interval by ±1 semitone
  or one onset by ±0.25 beat — small enough that evolved candidates barely differ from
  their parents even when evolution does run.
- `diversity_tracker.py`'s 1.1×/1.0×/0.75× curve is too soft to overcome the pipeline's
  70%-weighted chromatic composite score, so the same 1-2 templates win for a given
  color/section on every run regardless of how many times they've already been used.

## Goals / Non-Goals
- Goals: real rhyme scheme present in generated drafts; measurable convergence toward
  syllable targets via revision rather than re-rolling; visible melody template variety
  across repeated generations for the same song/color
- Non-Goals: changing the 30%/70% theory/chromatic scoring convention used across all
  phases; changing the default lyric-generation model; rewriting the melody template
  library; guaranteeing perfect (vs. slant/near) rhyme, which is normal in songwriting

## Decisions

### Rhyme scheme derivation
Scheme notation: each line gets a letter (rhyme group) or `X` (free, no rhyme
constraint); lines sharing a letter must rhyme with each other, `X` lines are
unconstrained. Default scheme by rhyme-eligible line count in a section: 2 lines →
`AA`, 4 lines → `XAXA` (lines 2 and 4 rhyme; lines 1 and 3 are free — the "ballad
meter" pattern most real song lyrics actually use), any other count → unrhymed. An
optional `rhyme_scheme` map in `song_proposal.yml`, keyed by section label (any string
in the letter/`X` notation, e.g. `AXAX`, `ABAB`, `AABB`, or `none` to disable),
overrides the default per section. Sections sharing a base label (`verse_1`,
`verse_2`) reuse the same scheme — this makes mechanical the "same meter and rhyme
scheme as {base_label}" instruction that already exists as free text for VARIATION
sections but was never actually backed by a real scheme.
- Alternative considered: always AABB (simplest to implement) — rejected, real songs
  vary scheme by section; always-AABB just trades one kind of sameness for another.
- Alternative considered: fully-constrained `ABAB` as the 4-line default — rejected in
  favor of `XAXA`; forcing every line into a rhyme pair pushes generation back toward
  the single-word/fragment style this change is trying to fix, and constraining only
  the even lines is both more natural and more common in real lyrics.

### Rhyme verification
Use `pronouncing` (CMUdict-backed) to compare the stressed rhyming part of line-final
words. For words absent from the dictionary — common here given the project's
invented/portmanteau vocabulary (rebracketing, infranym encoding) — fall back to a
same-family suffix heuristic consistent with the existing syllable counter's
"vowel-cluster heuristic, no NLP dependency" precedent. A fallback match is a "maybe"
and does **not** count as a failure (avoids spurious revision loops on words CMUdict
simply can't judge).
- Alternative considered: heuristic-only, no new dependency — rejected, a suffix
  heuristic alone is too unreliable to drive automatic revision requests.
- `pronouncing` is pure-Python, MIT-licensed, bundles CMUdict as text data, no native
  build step — low-risk dependency addition.

### Revise loop bound
Max 2 follow-up turns per candidate. Each turn lists only the specific lines that
missed their syllable target or failed a rhyme pair, with the reason, and asks Claude
to return the full corrected lyrics text (same conversation, prior draft in context).
After 2 turns, accept the best-effort draft — `lyrics_review.yml` fitting scores and
human review in ACE Studio remain the final backstop, unchanged.

### Chorus/hook content guidance
`_infer_repeat_type` (`production_plan.py:112`) already classifies any section whose
label contains `chorus`, `refrain`, or `hook` as `EXACT` — the pipeline already writes
it once and copies it verbatim into every later instance (`EXACT_REPEAT`). That part
of "choruses should repeat" is already solved and just requires consistent section
labeling (the user's own commitment). What's missing is that the *first* instance is
still prompted identically to a verse. Add hook-specific guidance — short, one strong
central phrase, repetition of a line/phrase within the section is fine, simpler
vocabulary than verses — gated on the same `EXACT` classification, so it costs no new
taxonomy and directly rewards correct labeling.
- Alternative considered: a new explicit `section_role` field — rejected, redundant
  with the label-based inference that already exists and is already load-bearing for
  repeat behavior.

### Evolve default
Flip `--evolve` to on by default; add `--no-evolve` as an explicit opt-out. `--evolve`
itself remains accepted (as a no-op) so existing scripts that pass it explicitly don't
break. This is the smallest change that fixes "despite evolve" — evolution stops being
a step people have to remember to ask for.

### Mutation strength
Raise `_MUTATION_PROB` 0.15 → 0.35. Widen bass/melody mutation magnitude: interval
shift up to ±2 semitones (was ±1), onset shift up to ±0.5 beat (was ±0.25).
Randomize the bass/melody crossover splice point instead of the fixed bar-midpoint —
current code inspection showed the fixed split point plus tiny mutations converge the
population tightly around 1-2 seed templates within a few generations.

### Diversity curve
Steepen `diversity_tracker.py`: 0 prior uses → 1.15× (bonus), 1 prior use → 1.0×
(neutral), 2+ prior uses → `max(0.35, 0.6 - 0.1 * (uses - 2))` (was a flat 0.75× at 3+
uses). This is still a multiplier, not a hard filter, so a template that's a
dramatically better theory+chromatic fit can still win — it just costs more the more
it's already been used.

## Risks / Trade-offs
- New dependency (`pronouncing`) → low risk, see above
- Extra Claude calls for revision → bounded to 2 extra turns/candidate, only fires on
  actual check failures
- Evolve-by-default is a CLI behavior change (**BREAKING**) for any script or CI job
  calling `melody_pipeline.py` without expecting evolved candidates in the pool —
  `--no-evolve` preserves the old behavior for anything that needs it
- Steeper diversity curve could over-penalise a genuinely strong recurring template —
  mitigated by diversity being a multiplier layered on top of the existing 30/70
  theory/chromatic composite, not a replacement for it
- Steeper diversity curve also affects `bass_pipeline.py` (shared tracker) — disclosed
  in proposal.md as an accepted side effect, not hidden scope creep

## Migration Plan
No data migration. Existing `used_templates.json` registries keep their current shape
(`{template_name: count}`); only the interpretation of the count changes. Existing
`lyrics_review.yml` / `melody/review.yml` formats are unchanged (verify/revise loop
adds new optional fields, doesn't remove any).

## Open Questions
- Should `rhyme_scheme: none` also be a valid top-level default (not just per-section)
  for songs/artists that intentionally don't rhyme? Proposed: not in this change —
  section-level override is sufficient for now; revisit if it comes up in practice.
