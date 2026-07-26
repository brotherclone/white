# Change: Wire "sounds like" artist references into ideation prompts

## Why
Manifests carry a `sounds_like` field (artist comparisons), but it never
actually reaches any color agent's LLM prompt during proposal ideation:

- `get_my_reference_proposals()` (`packages/extraction/src/white_extraction/util/manifest_loader.py`),
  used by Black, Red, Orange, Yellow, Green, and Violet agents for their
  "reference works in this artist's style" prompt section, builds each
  reference entry from `bpm`, `tempo`, `key`, `title`, `mood`, `genres`,
  `concept` only — `sounds_like` is dropped.
- Blue Agent is the only agent that calls the sibling `get_sounds_like_by_color()`
  helper (`blue_agent.py:1053`), storing the result in
  `state.musical_params.reference_artists` — but that field is never
  interpolated into any prompt string anywhere in `blue_agent.py` (confirmed
  by grep: `reference_artists` appears only at its assignment, line 1061).
  It's computed and immediately discarded.
- Indigo, White have no `sounds_like` handling at all.

Net effect: none of the 8 color agents currently receive artist-comparison
context when generating their counter-proposal, even though the data exists
and (for Blue) is already being fetched.

This is separate from the already-working, already-spec'd `sounds-like-feature`
/`sounds-like-bootstrap` capabilities, which cover ML-training embedding
precomputation — not ideation-time prompt content — and from the production-time
`lyric_pipeline.py`'s `load_artist_context(meta.get("sounds_like"))`, which
already works correctly for lyric generation on promoted songs.

## What Changes
- Have each color agent call the existing `get_sounds_like_by_color()` +
  `sample_reference_artists()` helpers (already implemented, currently only
  called by Blue Agent) and add a line to its reference-works prompt section
  listing sampled sounds-like artists for that color. No change to
  `get_my_reference_proposals()`'s return shape — `SongProposalIteration`
  stays a clean proposal schema, and this reuses machinery that already
  exists and already works, rather than adding a new one.
- Surface Blue Agent's already-computed `state.musical_params.reference_artists`
  in its `generate_alternate_song_spec` prompt instead of leaving it unread.
- Scope explicitly excludes the ML-training embedding pipeline
  (`sounds-like-feature`) and the lyric-generation pipeline (already working) —
  this change is ideation-proposal-prompt content only.

## Impact
- Affected specs: `ideation-sounds-like` (new capability)
- Affected code:
  - `packages/extraction/src/white_extraction/util/manifest_loader.py` —
    extend `get_my_reference_proposals()` or add a sibling helper
  - `packages/ideation/src/white_ideation/agents/{black,red,orange,yellow,green,violet}_agent.py` —
    consume the extended reference data in the existing reference-works
    prompt section
  - `packages/ideation/src/white_ideation/agents/blue_agent.py` — reference
    `state.musical_params.reference_artists` in the counter-proposal prompt
