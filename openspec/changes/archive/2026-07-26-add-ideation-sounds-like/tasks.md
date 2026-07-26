## 1. Reference data plumbing
- [x] 1.1 No new plumbing needed — `get_sounds_like_by_color(color_character)`
      and `sample_reference_artists(artists)` in
      `packages/extraction/src/white_extraction/util/manifest_loader.py`
      already exist and already do exactly this; only Blue Agent was calling
      them
- [x] 1.2 Decided: reuse the existing helpers as-is rather than touching
      `get_my_reference_proposals()` or `SongProposalIteration`

## 2. Per-agent prompt updates
- [x] 2.1 Black Agent — include sounds_like in reference-works section
- [x] 2.2 Red Agent
- [x] 2.3 Orange Agent
- [x] 2.4 Yellow Agent
- [x] 2.5 Green Agent — also fixed an unrelated pre-existing bug found while
      testing this: the "Some other examples from the archive" line
      interpolated `the_rainbow_table_colors['G']` a second time instead of
      `get_my_reference_proposals('G')`, so Green never actually saw
      reference examples at all, just its own color code repeated
- [x] 2.6 Violet Agent
- [x] 2.7 Blue Agent — surfaced `state.musical_params.reference_artists`
      (already computed at `blue_agent.py:1053-1061`) in
      `generate_alternate_song_spec`'s prompt

## 3. Tests
- [x] 3.1 `manifest_loader` unit test: not added — `get_sounds_like_by_color`/
      `sample_reference_artists` are pre-existing, already-used functions,
      not new code from this change
- [x] 3.2 Per-agent unit test: prompt includes sounds_like artists when
      reference data has them (all 6 of Black/Red/Orange/Yellow/Green/Violet)
- [x] 3.3 Blue Agent regression test: `reference_artists` appears in the
      prompt text passed to `_invoke_structured`

## 4. Verification
- [x] 4.1 Run full test suite (`packages/ideation/tests`, `packages/core/tests`,
      `packages/extraction/tests`)
- [x] 4.2 Live smoke test: run the full proposal chain once and confirm (via
      chain_artifacts debug snapshots) that at least one color agent's prompt
      contains sounds-like artist names
