## 1. ACE Studio Client
- [x] 1.1 Add `find_available_track() → int` to `AceStudioClient`
- [x] 1.2 Add `add_section_clips(sections, track_index, language) → list[dict]` to `AceStudioClient`

## 2. Singer Registry
- [x] 2.1 Extend `singer_voices.yml` schema: add `ace_id` + `last_verified` fields to existing entries
- [ ] 2.2 Add `refresh_singer_ids(ace, registry) → dict` helper in `ace_studio_export.py`

## 3. ACE Studio Export
- [x] 3.1 Replace monolithic clip export with section-aware export via `add_section_clips()`
- [x] 3.2 Use `find_available_track()` for auto track selection
- [x] 3.3 Write `ace_studio` block to `song_context.yml` after successful export
- [x] 3.4 Warn if existing `ace_studio` block is present before overwriting

## 4. ACE Studio Import
- [x] 4.1 Add `--locate-render` mode: copy ACE render to `melody/ace_render.wav`, write path to song_context.yml `ace_studio.render_path`

## 5. Pipeline Runner
- [x] 5.1 Add `ace` subcommand with `export`, `status`, `import` sub-subcommands

## 6. Tests
- [x] 6.1 Test `find_available_track` — empty track + all occupied cases
- [x] 6.2 Test `add_section_clips` — section boundary tick math
- [x] 6.3 Test section-aware export writes correct number of clips
- [x] 6.4 Test song_context ace_studio block write/overwrite
