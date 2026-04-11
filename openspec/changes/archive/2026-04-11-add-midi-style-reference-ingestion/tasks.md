## 1. Pydantic Model
- [x] 1.1 Create `app/structures/music/style_profile.py` with `StyleProfile` Pydantic model

## 2. Style Reference Module
- [x] 2.1 Create `app/generators/midi/style_reference.py`
- [x] 2.2 Implement `extract_style_profile(midi_files) → StyleProfile` — extract features from MIDI files
- [x] 2.3 Implement `load_or_extract_profile(artist_slug, style_refs_dir) → Optional[StyleProfile]` with YAML cache
- [x] 2.4 Implement `aggregate_profiles(profiles) → StyleProfile` — average across artists
- [x] 2.5 CLI: `populate` subcommand to copy files and extract profile

## 3. init_production Integration
- [x] 3.1 In `_write_song_context`: after sounds_like, extract style profiles for each artist and write `style_reference_profile` block

## 4. Pipeline Score Adjustments
- [x] 4.1 Add `style_profile_tag_adjustment(profile, pattern_tags, instrument) → float` to `aesthetic_hints.py`
- [x] 4.2 In `drum_pipeline.py`: load profile from song_context; apply density-based adjustment
- [x] 4.3 In `bass_pipeline.py`: load profile; apply duration/rest/harmonic_rhythm adjustments
- [x] 4.4 In `melody_pipeline.py`: load profile; apply density/rest adjustments

## 5. Tests
- [x] 5.1 Test feature extraction from MIDI files
- [x] 5.2 Test profile YAML caching (stale on source change, fresh on first run)
- [x] 5.3 Test aggregate_profiles averages correctly
- [x] 5.4 Test style_profile_tag_adjustment for drum, bass, melody
