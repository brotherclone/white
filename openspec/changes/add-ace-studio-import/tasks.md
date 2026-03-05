## 1. Parser Core
- [ ] 1.1 Implement `parse_ace_export(midi_path) -> list[dict]` — reads tempo, ticks-per-beat,
        collects lyric meta + note_on/note_off pairs into per-syllable dicts
- [ ] 1.2 Implement `merge_syllables(syllable_events) -> list[dict]` — collapses `word#1 word#2 word#3`
        into single word tokens, carrying start_beat from `#1` and end_beat from the last fragment
- [ ] 1.3 Implement `find_ace_export(production_dir) -> Path | None` — locates
        `VocalSynthvN/VocalSynthvN_<singer>.mid` by glob, returns the highest-versioned file

## 2. LRC Export
- [ ] 2.1 Implement `export_lrc(word_events, tempo, output_path)` — converts beat positions
        to `MM:SS.cc` timestamps and writes standard LRC format
- [ ] 2.2 CLI: `python -m app.generators.midi.ace_studio_import --production-dir <dir> [--lrc-out <path>]`
        defaults to `<production_dir>/vocal_alignment.lrc`

## 3. Python API
- [ ] 3.1 Expose `load_ace_export(production_dir) -> list[dict] | None` — one-call helper used
        by drift report; returns None with a warning if no export MIDI found

## 4. Tests
- [ ] 4.1 Unit: syllable merge (`iron#1 iron#2` → `iron`, `da` remains `da`)
- [ ] 4.2 Unit: beat-to-timestamp conversion at 60 BPM / 3/4 and 120 BPM / 4/4
- [ ] 4.3 Unit: LRC line format (`[MM:SS.cc] word`)
- [ ] 4.4 Integration: parse the real `VocalSynthv0_1_Anderson.mid` and assert word count,
        first word, last word, and total duration
