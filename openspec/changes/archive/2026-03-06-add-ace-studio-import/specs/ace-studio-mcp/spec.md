## ADDED Requirements

### Requirement: ACE Studio MIDI Import

The pipeline SHALL parse ACE Studio vocal export MIDI files back into structured,
word-level note events so that downstream tools can work with the actual performed
vocal data rather than estimates from source MIDI loops.

The importer SHALL locate the export MIDI by globbing `VocalSynthvN/VocalSynthvN_*.mid`
within the production directory and selecting the highest-versioned file found.

Lyric meta messages in the export use ACE Studio's syllable-split convention
(`word#1`, `word#2`, …) for polysyllabic words. The importer SHALL merge fragments
into single word tokens, carrying the start position of the first fragment and the
end position of the last.

Each emitted word event SHALL include:
- `word` — reconstructed word string (no `#N` suffix)
- `syllable_count` — number of ACE syllable fragments that formed this word
- `start_beat` — fractional beat number (relative to start of file)
- `end_beat` — fractional beat number
- `pitch` — MIDI note number of the note sounding at word onset
- `velocity` — MIDI velocity at onset

#### Scenario: Happy path — standard ACE export

- **WHEN** `load_ace_export(production_dir)` is called
- **AND** a `VocalSynthvN/VocalSynthvN_*.mid` file exists
- **THEN** a list of word event dicts is returned, one per unique word token
- **AND** polysyllabic fragments are merged (`iron#1 iron#2` → `{word: "iron", syllable_count: 2}`)
- **AND** single-syllable words are passed through unchanged (`{word: "rust", syllable_count: 1}`)

#### Scenario: No export MIDI found

- **WHEN** `load_ace_export(production_dir)` is called
- **AND** no `VocalSynthvN/` folder or MIDI file exists
- **THEN** `None` is returned
- **AND** a warning is logged naming the production directory

#### Scenario: Multiple versioned exports

- **WHEN** both `VocalSynthv0/` and `VocalSynthv1/` folders exist
- **THEN** the highest-versioned folder's MIDI is used
- **AND** a log message notes which file was selected

---

### Requirement: LRC File Export

The importer SHALL derive word-level absolute timestamps from the MIDI tempo and
ticks-per-beat, and write a standard LRC subtitle file.

Timestamp format: `[MM:SS.cc]` where `cc` is hundredths of a second.
One LRC line per word. Lines ordered chronologically. File written as UTF-8.

#### Scenario: LRC generated from parse

- **WHEN** `export_lrc(word_events, tempo, output_path)` is called
- **THEN** an LRC file is written at `output_path`
- **AND** each line has the form `[MM:SS.cc] word`
- **AND** timestamps reflect the actual onset of each word in wall-clock time

#### Scenario: CLI default output path

- **WHEN** `python -m app.generators.midi.ace_studio_import --production-dir <dir>` is run
- **AND** `--lrc-out` is not supplied
- **THEN** the LRC file is written to `<production_dir>/vocal_alignment.lrc`

#### Scenario: 60 BPM round-trip

- **WHEN** the source MIDI has tempo 1000000 µs/beat (60 BPM) and ticks-per-beat 480
- **AND** a word event starts at beat 36.0
- **THEN** its LRC timestamp is `[00:36.00]`
