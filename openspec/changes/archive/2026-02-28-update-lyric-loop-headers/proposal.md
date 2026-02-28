# Change: Use approved melody loop labels as lyric section headers

## Why

Production plan section names (`[Verse]`, `[Bridge]`) are too coarse — multiple
distinct melody loops may live inside one section, each with a different note
count and contour. Using loop labels (`[melody_verse_alternate]`) as headers
aligns lyrics.txt directly with the MIDI files in `melody/approved/`, making
ACE Studio import unambiguous and syllable fitting per-loop accurate.

## What Changes

- `lyric_pipeline.py`: `_read_vocal_sections()` returns one entry per unique
  approved melody label (not per unique section name)
- Prompt headers change from `[Verse]` to `[melody_verse_alternate]` etc.
- `lyrics.txt` (and candidate .txt files) use loop labels as section headers
- `_parse_sections()` / syllable fitting work on loop labels, not section names
- `lyrics_review.yml` `vocal_sections` field lists loop labels

## Impact

- Affected specs: `lyric-pipeline` (new capability spec)
- Affected code: `app/generators/midi/lyric_pipeline.py`
- No breaking changes to other phases
