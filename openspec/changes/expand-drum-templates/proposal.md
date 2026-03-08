# Proposal: expand-drum-templates

## Summary

Add a `breakbeat` genre family to `drum_patterns.py` populated with templates
transcribed from the classic-breakbeats reference spreadsheet. This roughly doubles
the 4/4 template count and opens up hip-hop, funk, and soul genre tags for automatic
drum selection.

## Motivation

The current library has 8 genre families (ambient, electronic, krautrock, rock,
classical, experimental, folk, jazz/americana) — all broadly European/rock-adjacent.
The White project's color palette maps heavily to soul, R&B, and hip-hop reference
points (especially Red, Orange, Blue, Violet), but there are no templates that
speak to those idioms. The reference spreadsheet provides 9 classic breaks as an
authoritative transcription source.

## Source of Truth

Pattern grid data MUST be transcribed from:
> https://docs.google.com/spreadsheets/d/19_3BxUMy3uy1Gb0V8Wc-TcG7q16Amfn6e8QVw4-HuD0/

The sheet uses a standard 16-step grid: `1, e, +, a, 2, e, +, a, 3, e, +, a, 4, e, +, a`
Each step maps to a quarter-beat position increment of **0.25** (matching the
existing system's float beat coordinates).

Grid-to-float mapping:
```
1=0.0  1e=0.25  1+=0.5  1a=0.75
2=1.0  2e=1.25  2+=1.5  2a=1.75
3=2.0  3e=2.25  3+=2.5  3a=2.75
4=3.0  4e=3.25  4+=3.5  4a=3.75
```

## Named Patterns (from spreadsheet left column)

These are the 9 single-bar classic breaks to transcribe as medium-energy templates.
Low and high energy variants are derived from them.

| Template name | Source break |
|---|---|
| `billie_jean` | Billie Jean (MJ) |
| `funky_drummer` | The Funky Drummer (James Brown / Clyde Stubblefield) |
| `impeach_the_president` | Impeach The President (The Honey Drippers) |
| `when_the_levee_breaks` | When The Levee Breaks (Led Zeppelin) |
| `walk_this_way` | Walk This Way (Aerosmith) |
| `its_a_new_day` | It's a New Day (James Brown / Lyn Collins) |
| `papa_was_too` | Papa Was Too (Joe Tex / Skull Snaps) |
| `the_big_beat` | The Big Beat (Billy Squier) |
| `ashleys_roachclip` | Ashley's Roachclip (Soul Searchers) |

## Scope

**In scope**
- Add `breakbeat` entry to `GENRE_FAMILY_KEYWORDS` (keywords: "breakbeat", "hip-hop",
  "boom bap", "break", "funk", "soul", "r&b", "groove")
- Add `tambourine` (MIDI 54), `conga_high` (63), `conga_low` (64), `cowbell` (56)
  to `GM_PERCUSSION`
- Add `TEMPLATES_4_4_BREAKBEAT` list: 9 medium-energy classic break templates + at
  least 3 low-energy and 3 high-energy variants (minimum 15 templates total)
- Register `TEMPLATES_4_4_BREAKBEAT` in `ALL_TEMPLATES`
- All existing tests continue to pass; new tests cover breakbeat coverage

**Out of scope**
- Multi-part / fill patterns from the right column of the spreadsheet (future)
- New time signatures (e.g. 6/8 for shuffle feels) — future
- Funk as a separate genre family — fold into `breakbeat` keywords for now
- Any changes to `drum_pipeline.py` or scoring logic

## Files Affected

- `app/generators/midi/patterns/drum_patterns.py` — primary change
- `tests/generators/midi/test_drum_pipeline.py` — new assertions for breakbeat family

## Notes for Implementer

- The spreadsheet's right column contains 4-part multi-bar patterns. These are
  **not** in scope for this change. Capture single-bar patterns only.
- `tambourine` at MIDI 54 overlaps with some GM implementations using 54 for
  Tambourine — confirm with a quick MIDI preview in Logic before committing.
- `test_each_family_has_low_medium_high` enforces all three energy levels — the
  breakbeat family must have at least one template per energy level.
- Velocity conventions: accented snare backbeats → "accent"; most breakbeat kicks
  → "normal"; ghost 16th hats → "ghost". The Funky Drummer has significant ghost
  note complexity — encode faithfully.
