# Proposal: add-diatonic-workhorse-chords

## Why

The Refractor-scored Markov candidates all chase the chromatic target, which
makes them expressive but uniformly exotic. Verses in particular benefit from
grounded, guitar-playable foundations — a I–V–vi–IV under a verse lets the
more adventurous chromatic sections feel like genuine departures rather than
the default.

Choruses, bridges, and other sections should remain Refractor-only: the
rainbow chromatic identity is load-bearing there. Verses are the one place
where "workhorse" harmony is an asset.

## What Changes

`chord_pipeline.py` gains a second generation pass for verse sections only.
After the Markov candidates are scored and ranked, a fixed set of common
diatonic progressions is built by pulling chords directly from the chord bank
by their `function` field (I, IV, V, vi, etc.) and appended to the candidate
list as `source: diatonic`. These candidates carry no Refractor score and are
not ranked — they are clearly labelled workhorse options for the reviewer to
pick from if the Markov results feel too exotic.

Non-verse sections (chorus, bridge, interlude, jams, etc.) are unchanged.

## Design Decisions

- **Source**: diatonic candidates are assembled from the existing chord bank
  (`function` + `key_root` + `key_quality` columns), so no new MIDI synthesis
  is needed — the voicings are the same real-instrument recordings already used
  by Markov candidates.
- **Section filter**: only sections whose label contains `"verse"` receive
  diatonic candidates. All other sections get Refractor candidates only.
- **Patterns**: a fixed set of 8 progressions covering the most common major
  and minor diatonic patterns (see spec). Length matches the section's bar
  count by repeating or trimming the pattern.
- **No Refractor score**: diatonic candidates have `scores: null` in
  `review.yml`. The reviewer sees them clearly separated by the `source:
  diatonic` field.
- **Count**: up to 8 diatonic candidates per verse section (one per pattern),
  added after all Markov candidates.
- **Labelling**: IDs follow `{section}_diatonic_{pattern_name}` e.g.
  `verse_diatonic_I_V_vi_IV`.

## Impact

- `chord_pipeline.py`: new `build_diatonic_candidates()` function + call site
- `review.yml`: additional candidates with `source: diatonic`, `scores: null`
- No changes to scoring, promotion, or downstream phases
- Closes issue #183
