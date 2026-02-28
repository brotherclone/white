# Change: Add Artist Style Catalog

## Why

`sounds_like` references appear in every song proposal and production plan, but are
currently inert — just a list of names that nothing reads. They represent the human's
intuition about the aesthetic neighbourhood each song inhabits, which is exactly the
kind of signal the generative pipeline should have access to.

The challenge is using this without touching other artists' actual work. The solution:
generate short text descriptions of each artist's *aesthetic* (sonic texture, thematic
tendencies, production character, emotional register) using Claude's baseline knowledge,
let the human review and fill gaps, then embed those descriptions into prompts as style
context. No audio, lyrics, MIDI, or copyrighted material involved — only factual/critical
characterisation of a public artistic style, expressed in Claude's own words.

## What Changes

- **`app/data/artist_catalog.yml`** — the canonical catalog: one entry per artist, each
  with a generated description, style tags, an optional human-filled chromatic hint
  (which temporal/spatial/ontological modes the artist tends toward), a ChromaticScorer
  score (from embedding the description), and freeform `notes` for White-project framing.

- **`app/generators/artist_catalog.py`** — CLI tool that:
  1. Collects unique artist names from two sources: production plan YAMLs (`--thread`) and
     the full training parquet (`--from-training-data`), parsing the `"Artist, discogs_id: N"`
     format; stores Discogs IDs as reference metadata
  2. Generates missing descriptions via Claude API (uses the project's own API key / .env)
  3. Optionally re-scores all descriptions through ChromaticScorer (`--score-chromatic`)
  4. Prints a summary of the catalog state

- **Pipeline injection (lyric + chord prompts)** — when a production plan has
  `sounds_like` entries that exist in the catalog, the relevant descriptions are injected
  into the Claude prompt as style context. This is read-only; no training data is created
  from artist descriptions.

## What This Is NOT

- Not scraping third-party sites (Wikipedia, AllMusic) — all descriptions are
  Claude-generated and then human-reviewed
- Not training on copyrighted content (audio, lyrics, MIDI from other artists)
- Not adding artist data to the ChromaticScorer training set
- Not auto-classifying artists as belonging to a color — that's a human judgment filled
  in `chromatic_hint`, the ChromaticScorer score is informational only

## Impact

- Affected specs: `artist-style-catalog` (ADDED — new capability)
- Affected code:
  - `app/data/artist_catalog.yml` — new data file (grows with project)
  - `app/generators/artist_catalog.py` — new CLI (~200 lines)
  - `app/generators/midi/lyric_pipeline.py` — inject catalog context into prompt
  - `app/generators/midi/chord_pipeline.py` — inject catalog context into prompt
  - `tests/generators/test_artist_catalog.py` — new tests
