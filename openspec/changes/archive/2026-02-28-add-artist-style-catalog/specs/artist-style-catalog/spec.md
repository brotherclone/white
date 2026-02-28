## ADDED Requirements

### Requirement: Artist Catalog Data Format

The artist catalog SHALL be a YAML file at `app/data/artist_catalog.yml` with one entry
per artist. Each entry SHALL contain enough structured fields to support both human review
and machine injection into generative prompts.

#### Scenario: Catalog entry structure

- **WHEN** an artist entry is written to the catalog
- **THEN** it SHALL contain:
  - `slug`: snake_case identifier derived from the artist name
  - `status`: one of `draft` (generated, not yet reviewed) or `reviewed` (human-approved)
  - `description`: 100–200 word prose covering sonic texture, lyrical/thematic tendencies,
    production character, and emotional register — expressed in terms of style, not biography
  - `style_tags`: list of genre/aesthetic tags (e.g., `[shoegaze, noise-rock, dream-pop]`)
  - `chromatic_hint`: optional human-filled dict with `temporal`, `spatial`, `ontological`
    keys indicating which mode the artist most strongly inhabits (e.g., `temporal: present`)
  - `chromatic_score`: dict populated by `--score-chromatic` run; contains `temporal`,
    `spatial`, `ontological`, `confidence`, `match` (match is against no specific target —
    reported as raw mode probabilities only)
  - `notes`: freeform string for White-project-specific framing, left blank by generation

#### Scenario: Catalog key format

- **WHEN** an artist name contains special characters or spaces (e.g., "My Bloody Valentine")
- **THEN** the YAML key SHALL be the exact artist name as it appears in `sounds_like`
- **AND** the `slug` field SHALL be the snake_case version for use in filenames

---

### Requirement: Artist Description Generation

The catalog CLI SHALL generate descriptions for artists not yet in the catalog, using the
Claude API with a prompt focused on aesthetic characterisation — no biographical details,
no lyrics, no content reproduction.

#### Scenario: Generate missing entries — production thread

- **WHEN** `artist_catalog.py --thread <dir> --generate-missing` is run
- **THEN** the tool scans all `production_plan.yml` files under the thread for `sounds_like`
  entries
- **AND** deduplicates across songs to produce a unique artist list
- **AND** for each artist not already in the catalog, calls the Claude API to generate a
  description with `status: draft`
- **AND** appends the new entries to `app/data/artist_catalog.yml`
- **AND** prints which artists were added and which were already present

#### Scenario: Generate missing entries — training data

- **WHEN** `artist_catalog.py --from-training-data --generate-missing` is run
- **THEN** the tool reads the `sounds_like` column from
  `training/data/training_data_with_embeddings.parquet`
- **AND** parses the comma-separated string, stripping `, discogs_id: \d+` fragments to
  extract clean artist names
- **AND** deduplicates across all rows and across any thread source to produce a unified
  unique artist list
- **AND** proceeds to generate descriptions for uncatalogued artists as above
- **AND** stores the `discogs_id` (if present) in the catalog entry for reference

#### Scenario: Discogs ID preservation

- **WHEN** a training-data artist entry has a Discogs ID
- **THEN** the catalog entry SHALL include a `discogs_id` field (integer or null)
- **AND** this ID is informational only — not used for scraping or API lookups

#### Scenario: Generation prompt constraints

- **WHEN** Claude is asked to describe an artist
- **THEN** the prompt SHALL instruct Claude to: describe sonic texture and production
  character, describe lyrical and thematic tendencies (not reproduce lyrics), describe
  emotional register, note if the artist is primarily instrumental, stay within 150 words,
  avoid biography (no birth dates, label history, chart positions)
- **AND** the prompt SHALL explicitly ask Claude NOT to reproduce any copyrighted text

#### Scenario: Unknown artist

- **WHEN** Claude cannot confidently describe an artist (e.g., very obscure)
- **THEN** the entry is written with `description: null`, `status: draft`, and a
  `notes` field pre-filled with "Unknown artist — fill description manually"
- **AND** the tool prints a warning for each such entry

#### Scenario: Idempotent re-run

- **WHEN** `--generate-missing` is run and all catalog entries already exist
- **THEN** no API calls are made
- **AND** the catalog file is not modified

---

### Requirement: Human Review Workflow

The catalog SHALL support a lightweight human review workflow where `status: draft`
entries can be edited and promoted to `status: reviewed`.

#### Scenario: Review prompt

- **WHEN** `artist_catalog.py --summary` is run
- **THEN** the tool prints: total artists, count with `status: draft`, count with
  `status: reviewed`, count with `chromatic_hint` filled, count with `chromatic_score`
  populated

#### Scenario: Fill in blanks template

- **WHEN** a `draft` entry is generated
- **THEN** the `chromatic_hint` field SHALL be written as a commented-out template:
  ```yaml
  # chromatic_hint:
  #   temporal: past | present | future
  #   spatial: thing | place | person
  #   ontological: imagined | forgotten | known
  ```
- **AND** the `notes` field SHALL be an empty string ready for the human to fill

---

### Requirement: ChromaticScorer Scoring of Descriptions

The catalog CLI SHALL optionally score all descriptions through ChromaticScorer in
text-only mode to provide an objective chromatic coordinate for each artist's description,
reported as raw mode probabilities (not matched against a specific color target).

#### Scenario: Score chromatic

- **WHEN** `artist_catalog.py --score-chromatic` is run (using `.venv312`)
- **THEN** for each entry with a non-null `description`, the description is encoded via
  DeBERTa and scored through the ONNX fusion model
- **AND** the `chromatic_score` field is written with temporal/spatial/ontological
  probability distributions and confidence
- **AND** entries with `description: null` are skipped with a warning

#### Scenario: Score does not override human hint

- **WHEN** `--score-chromatic` writes a `chromatic_score`
- **THEN** the `chromatic_hint` field (if filled by human) is NOT modified
- **AND** both fields coexist in the entry for comparison

---

### Requirement: Pipeline Prompt Injection

The lyric and chord generation pipelines SHALL inject artist catalog descriptions as style
context when a production plan's `sounds_like` entries match catalog entries, using
`reviewed` entries preferentially over `draft` entries.

#### Scenario: Lyric prompt injection

- **WHEN** `lyric_pipeline.py` is run and the production plan's `sounds_like` contains
  artists present in the catalog
- **THEN** a "STYLE REFERENCES" block is appended to the Claude prompt, listing each
  matching artist's name and description
- **AND** the block notes: "Use these as aesthetic reference only — do not imitate
  specific lyrics or identifiable phrases"

#### Scenario: Chord prompt injection

- **WHEN** `chord_pipeline.py` is run and `sounds_like` artists are in the catalog
- **THEN** style tags and chromatic_hint (if filled) from matching entries are appended
  to the prompt as production context

#### Scenario: Missing catalog entries

- **WHEN** a `sounds_like` artist is not in the catalog
- **THEN** the pipeline proceeds without injection and prints a note:
  "Artist '<name>' not in catalog — run artist_catalog.py --generate-missing to add"

#### Scenario: Injection uses reviewed entries preferentially

- **WHEN** both `draft` and `reviewed` entries might be injected
- **THEN** `reviewed` entries are always preferred
- **AND** `draft` entries are used as fallback with a console note: "Using draft
  description for '<name>' — consider reviewing"
