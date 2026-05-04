## MODIFIED Requirements

### Requirement: Artist Description Generation

The catalog CLI SHALL generate descriptions for artists not yet in the catalog, using the
Claude API with a prompt focused on aesthetic characterisation — no biographical details,
no lyrics, no content reproduction.

`generate_missing()` SHALL also accept an explicit `artists` list as a programmatic call
path, in addition to the existing CLI modes. When called this way it MUST deduplicate the
list against the existing catalog and generate descriptions only for uncatalogued entries.

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

#### Scenario: Generate missing entries — explicit artist list

- **WHEN** `generate_missing(artists=["Artist A", "Artist B"])` is called programmatically
- **THEN** each artist not already in the catalog receives a generated description with
  `status: draft`
- **AND** artists already present in the catalog are skipped silently
- **AND** the function returns a list of newly added artist slugs

#### Scenario: Discogs ID preservation

- **WHEN** a training-data artist entry has a Discogs ID
- **THEN** the catalog entry SHALL include a `discogs_id` field (integer or null)
- **AND** this ID is informational only — not used for scraping or API lookups

#### Scenario: Generation prompt constraints

- **WHEN** Claude is asked to describe an artist
- **THEN** the prompt SHALL instruct Claude to: describe sonic texture and production
  character, describe lyrical and thematic tendencies (not reproduce lyrics), describe
  emotional register, note if the artist is primarily instrumental, stay within 150 words
