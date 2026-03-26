## ADDED Requirements

### Requirement: White Lyric Cut-Up Mode
When the song is White, the lyric pipeline SHALL collect approved lyric text from each
sub-proposal listed in the song context and supply it to Claude as explicit source material,
instructing a cut-up: extract phrases and images from the source lyrics, recombine and
transform them into a coherent new lyric that feels synthesised rather than collaged.

Sub-lyric collection SHALL:
1. For each sub-proposal directory, check `melody/candidates/lyrics_review.yml` for entries
   with `status: approved` and load those files.
2. If no review file exists, load all `melody/candidates/lyrics_*.txt` files as candidates.
3. If a sub-proposal has no `melody/candidates/` directory, skip it silently.

The prompt MUST preserve the section structure (vocal section headers) of the White
production plan. Non-White lyric generation is unchanged.

#### Scenario: approved sub-lyrics used as cut-up source

- **WHEN** a White lyric pipeline run has three sub-proposals with approved lyric files
- **THEN** the Claude prompt includes a `## Source Lyrics` section with labeled excerpts
  from each sub-song
- **AND** Claude is instructed to cut up and recombine the phrases, not generate from scratch
- **AND** the output lyric files follow the same section-header format as non-White candidates

#### Scenario: fallback when no approved sub-lyrics available

- **WHEN** the sub-proposal directories have no approved lyric files
- **THEN** the pipeline falls back to standard lyric generation (concept-driven)
- **AND** a note is included in the prompt identifying this as a White synthesis song

#### Scenario: non-White lyric pipeline unchanged

- **WHEN** the lyric pipeline is run for any color other than White
- **THEN** sub-lyric collection is not performed; the standard prompt is used
