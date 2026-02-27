# Change: Add Lyric Generation Pipeline

## Why
Vocal sections need lyrics, and lyrics need to be generated, scored against the song's
chromatic target, reviewed, and promoted — consistent with every other production phase.
The ChromaticScorer already supports lyric text via DeBERTa embeddings; this change
wires it into a repeatable pipeline.

## What Changes
- New `app/generators/midi/lyric_pipeline.py` — generates N candidate lyric drafts via
  Claude API, scores each with ChromaticScorer (text-only mode), writes candidates +
  `melody/lyrics_review.yml`
- `promote_part.py` extended to handle `.txt` files (not just `.mid`)
- Lyric format convention: `[section_name]` headers + plain text lines per phrase
- `song_evaluator.py` already reads `melody/lyrics.txt` — no changes required downstream

## Design Decisions

### All-or-nothing candidates
Unlike MIDI phases where each section has independent candidates, lyrics are holistic —
verse and chorus lyrics must tell one coherent story. Candidates are **complete song
lyric sets** (all vocal sections), not per-section files.

### Scoring: chromatic only, no theory
- Chromatic alignment: DeBERTa encodes the full lyric text, fusion model scores
  temporal/spatial/ontological against the song's color target (null audio/MIDI inputs)
- No automated "theory" scoring — rhyme, meter, and poetic quality are human-evaluated
- review.yml stores chromatic scores alongside status/notes

### Generation via Claude API
`lyric_pipeline.py` calls the Anthropic SDK with a structured prompt built from:
- Song concept + color + target chromatic modes
- Approved melody contour types per vocal section (informs phrase rhythm)
- BPM + time signature (informs syllable density)
- Per-section syllable targets derived from note counts (e.g. "verse: 12–17 syllables")
The pipeline is self-contained; no external prompt management needed.
Default model: `claude-sonnet-4-6` (overridable via `--model`).

### Note source for fitting score
Note counts come from `melody/approved/<label>*.mid` multiplied by `section.repeat` —
not from a merged `melody/melody.mid`, which is never written by the pipeline.  This
means fitting scores are available without running the assembly step first.

### Syllable counting
Vowel-cluster heuristic: count contiguous vowel-character groups per word, floor 1.
No NLP dependency.  Comment lines (`# ...`) and section headers (`[name]`) are stripped
before counting.

### Incremental review file
`lyrics_review.yml` is append-only.  Existing entries (including human `status` edits)
are never overwritten.  A `--sync-candidates` flag registers manually-written `.txt`
files in `melody/candidates/` as stub entries without scoring them.

## Impact
- Affected specs: (new) `lyric-generation`
- Affected code: `app/generators/midi/lyric_pipeline.py` (new),
  `app/generators/midi/promote_part.py` (minor extension)
- Yellow song: first song to use this pipeline; three scored candidates already written
  as the worked example in `melody/candidates/lyrics_0{1,2,3}.txt`
