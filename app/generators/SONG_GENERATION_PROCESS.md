# Song Generation Process

Status labels: **[IMPLEMENTED]** | **[PARTIAL]** | **[CONCEPTUAL]**

This document tracks the full intended pipeline for generating a White project song,
from proposal to deliverable. Update as steps are added, refined, or promoted to full
implementation.

---

## 0. Pre-Production

### 0.1 Song Proposal **[IMPLEMENTED]**
- The Indigo Agent (or equivalent color agent) generates a `song_proposal_*.yml` containing
  concept, mood, key, BPM, singer, rainbow_color, infranym protocol, and optionally genres.
- Proposals live in `<thread>/yml/`.

### 0.2 Artist Catalog **[PARTIAL]**
- `sounds_like` artists from the composition proposal are catalogued in
  `app/reference/music/artist_catalog.yml` using `app/generators/artist_catalog.py`.
- Run: `python -m app.generators.artist_catalog --thread <thread_dir> --generate-missing`
- **Gap**: `collect_sounds_like()` only reads `production_plan.yml`, not
  `composition_proposal.yml`. Artists must currently be added manually or via a short
  script. Fix: extend `collect_sounds_like()` to also scan `composition_proposal.yml`.
- **Gap**: `CATALOG_DEFAULT_PATH` has a double-`app/` bug — pass `--catalog` explicitly
  until fixed.
- Catalog entries feed style context into lyric generation prompts.

---

## 1. Chord Phase **[IMPLEMENTED]**

```bash
python -m app.generators.midi.pipelines.chord_pipeline --production-dir <dir> --song-proposal <yml>
```

- Generates N candidate chord progressions using Markov graphs + theory scoring.
- Scores with Refractor (30% theory / 70% chromatic).
- Writes `chords/candidates/` + `chords/review.yml`.
- Human annotates `review.yml` (label, status: approved/accepted).
- Promote: `python -m app.generators.midi.production.promote_part --production-dir <dir> --phase chords`

---

## 2. Drum Phase **[IMPLEMENTED]**

```bash
python -m app.generators.midi.pipelines.drum_pipeline --production-dir <dir>
```

- Reads approved chord labels and section energy from plan.
- Selects drum templates by genre family and energy level.
- Scores and ranks. Writes `drums/candidates/` + `drums/review.yml`.
- Promote: `promote_part --phase drums`

---

## 3. Bass Phase **[IMPLEMENTED]**

```bash
python -m app.generators.midi.pipelines.bass_pipeline --production-dir <dir>
```

- Reads approved chords + kick onsets from drums.
- Generates bass lines (root, walking, pedal, arpeggiated, octave, syncopated styles).
- Clamps to bass register (MIDI 24–60), aligns with kick.
- Promote: `promote_part --phase bass`

---

## 4. Melody Phase **[PARTIAL — templates need redesign]**

```bash
.venv312/bin/python -m app.generators.midi.pipelines.melody_pipeline --production-dir <dir> --singer <name>
```

- Reads approved chords, generates melodic contour from interval templates.
- Enforces singer vocal range (`SINGERS` dict in `melody_patterns.py`).
- **Known issue**: Templates are too note-dense for vocal delivery — sounds like musical
  theatre rather than sparse folk/rock. See openspec `redesign-melody-templates`.
- **Known issue**: No `use_case` distinction between vocal and lead-instrument patterns.
- Promote: `promote_part --phase melody`
- Use `.venv312` (not `.venv`) — Refractor requires numpy 1.x / torch compat.

---

## 5. Composition Proposal **[IMPLEMENTED]**

```bash
.venv312/bin/python -m app.generators.midi.production.composition_proposal --production-dir <dir> --song-proposal <yml>
```

- Claude generates a full arrangement proposal with section order, energy arc, transition
  notes, sounds_like references, and compositional rationale.
- Writes `composition_proposal.yml`.
- Human uses this as a guide in Logic Pro.

---

## 6. Logic Pro Arrangement **[HUMAN STEP]**

- Open Logic Pro, assemble approved loops according to `composition_proposal.yml`.
- Save Logic project as `white_<color>_<title>.logicx` (one file per song).
- Export arrangement as `arrangement.txt` (Logic bar-position format).
- Save `arrangement.txt` and `arrangement_claude_1.txt` (Claude's original proposal)
  to the production directory for diff tracking.

---

## 7. Lyric Generation **[PARTIAL]**

```bash
python -m app.generators.midi.pipelines.lyric_pipeline --production-dir <dir> --song-proposal <yml>
```

- Generates N complete lyric drafts via Claude API.
- Scores each draft with Refractor (text-only mode, chromatic alignment).
- Writes `melody/lyrics_review.yml` + candidate drafts.
- Human selects and promotes: `promote_part --phase lyrics`
- Promoted draft preserved as `melody/lyrics_draft.txt` (pre-edit snapshot).
- **Gap**: Artist catalog context (sounds_like descriptions) not yet wired into the
  prompt — planned but not implemented.
- **Currently used as**: Hand-written lyrics placed directly in `melody/lyrics_draft.txt`.

---

## 8. ACE Studio — Vocal Synthesis **[PARTIAL]**

- Import approved melody MIDI loops into ACE Studio.
- Assign singer (matches `SINGERS` registry in pipeline).
- Enter lyrics section by section from `melody/lyrics_draft.txt`.
- Refine syllable fit, timing, and pitch manually.
- Export: ACE Studio writes `VocalSynthv*.mid` to production directory.

### ACE Studio MCP Integration **[BLOCKED]**
- `app/reference/mcp/ace_studio/` — MCP client built, feasibility probe run.
- **Status**: ACE Studio MCP server returns empty tools list. Blocked pending ACE update.
- See `app/reference/mcp/ace_studio/FEASIBILITY.md`.

---

## 9. Drift Report **[IMPLEMENTED]**

```bash
python -m app.generators.midi.production.drift_report --production-dir <dir>
```

- Parses ACE Studio MIDI export (`VocalSynthv*.mid`).
- Segments vocal events by arrangement sections (track 4 = melody).
- Compares pitch/rhythm against approved melody loops.
- Computes Levenshtein distance against `melody/lyrics_draft.txt`.
- Writes `drift_report.yml`.
- **Requires**: `arrangement.txt` in SMPTE timecode format (Logic Pro SMPTE export,
  not bar-position format). Bar-position format used in indigo song is not yet supported
  by drift report's timecode parser.

---

## 10. Song Evaluator **[IMPLEMENTED]**

```bash
python -m app.generators.midi.production.song_evaluator <production-dir>

# With ACE import metrics:
python -m app.generators.midi.production.song_evaluator <production-dir> --ace-import

# Re-score chromatic alignment against actual color target:
python -m app.generators.midi.production.song_evaluator <production-dir> --rescore

# Re-score lyrics post-ACE edit:
python -m app.generators.midi.production.song_evaluator <production-dir> --rescore-lyrics
```

- Reports: color, phases complete, total bars, vocal coverage, chromatic alignment,
  theory quality, production completeness, structural integrity, lyric maturity.
- Writes `song_evaluation.yml`.
- `--ace-import`: adds actual vocal coverage, syllable density from ACE export.
- Falls back to `arrangement.txt` (bar-position format) when no `production_plan.yml`.
- Falls back to `composition_proposal.yml` / `chords/review.yml` for color + title
  when no `production_plan.yml`.

---

## 11. Lyric Feedback Loop **[PARTIAL]**

```bash
# After ACE Studio editing:
python -m app.generators.midi.production.song_evaluator <dir> --rescore-lyrics

# Export (draft → edited) pairs for training:
python -m app.generators.midi.production.lyric_feedback_export \
    --thread shrink_wrapped/white-the-breathing-machine-learns-to-sing \
    --output lyric_feedback.jsonl
```

- Captures human edits to lyrics as (draft, edited) pairs with chromatic delta scores.
- Structured JSONL output for future prompt engineering or fine-tuning.
- **Gap**: No songs have completed ACE editing yet — feedback dataset is empty.

---

## 12. Infranym Encoding **[IMPLEMENTED]**

- Song proposals include an `INFRANYM PROTOCOL` block specifying the secret, encoding
  method (note cipher or morse duration), and BPM.
- The Indigo Agent generates a `VocalSynthv*.midi` infranym file (currently 0 bytes —
  see known bug).
- Manual fallback: `app/agents/tools/infranym_midi_tools.py` — `generate_morse_duration()`
  or `generate_note_cipher()`.
- Infranym MIDI goes in `<thread>/midi/<uuid>_i_morse_*.midi`.

---

## 13. Assembly Manifest **[PARTIAL]**

- `app/generators/midi/assembly_manifest.py` — parses Logic Pro arrangement export
  (SMPTE timecode format) into a structured section map.
- Updates `production_plan.yml` with real section timestamps and loop assignments.
- **Gap**: Not yet run on any song. Requires SMPTE-format arrangement export from Logic
  (different from the bar-position format currently used).

---

## 14. Refractor Rescoring **[PARTIAL]**

- `--rescore` flag re-evaluates chromatic alignment of all approved MIDI against the
  song's actual color target using the Refractor ONNX model.
- Sounds-like embeddings (5th modality, 3328-dim input) are computed but not yet passed
  to Refractor during pipeline scoring runs — scoring currently uses MIDI + concept only.
- **Gap**: `sounds_like_emb` from artist catalog not yet wired into chord/drum/bass/melody
  pipeline scoring calls.

---

## Known Bugs / Pending Fixes

| Bug                                                             | Location                                 | Priority                                        |
|-----------------------------------------------------------------|------------------------------------------|-------------------------------------------------|
| Indigo Agent infranym MIDI writes 0 bytes                       | `app/agents/indigo_agent.py`             | Medium                                          |
| `CATALOG_DEFAULT_PATH` double-`app/` path                       | `app/generators/artist_catalog.py`       | Low                                             |
| `collect_sounds_like()` misses `composition_proposal.yml`       | `app/generators/artist_catalog.py`       | Low                                             |
| Drift report timecode parser doesn't handle bar-position format | `app/generators/midi/drift_report.py`    | Low                                             |
| Melody templates too note-dense for vocal use                   | `app/generators/midi/melody_patterns.py` | High — see `redesign-melody-templates` openspec |
