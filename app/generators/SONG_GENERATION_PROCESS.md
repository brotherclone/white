# Song Generation Process

Status labels: **[IMPLEMENTED]** | **[PARTIAL]** | **[CONCEPTUAL]**

This document tracks the full pipeline for generating a White project song.
Steps marked **[HUMAN STEP]** require manual action in Logic Pro or ACE Studio.

> **Python environment**: Use `.venv312` for all pipeline commands — Refractor requires numpy 1.x / torch compatibility (transformers 4.x; `.venv` has 5.x which blocks `torch.load`).
> Example: `.venv312/bin/python app/generators/midi/pipelines/chord_pipeline.py ...`

> **Handoff from agents**: The White agent chain runs to shrinkwrap (`app/util/shrinkwrap_chain_artifacts.py`), producing a human-readable directory under `shrink_wrapped/<song-title>/`. Production pipeline steps (chord gen onward) are manual from there — chord generation is **not** auto-triggered from the agent.

---

## 0. Pre-Production

### 0.1 Song Proposal **[IMPLEMENTED]**
The color agent generates `song_proposal_*.yml` with concept, mood, key, BPM, singer, rainbow_color, and genres. Proposals live in `<thread>/yml/`.

### 0.2 Artist Catalog **[PARTIAL]**
```bash
.venv312/bin/python -m app.generators.artist_catalog --thread <thread_dir> --generate-missing
```
Catalogs `sounds_like` artists from the proposal into `app/reference/music/artist_catalog.yml`. Feeds style context into lyric generation.

**Known gaps**:
- `collect_sounds_like()` only reads `production_plan.yml`, not `song_proposal.yml` — add artists manually until fixed.
- `CATALOG_DEFAULT_PATH` has a double-`app/` bug — pass `--catalog` explicitly.

---

## 1. Chord Phase **[IMPLEMENTED]**

```bash
.venv312/bin/python app/generators/midi/pipelines/chord_pipeline.py \
    --thread shrink_wrapped/<song-title> \
    --song <iteration_id>.yml \
    --num-candidates 200 --top-k 10
```

`--song` is a filename inside `<thread>/yml/`. The pipeline creates `<thread>/production/<iteration_id>/chords/`. Generates candidates via Markov + theory scoring + Refractor chromatic scoring.

**Human step**: Open `chords/review.yml`, set `label` and `status: approved` on chosen candidates.

```bash
PYTHONPATH=/Volumes/LucidNonsense/White \
.venv312/bin/python -m app.generators.midi.production.promote_part \
    --review shrink_wrapped/<song-title>/production/<slug>/chords/review.yml
```

---

## 2. Drum Phase **[IMPLEMENTED]**

```bash
.venv312/bin/python -m app.generators.midi.pipelines.drum_pipeline --production-dir <dir>
```

Reads approved chord sections, picks drum templates by genre family + energy. Writes `drums/candidates/` + `drums/review.yml`.

**Human step**: Label and approve in `drums/review.yml`, then promote.

```bash
.venv312/bin/python -m app.generators.midi.production.promote_part \
    --review <dir>/drums/review.yml
```

---

## 3. Bass Phase **[IMPLEMENTED]**

```bash
.venv312/bin/python -m app.generators.midi.pipelines.bass_pipeline --production-dir <dir>
```

Reads approved chords + kick onsets. Generates bass lines (root, walking, pedal, arpeggiated, octave styles). Clamps to MIDI 24–60. Writes `bass/candidates/` + `bass/review.yml`.

**Human step**: Label and approve, then promote.

```bash
.venv312/bin/python -m app.generators.midi.production.promote_part \
    --review <dir>/bass/review.yml
```

---

## 4. Melody Phase **[PARTIAL]**

```bash
.venv312/bin/python -m app.generators.midi.pipelines.melody_pipeline --production-dir <dir>
```

Singer is auto-detected from the song proposal. Reads approved chords, generates melodic contour from interval templates, enforces vocal range. Writes `melody/candidates/` + `melody/review.yml`.

**Human step**: Label and approve, then promote.

```bash
.venv312/bin/python -m app.generators.midi.production.promote_part \
    --review <dir>/melody/review.yml
```

**Known issue**: Templates tend to be note-dense — sounds more like musical theatre than sparse folk/rock. See openspec `redesign-melody-templates`.

---

## 5. Composition Proposal **[IMPLEMENTED — optional]**

```bash
.venv312/bin/python -m app.generators.midi.production.composition_proposal \
    --production-dir <dir> --song-proposal <yml>
```

Claude proposes a full arrangement arc (section order, energy, transitions, sounds_like references, rationale). Writes `composition_proposal.yml`. Use as a guide in Logic — not binding.

---

## 6. Logic Pro Arrangement **[HUMAN STEP]**

- Assemble approved loops in Logic Pro, using `composition_proposal.yml` as a guide (or compose freely).
- Export arrangement as `arrangement.txt` (Logic bar-position format).
- Save Claude's proposal as `arrangement_claude.txt` alongside for diff tracking.

---

## 7. Production Plan **[IMPLEMENTED]**

```bash
.venv312/bin/python -m app.generators.midi.production.production_plan \
    --production-dir <dir> --song-proposal <yml>
```

Reads approved chord sections and bar counts. Writes `production_plan.yml` with section order, bar lengths, and BPM. Edit to set repeat counts and vocal flags.

---

## 8. Lyric Generation **[IMPLEMENTED]**

```bash
.venv312/bin/python -m app.generators.midi.pipelines.lyric_pipeline --production-dir <dir>
```

Generates 3 lyric drafts via Claude API, scores each with Refractor (chromatic alignment). Writes `melody/lyrics_review.yml` + candidate `.txt` files.

**Human step**: Read candidates, set `status: approved` on chosen draft in `lyrics_review.yml`, then promote.

```bash
.venv312/bin/python -m app.generators.midi.production.promote_part \
    --review <dir>/melody/lyrics_review.yml
```

Promotes to `melody/lyrics.txt`. Original draft preserved as `melody/lyrics_draft.txt`.

**Known gap**: Artist catalog `sounds_like` context not yet wired into the prompt.

---

## 9. Assembly **[IMPLEMENTED]**

```bash
.venv312/bin/python -m app.generators.midi.production.assembly_manifest \
    --production-dir <dir> \
    --arrangement <dir>/arrangement.txt \
    --assemble
```

Reads `arrangement.txt` and approved loop MIDIs. Assembles full-length MIDI files (one per track) into `assembled/`. Updates `production_plan.yml` with actual section timings and writes a structural `drift_report.yml` (section timing vs computed plan).

Output: `assembled/assembled_chords.mid`, `assembled_drums.mid`, `assembled_bass.mid`, `assembled_melody.mid`.

---

## 10. ACE Studio — Vocal Synthesis **[PARTIAL]**

```bash
.venv312/bin/python -m app.generators.midi.production.ace_studio_export \
    --production-dir <dir>
```

Pushes `assembled/assembled_melody.mid` + `melody/lyrics.txt` to the open ACE Studio project via MCP (localhost:21572). Sets BPM, time signature, loads singer, inserts all notes with lyrics.

**Singer mapping**: White project singer names map to ACE Studio voice names via `app/reference/mcp/ace_studio/singer_voices.yml`. Only Shirley → Ember Rose is confirmed; others are null. If the singer isn't resolved automatically, select the voice manually in ACE Studio.

**Human step in ACE Studio**:
- Confirm or select the correct voice (see `singer_voices.yml` for the mapping).
- Refine syllable fit, timing, and phrasing.
- Render and export the MIDI (`VocalSynthv*.mid`). Save to `melody/` or the production root.

**Post-render** — parse ACE Studio's MIDI export back to LRC:
```bash
.venv312/bin/python -m app.generators.midi.production.ace_studio_import \
    --production-dir <dir>
```
Writes `vocal_alignment.lrc`.

---

## 11. Drift Report **[IMPLEMENTED]**

```bash
.venv312/bin/python -m app.generators.midi.production.drift_report --production-dir <dir>
```

Compares ACE Studio vocal export against approved melody loops. Looks for `VocalSynthv*.mid` in the production root or `melody/` subfolder. Supports both Logic Pro export formats (SMPTE timecode and bar/beat); BPM and time signature are read from `production_plan.yml`.

Reports per-section pitch match %, rhythm drift (beats), note count delta, word count, and a global lyric edit distance (Levenshtein) against `melody/lyrics.txt`.

**Note**: Overwrites the structural `drift_report.yml` written by assembly manifest in step 9.

---

## 12. Song Evaluator **[IMPLEMENTED]**

```bash
# Basic evaluation:
.venv312/bin/python -m app.generators.midi.production.song_evaluator <dir>

# Include ACE import metrics (after vocal render):
.venv312/bin/python -m app.generators.midi.production.song_evaluator <dir> --ace-import

# Re-score chromatic alignment:
.venv312/bin/python -m app.generators.midi.production.song_evaluator <dir> --rescore

# Re-score lyrics after ACE editing:
.venv312/bin/python -m app.generators.midi.production.song_evaluator <dir> --rescore-lyrics
```

Reports color, phases complete, total bars, vocal coverage, chromatic alignment, theory quality, production completeness, and lyric maturity. Writes `song_evaluation.yml`.

**Known gap**: `--rescore-lyrics` reads concept+color from `production_plan.yml`; for arrangement-first songs with no plan, these are empty. Pass `--song-proposal` as a fallback (not yet implemented).

---

## 13. Lyric Feedback Export **[PARTIAL]**

```bash
.venv312/bin/python -m app.generators.midi.production.lyric_feedback_export \
    --thread shrink_wrapped/white-the-breathing-machine-learns-to-sing \
    --output lyric_feedback.jsonl
```

Exports (draft → edited) lyric pairs with chromatic delta scores as JSONL for future prompt engineering or fine-tuning. Needs 20+ pairs for reliable few-shot injection, 100+ for LoRA.

---

## 14. Infranym Encoding **[PARTIAL]**

Song proposals include an `INFRANYM PROTOCOL` block. Manual tool: `app/agents/tools/infranym_midi_tools.py` (`generate_morse_duration()` or `generate_note_cipher()`). Infranym MIDI goes in `<thread>/midi/<uuid>_i_morse_*.midi`.

**Known bug**: Indigo Agent infranym MIDI writes 0 bytes (`app/agents/indigo_agent.py`).

---

## Known Bugs / Pending Fixes

| Bug                                                          | Location                                              | Priority                                        |
|--------------------------------------------------------------|-------------------------------------------------------|-------------------------------------------------|
| Indigo Agent infranym MIDI writes 0 bytes                    | `app/agents/indigo_agent.py`                          | Medium                                          |
| `CATALOG_DEFAULT_PATH` double-`app/` path                    | `app/generators/artist_catalog.py`                    | Low                                             |
| `collect_sounds_like()` misses `song_proposal.yml`           | `app/generators/artist_catalog.py`                    | Low                                             |
| Melody templates too note-dense for vocal use                | `app/generators/midi/patterns/melody_patterns.py`     | High — see `redesign-melody-templates` openspec |
| `singer_voices.yml` not wired into `ace_studio_export.py`    | `app/generators/midi/production/ace_studio_export.py` | Medium — see `wire-singer-voices-registry` openspec |
| `--rescore-lyrics` needs `--song-proposal` fallback          | `app/generators/midi/production/song_evaluator.py`    | Low                                             |
| Drift report overwrites structural drift from assembly step  | `app/generators/midi/production/drift_report.py`      | Low — consider separate filenames               |
| `.venv` uses transformers 5.x — blocks `torch.load` on torch 2.2.2 | `pyproject.toml` / `.venv`                     | Medium — use `.venv312` for all pipeline steps  |
