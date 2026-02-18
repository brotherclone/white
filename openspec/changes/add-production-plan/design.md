# Design: Production Plan

## Context

Every song in the production pipeline has a `production/<song_slug>/` directory containing phase subdirectories (`chords/`, `drums/`, `harmonic_rhythm/`, `strums/`, `bass/`, `melody/`). Each phase's `approved/` directory holds MIDI loops named after section labels (e.g., `verse.mid`, `chorus.mid`).

What's missing: a document that says *how those loops fit together as a song*. The production plan fills this gap.

## Goals / Non-Goals

**Goals:**
- Define song section sequence, bar counts, and repeat counts
- Capture vocals intent per section
- Give drum fills (and future phases) structural context
- Bridge toward the final `Manifest` (which requires structure + timestamps)
- Be human-editable after generation

**Non-Goals:**
- Replacing the song proposal (that's upstream of production)
- Storing loop file references (derived dynamically at assembly time)
- Capturing final timestamps (those come at render time)
- Generating audio

## Production Plan Schema

```yaml
# production_plan.yml
song_slug: sequential_dissolution_v2
generated: 2026-02-18T10:00:00Z
source_proposal: thread/.../song_proposal.yml   # path reference only

# Song-level metadata (copied from proposal, human may extend)
title: Sequential Dissolution
bpm: 120
time_sig: 4/4
key: F# minor
color: Black
vocals_planned: true       # are there any vocals at all?
sounds_like: []            # list of {name, discogs_id} — human fills in

# Section sequence — human edits order and repeat counts
sections:
  - name: intro
    bars: 4          # derived from approved chord loop (or default N*1)
    repeat: 1
    vocals: false
    notes: ""

  - name: verse
    bars: 8
    repeat: 2
    vocals: true
    notes: ""

  - name: chorus
    bars: 4
    repeat: 2
    vocals: true
    notes: ""

  - name: bridge
    bars: 8
    repeat: 1
    vocals: false
    notes: "instrumental break"

  - name: outro
    bars: 4
    repeat: 1
    vocals: false
    notes: ""
```

## Bar Count Derivation

Priority order for each section:
1. Approved harmonic rhythm MIDI length (most accurate — reflects the actual loop)
2. Approved chord MIDI length (N chords × uniform 1 bar)
3. Default: `n_chords × 1` bar (from chord review candidate metadata)

The generator reads these in priority order and reports which source it used.

## Section Order in Generated Plan

The generator reads the chord review YAML and outputs sections in the order the human labeled them during the chord approval phase. This preserves the human's intent from the labeling step. The human then reorders by editing the YAML if needed.

## Drum Pipeline Integration

The drum pipeline today is section-aware (it reads chord labels and generates per-section patterns) but doesn't know what comes *next*. The production plan adds this.

Change: when `production_plan.yml` exists in the production directory, the drum pipeline reads it and annotates the review YAML with `next_section` for each entry:

```yaml
candidates:
  - section: Verse
    next_section: Chorus   # from production plan
    label: verse_01
    status: pending
```

This annotation is informational for now. The future drum-fills phase will use it to generate fill bar variants.

## Relation to Final Manifest

The `Manifest` model requires:
- `bpm`, `tempo`, `key`, `rainbow_color`, `title` ← from proposal (already in plan)
- `release_date`, `album_sequence` ← editorial, out of scope
- `main_audio_file`, `TRT` ← from final render, out of scope
- `vocals: bool`, `lyrics: bool` ← from `vocals_planned` + sections
- `sounds_like` ← human fills in plan
- `structure: list[ManifestSongStructure]` ← can be derived from plan sections + bar counts + BPM (bar count × bars-per-minute = seconds)
- `lrc_file` ← from lyrics phase, out of scope
- `audio_tracks` ← from rendered stems, out of scope

At the end of production, a manifest bootstrap command can read the production plan and emit a partial `Manifest` YAML with all derivable fields pre-filled, leaving render-time fields blank.

## Risks / Trade-offs

- **Manual editing required**: The plan is only useful if the human actually fills in repeat counts. Mitigation: sensible defaults (repeat: 1) mean it's valid without any editing.
- **Drift from approved loops**: If the human re-runs a phase and changes which loops are approved, the plan's bar counts may be stale. Mitigation: the generator can be re-run with `--refresh` to update bar counts from current approved loops.
- **Section label mismatches**: The plan uses section names as keys to match loops in `approved/` directories. Labels must be consistent. Mitigation: the generator warns when a plan section name has no corresponding file in any phase's `approved/` directory.
