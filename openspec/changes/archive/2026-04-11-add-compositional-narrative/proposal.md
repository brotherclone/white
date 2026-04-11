# Change: Compositional Narrative

## Why

The loops were a scaffold. They proved the pipeline could generate musically
coherent material section by section. But a collection of individually good
loops assembled in order is not the same thing as a song. A song knows where
it's going. It has a centre of gravity. The listener feels the shape of it even
if they can't name what they're feeling.

The `add-section-mood-arcs` proposal adds a tension map — a number per section.
That's necessary but not sufficient. Real compositional thinking isn't a curve.
It's a set of intentions that cut across all dimensions of the music at once:
where the harmonic weight lives, when the texture opens up, how the rhythm shifts
to mark a structural arrival, which instrument leads in which section and which
drops back to support. These decisions interact. A chorus that's supposed to feel
bigger can't just have higher energy — it needs the bass to move more, the hi-hat
to open up, the melody to sit in a higher register than the verse, the chord
voicing to spread wider. All of those things happening together is what makes the
listener's chest open.

None of that is possible when bass, drums, and melody are generated independently
from a shared tension value. The pipeline needs a compositional document — a
genuine artistic brief written before any loop generation starts — that all phases
read and serve.

This proposal is for that document: the `composition_narrative.yml`, written by
Claude as a composer, describing the full dynamic and textural intention of the
song before a single MIDI note is generated.

## What Changes

### `composition_narrative.yml` — the compositional brief

A new file written to the production directory by `composition_proposal.py`
alongside (and superseding the intent of) the current composition proposal.

The narrative is a multi-dimensional description of the song's intended shape.
It lives at the section level but carries information that crosses section
boundaries — how sections relate to each other, what the song is building toward,
what it leaves behind.

**Schema:**

```yaml
schema_version: "1"
generated_by: claude
rationale: |
  This is an elegy that moves like grief does — not in a clean arc but in
  waves. The opening feels like a normal Tuesday, which makes what comes
  after worse. I've put the emotional peak at chorus_2 rather than the end
  because grief doesn't resolve; the outro should feel like sitting with
  something rather than release. The bridge is the pivot — the moment the
  song admits what it's actually about. Everything before the bridge is
  preparation. Everything after is the aftermath.

the_moment:
  section: chorus_2
  description: |
    This is the song. Everything else is approach or departure. The melody
    should feel like something finally said out loud. The bass should move
    for the first time with purpose. The drums should open — less hi-hat,
    more room, so the space itself carries weight.

sections:
  intro:
    arc: 0.10
    register: low          # bass and low-mid only — no high frequency content
    texture: sparse        # single sustained voice, long rests
    harmonic_complexity: simple   # tonic only or simple triad, no extensions
    rhythm_character: absent      # drums barely present or silent
    lead_voice: bass       # bass carries the opening — melody enters late
    narrative: |
      Before anything has happened. The song hasn't admitted its subject yet.
      A sustained tone that could be anything.

  verse_1:
    arc: 0.30
    register: low_mid
    texture: sparse
    harmonic_complexity: simple
    rhythm_character: minimal     # kick and ghost snare only
    lead_voice: melody
    narrative: |
      The subject arrives but quietly, as if testing whether it's allowed to
      be here. The melody should feel uncertain — stepwise, short phrases
      with long rests between. The bass follows but doesn't push.

  chorus_1:
    arc: 0.65
    register: mid
    texture: moderate
    harmonic_complexity: moderate  # 7ths and sus chords introduced
    rhythm_character: present      # full pattern, but not the biggest it gets
    lead_voice: melody
    contrast_with: verse_1
    narrative: |
      Bigger than the verse but deliberately held back. This chorus should
      feel like relief that's uncertain of itself. Not the peak — the peak
      is coming. Introduce the open hi-hat here but keep the kick simple.

  verse_2:
    arc: 0.40
    register: low_mid
    texture: moderate             # slightly denser than verse_1 — we've been here before
    harmonic_complexity: simple
    rhythm_character: minimal
    lead_voice: melody
    narrative: |
      The second verse carries the weight of having already heard the chorus.
      It knows more. Slightly busier bass, but the melody still restrained.
      The harmonic rhythm can slow down here — the song is settling in.

  bridge:
    arc: 0.15
    register: low
    texture: near_absent
    harmonic_complexity: tense    # unresolved — suspended, no clear tonic
    rhythm_character: absent
    lead_voice: none              # no single voice leads — texture only
    narrative: |
      The pivot. Everything stops pretending. This should feel like the song
      holding its breath — suspended harmony, near-silence, and then just
      enough bass to keep the listener grounded. The melody is gone. The
      drums are gone. What remains is the question the song has been asking.

  chorus_2:
    arc: 0.90
    register: mid_high
    texture: full
    harmonic_complexity: rich     # full voicings, extensions, the works
    rhythm_character: open        # kick pattern, open hi-hat, more space than density
    lead_voice: melody
    contrast_with: bridge
    is_the_moment: true
    narrative: |
      This is the song. Everything shifts register — bass octave up from
      where it's been, melody in the upper range, drums fully present but
      with space rather than density. The chord voicings spread wide. The
      contrast with the bridge should feel like the held breath finally
      released. Not triumphant — lamentful at full volume.

  outro:
    arc: 0.20
    register: low_mid
    texture: sparse
    harmonic_complexity: simple
    rhythm_character: minimal
    lead_voice: bass
    narrative: |
      Aftermath. The melody is mostly gone. The bass carries the memory of
      the chorus but slower. This shouldn't resolve cleanly — it should feel
      like sitting with something that doesn't go away. End on the tonic but
      without feeling like arrival.
```

### How Claude generates this

`composition_proposal.py` is extended with a `generate_narrative()` function.
The prompt gives Claude:

- The full song proposal (concept, color, key, BPM, time signature, singer)
- The `sounds_like` artists and `style_reference_profile` (if available)
- The section sequence from the production plan
- The chromatic synthesis document for the color
- The `aesthetic_hints` from init

The prompt asks Claude to:
1. Write the `rationale` — why this shape, in plain language
2. Designate `the_moment` — one section, and why
3. For each section, write the arc value, the four dimension values (register,
   texture, harmonic_complexity, rhythm_character), the `lead_voice`, and a
   one-paragraph `narrative`

The four dimension values are controlled vocabularies:

- **register**: `low`, `low_mid`, `mid`, `mid_high`, `high`
- **texture**: `absent`, `near_absent`, `sparse`, `moderate`, `full`
- **harmonic_complexity**: `simple`, `moderate`, `tense`, `rich`
- **rhythm_character**: `absent`, `minimal`, `present`, `busy`, `open`

These vocabularies are intentionally small so the pipeline can map them to
concrete generation choices without ambiguity.

### How generation phases read the narrative

Each pipeline phase consults `composition_narrative.yml` for the current
section and maps dimension values to generation constraints:

**Drum pipeline:**

| rhythm_character | effect |
|---|---|
| `absent` | prefer `ambient_pulse` or silence; no snare |
| `minimal` | ghost snare, kick on 1 only; prefer `ghost_verse` |
| `present` | standard selection — moderate weight |
| `busy` | prefer dense patterns; penalise sparse |
| `open` | prefer patterns with open hi-hat over closed; reduce kick density |

**Bass pipeline:**

| texture + lead_voice | effect |
|---|---|
| `absent` texture | prefer rest-heavy templates; root drone at very low velocity |
| `lead_voice: bass` | prefer templates with movement; bass carries the section |
| `lead_voice: melody` | prefer pedal/drone; bass supports, doesn't compete |
| register `low` | clamp to C1–C2 range |
| register `mid_high` | allow C3–C4; permit octave jumps |

**Melody pipeline:**

| narrative dimension | effect |
|---|---|
| `lead_voice: none` | generate very sparse melody or skip section |
| `lead_voice: melody` | standard generation |
| register `low_mid` | clamp to lower range of singer voice |
| register `mid_high` | prefer upper range; starting pitch from upper chord tones |
| `harmonic_complexity: tense` | allow notes outside chord tones (suspension, passing) |
| `harmonic_complexity: simple` | strong preference for chord tones on strong beats |

**Chord pipeline:**

| harmonic_complexity | effect |
|---|---|
| `simple` | triads only; no extensions; root position |
| `moderate` | 7ths and sus chords permitted |
| `rich` | full voicings; 9ths and extensions; inversions |
| `tense` | diminished and half-diminished; sus4 unresolved; raised 4ths |

### `contrast_with` — inter-section relationship

When a section declares `contrast_with: <other_section>`, the pipeline checks
that the generated candidates for this section differ meaningfully from the
referenced section in at least two dimensions. If they don't — e.g. chorus_1
ends up sparse and minimal when it's supposed to contrast with verse_1 which is
also sparse — the pipeline warns before writing candidates:

```
⚠ chorus_1 should contrast with verse_1 but generated candidates are similar
  in texture (both sparse) and rhythm_character (both minimal).
  Consider reviewing the narrative or regenerating with --force-contrast.
```

This is a compositional continuity check, not a block — the human decides.

### `the_moment` in the evaluator

`song_evaluator.py` gains a `moment_score` metric: how well the designated
`the_moment` section stands out from the sections around it. Computed as the
mean deviation of the moment section's energy across all dimensions from its
immediate neighbours. A song that peaks at the right place scores higher than
one where the moment gets lost in a uniform texture.

`moment_score` feeds into `structural_integrity` in the composite score.

### Narrative as living document

The `composition_narrative.yml` is intended to be edited. Claude proposes;
the human refines. The section `narrative` fields are free text and never
parsed — they exist for the human and for Claude to read in future sessions when
revisiting the song. They are the artistic memory of what was intended.

The `rationale` and `the_moment.description` fields are carried into
`song_evaluation.yml` as reference so the evaluator output includes a reminder
of what the song was trying to do:

```yaml
# song_evaluation.yml
compositional_intent: |
  This is an elegy that moves like grief does — not in a clean arc but in waves.
the_moment: chorus_2
moment_score: 0.82
arc_delta: 0.14
```

## Impact

- Affected specs: `composition-proposal`, `production-plan`, `chord-pipeline`,
  `drum-pipeline`, `bass-pipeline`, `melody-pipeline`, `song-evaluator`
- Depends on: `add-section-mood-arcs` (arc field), `add-pipeline-orchestrator`
  (status display)
- New files:
  - `app/generators/midi/production/composition_narrative.py` — generation,
    parsing, section constraint extraction
  - `app/structures/music/narrative_constraints.py` — Pydantic models for
    section constraints and the full narrative document
- Modified files:
  - `app/generators/midi/production/composition_proposal.py` — `generate_narrative()`
  - `app/generators/midi/pipelines/chord_pipeline.py` — harmonic_complexity mapping
  - `app/generators/midi/pipelines/drum_pipeline.py` — rhythm_character mapping
  - `app/generators/midi/pipelines/bass_pipeline.py` — texture + lead_voice mapping
  - `app/generators/midi/pipelines/melody_pipeline.py` — register + lead_voice mapping
  - `app/generators/midi/production/song_evaluator.py` — moment_score, intent recall
  - `app/generators/midi/production/pipeline_runner.py` — display narrative rationale
    in status when present
- Tests:
  - `tests/generators/midi/production/test_composition_narrative.py`
    - Narrative round-trips cleanly through YAML
    - Section constraint extraction produces correct generation hints for each
      dimension value
    - `contrast_with` check fires correctly when sections are similar
    - `moment_score` calculation gives higher scores when the peak section
      differs meaningfully from neighbours
  - Pipeline tests: dimension values shift candidate rankings in expected direction
