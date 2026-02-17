# Melody & Lyrics Generation — Design

## 1. MelodyPattern dataclass

```
@dataclass
class MelodyPattern:
    name: str                          # e.g. "stepwise_ascend_4"
    contour: str                       # stepwise, arpeggiated, repeated, leap_step, pentatonic, scalar_run
    energy: str                        # low, medium, high
    time_sig: tuple[int, int]
    description: str
    intervals: list[int]              # signed semitone deltas from previous note (0 = start)
    rhythm: list[float]               # onset positions in beats (same length as intervals)
    durations: list[float] | None     # note durations in beats; None = sustain to next onset
```

Key difference from bass: melody templates define **relative intervals** (signed semitones from previous note), not chord-tone selections. The pipeline resolves the starting pitch from the chord root within the singer's range, then walks the interval sequence, clamping notes to stay in range.

### Contour types

| Contour | Description | Typical sections |
|---------|-------------|-----------------|
| stepwise | Conjunct motion (1-2 semitone steps) | Verse |
| arpeggiated | Chord-tone outlines (3-4-5 semitone leaps) | Chorus, bridge |
| repeated | Same note or narrow oscillation | Verse (speech-like) |
| leap_step | Large leap followed by stepwise return | Chorus climax |
| pentatonic | Major/minor pentatonic scale runs | Any |
| scalar_run | Scale-wise ascending/descending passages | Bridge, transitions |

## 2. Singer registry

```python
SINGERS = {
    "busyayo":   SingerRange("Busyayo",   45, 64, "baritone"),
    "gabriel":   SingerRange("Gabriel",    48, 67, "tenor"),
    "robbie":    SingerRange("Robbie",     48, 67, "tenor"),
    "shirley":   SingerRange("Shirley",    53, 72, "alto"),
    "katherine": SingerRange("Katherine",  57, 76, "alto"),
}
```

Singer selection:
1. If song proposal specifies `singer`, use that.
2. Otherwise, infer from key center — pick singer whose comfortable mid-range best covers the song's tonic.
3. CLI `--singer` override always wins.

## 3. Melody resolution

Given a `MelodyPattern` and a chord voicing:

1. **Starting pitch**: chord root, transposed into singer's comfortable mid-range (midpoint ± 5 semitones).
2. **Walk intervals**: apply each signed interval delta. If result goes outside singer range, mirror the interval (e.g., +3 becomes -3).
3. **Strong-beat chord-tone snap**: on strong beats (beat 1, beat 3 in 4/4), if the note is not a chord tone, snap to nearest chord tone within 2 semitones.
4. **Phrase endings**: last note of a phrase resolves to root or 5th.

## 4. Template inventory (target: 18+ templates)

### 4/4 time (12+)
- stepwise × 3 energy levels (ascending, descending, wave)
- arpeggiated × 2 energy levels (up, down)
- repeated × 2 (monotone verse, oscillating)
- leap_step × 2 (leap up, leap down)
- pentatonic × 2 (major, minor)
- scalar_run × 1 (bridge passage)

### 7/8 time (6+)
- stepwise × 2 (grouped 3+2+2, 2+2+3)
- arpeggiated × 2 (asymmetric emphasis)
- repeated × 1 (driving pulse)
- pentatonic × 1 (modal)

## 5. Theory scoring

Three components, equally weighted:

### Singability (0.0–1.0)
- **Interval penalty**: large leaps (> octave) penalized, stepwise motion rewarded
- **Range usage**: melodies using < 50% of available range penalized (too static)
- **Rest placement**: at least one rest per 4 bars (breathing room)

### Chord-tone alignment (0.0–1.0)
- Fraction of strong-beat notes that are chord tones (root, 3rd, 5th)
- Passing tones on weak beats are fine and don't penalize

### Contour quality (0.0–1.0)
- **Arch shape**: melody should have a high point (climax) roughly 2/3 through the section
- **Variety**: penalize excessive note repetition (> 4 consecutive same pitches)
- **Resolution**: final note should resolve to stable chord tone

### Composite
```
theory_score = mean(singability, chord_tone_alignment, contour_quality)
composite = 0.30 * theory_score + 0.70 * chromatic_score
```

## 6. Pipeline flow

```
read approved chords (voicings + labels)
read approved harmonic rhythm (durations)
read approved bass (for contrary motion awareness — optional)
read song proposal (key, BPM, time_sig, color, singer)
read chromatic synthesis (thematic reference — displayed in review.yml, not scored)

for each section:
    select templates by time_sig + energy
    for each template:
        resolve melody notes against chord voicings + singer range
        write candidate MIDI (channel 0, melody register)
        score: theory + chromatic
    rank by composite, keep top-k

write candidates/ MIDI files
write review.yml (includes chromatic synthesis excerpt per section for reference)
```

## 7. Vocal synthesis workflow (external)

After melody promotion, the human workflow is:

1. Import approved melody MIDI into **ACE Studio** (commercial SVS)
2. ACE Studio handles syllable parsing and phoneme alignment natively
3. Human adds lyrics in ACE Studio editor (Claude can draft lyrics from chromatic synthesis docs)
4. Render vocal audio in ACE Studio
5. Assembly in Logic Pro (loop grid → timeline)

No pipeline code is needed for this step — ACE Studio accepts standard MIDI directly.

## 8. Output structure

```
<song>/melody/
    candidates/
        <section>_<contour>_<pattern_name>.mid
    approved/
        <human-labeled>.mid
    review.yml
    lyrics.txt          (human-written or Claude-drafted, for reference)
```

## 9. MIDI channel

Melody uses **MIDI channel 0** (same as bass — separate files, not combined).
Note range: determined by singer assignment (45–76 depending on singer).
