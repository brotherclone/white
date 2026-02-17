# Add Melody & Lyrics Generation

## Motivation

The music production pipeline currently generates chords, drums, harmonic rhythm, strums, and bass — all instrumental layers. Melody is the final compositional element before the song is ready for vocal synthesis and human production.

The Chromatic Synthesis documents contain agent-generated lyrical themes per section. These serve as creative direction, not final lyrics. The human writes actual lyrics informed by that thematic material, then the pipeline generates singable melodies that fit the harmonic and rhythmic context.

Vocal synthesis uses **ACE Studio** (commercial singing voice synthesis), which imports standard MIDI files and handles lyrics/syllable alignment natively in its editor. The melody pipeline outputs standard MIDI; the human imports it into ACE Studio, adds lyrics, and renders vocal audio.

## Approach

### Pipeline

**Melody MIDI generation** (`melody_patterns.py` + `melody_pipeline.py`) — same generate → score → human gate → promote pattern as all other phases. Template-based melodic contour generation constrained to singer vocal ranges.

### Key design decisions

1. **Vocal range as first-class constraint**: Each singer has a defined range. Song proposals can specify a singer (or the pipeline infers one from the key). Melodies are hard-clamped to the assigned range.

2. **Contour templates, not absolute pitches**: Like bass patterns, melody templates define contour shapes (stepwise up, leap down to chord tone, repeated note, etc.) that the pipeline resolves against the current chord and vocal range.

3. **Section-aware generation**: Reads approved chord labels to determine verse/chorus/bridge structure. Different contour behaviors per section type (verse = more stepwise, chorus = wider range, bridge = contrasting).

4. **Rhythmic alignment with harmonic rhythm**: Melody note onsets align with the harmonic rhythm grid. Syllable density varies by section energy.

5. **Theory scoring**: Singability (interval size, range usage, rest placement), chord-tone alignment (strong beats on chord tones), and melodic contour quality (arch shapes, climax placement).

6. **Lyrics are human-authored**: The pipeline does NOT generate lyrics. Chromatic Synthesis text is thematic reference. Claude can draft lyrics from the song proposal and chromatic synthesis docs; the human refines them and adds them in ACE Studio.

### Singer roster

| Singer | Range | MIDI Range | Voice Type |
|--------|-------|------------|------------|
| Busyayo | A2–E4 | 45–64 | Baritone |
| Gabriel | C3–G4 | 48–67 | High Baritone/Tenor |
| Robbie (Fortune) | C3–G4 | 48–67 | High Baritone/Tenor |
| Shirley | F3–C5 | 53–72 | Low Alto |
| Katherine | A3–E5 | 57–76 | High Alto |

### What this change does NOT include

- Actual lyric generation (human task, with optional Claude drafting from chromatic synthesis)
- ACE Studio vocal rendering (external commercial tool, not part of this pipeline)
- Harmony/backing vocal generation (future phase)
- Audio rendering or mixing
