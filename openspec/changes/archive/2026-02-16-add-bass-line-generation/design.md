# Design: Bass Line Generation

## Bass Pattern Templates

### Template Structure

```python
@dataclass
class BassPattern:
    name: str              # e.g. "root_whole", "walking_quarter"
    style: str             # root, walking, pedal, arpeggiated, octave, syncopated
    energy: str            # low, medium, high
    time_sig: tuple[int, int]
    description: str
    notes: list[tuple[float, str, str]]
    # Each note: (beat_position, tone_selection, velocity_level)
    # tone_selection: "root", "5th", "3rd", "octave_up", "octave_down",
    #                 "chromatic_approach", "passing_tone"
```

### Tone Selection Rules

Templates specify *which chord tone* to play, not absolute pitches:

| Selection            | Resolution                                           |
|----------------------|------------------------------------------------------|
| `root`               | Lowest instance of chord root in bass register       |
| `5th`                | Perfect 5th above root                               |
| `3rd`                | 3rd above root (major or minor, from chord voicing)  |
| `octave_up`          | Root + 12 semitones (clamped to register ceiling)    |
| `octave_down`        | Root - 12 semitones (clamped to register floor)      |
| `chromatic_approach` | Semitone below next chord's root (approach note)     |
| `passing_tone`       | Scale-wise step between current and next chord tones |

Approach and passing tones require knowledge of the *next* chord in the progression. For the last chord in a section, these fall back to `root`.

### Template Inventory (Target)

**4/4 templates (~15):**

| Style       | Energy | Name                   | Description                                       |
|-------------|--------|------------------------|---------------------------------------------------|
| root        | low    | root_whole             | Root on beat 1, held for whole bar                |
| root        | medium | root_half              | Root on 1 and 3                                   |
| root        | high   | root_quarter           | Root on every beat                                |
| octave      | low    | octave_sparse          | Root on 1, octave on 3                            |
| octave      | medium | octave_bounce          | Root-octave alternating on quarters               |
| octave      | high   | octave_eighth          | Root-octave alternating on eighths                |
| walking     | medium | walking_quarter        | Root-3rd-5th-approach on quarters                 |
| walking     | high   | walking_eighth         | Quarter note walking + eighth passing tones       |
| arpeggiated | low    | arp_root_5th           | Root on 1, 5th on 3                               |
| arpeggiated | medium | arp_triad              | Root-3rd-5th on quarters, rest on 4               |
| arpeggiated | high   | arp_full               | Root-3rd-5th-octave on quarters                   |
| pedal       | low    | pedal_whole            | Root held entire bar (same as root_whole)         |
| pedal       | medium | pedal_pulse            | Root repeated on quarters (same note, re-attacked)|
| syncopated  | medium | syncopated_offbeat     | Root on 1, 5th on 2.5                             |
| syncopated  | high   | syncopated_funk        | Root on 1, ghost on 1.5, 5th on 2.5, root on 3.5 |

**7/8 templates (~5):**

| Style    | Energy | Name               | Description                             |
|----------|--------|--------------------|-----------------------------------------|
| root     | low    | root_7_sparse      | Root on 1 only                          |
| root     | medium | root_7_322         | Root on group starts (1, 1.5, 2.5)     |
| octave   | medium | octave_7_bounce    | Root-octave on group boundaries         |
| walking  | medium | walking_7          | Walking on group start positions        |
| arp      | medium | arp_7_322          | Root-5th-3rd on group starts            |

## Pipeline Flow

```
Inputs:
  chords/approved/*.mid    → chord voicings (roots, tones)
  chords/review.yml        → section labels, BPM, color
  drums/approved/*.mid     → kick pattern onsets
  harmonic_rhythm/review.yml → chord durations per section

Processing (per section):
  1. Extract chord roots + available tones from chord MIDI
  2. Extract kick onsets from drum MIDI
  3. Read harmonic rhythm durations (fallback: 1 bar/chord)
  4. For each template:
     a. Resolve tone selections to MIDI notes
     b. Generate MIDI (bass register, channel 0)
     c. Compute theory score:
        - root_adherence: % of strong-beat notes that are chord root
        - kick_alignment: % of bass onsets coinciding with kick hits
        - voice_leading:  smoothness (small intervals between chords)
     d. Score with ChromaticScorer
     e. Compute composite: 30% theory + 70% chromatic

Output:
  bass/candidates/<id>.mid
  bass/review.yml
  bass/approved/  (empty, awaiting human promotion)
```

## Theory Score Components

### Root Adherence (0.0 - 1.0)
Fraction of notes on strong beats (1 and 3 in 4/4) that are the chord root. Higher = more harmonically grounded.

### Kick Alignment (0.0 - 1.0)
Fraction of bass note onsets that coincide with a kick drum hit (within a tolerance of 1/16 note). Higher = tighter groove.

### Voice Leading (0.0 - 1.0)
Measures the smoothness of bass movement between adjacent chords. Scored inversely proportional to interval size:
- Unison/semitone: 1.0
- Whole step: 0.9
- Minor 3rd: 0.8
- Major 3rd: 0.7
- Perfect 4th/5th: 0.5
- Greater: 0.3

The theory score is the mean of these three components.

## MIDI Details

- Channel: 0 (standard bass channel)
- Register: MIDI notes 24-60 (C1 to C4)
- Note duration: defined by template (whole, half, quarter, eighth)
- Velocity: accent=100, normal=80, ghost=50 (slightly different from drums — bass doesn't need as wide a dynamic range)
- Ticks per beat: 480 (matching all other pipelines)
