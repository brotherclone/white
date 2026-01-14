# Change: Add Melody and Lyrics Generator

## Why
Chord progressions provide harmonic structure, but melodies carry the conceptual narrative and rebracketing markers. The melody generator must create singable vocal lines that align with lyrics, fit the harmonic structure, and embody the ontological transformations expressed in the text.

## What Changes
- Add `MelodyGenerator` class for generating melodic contours over chord progressions
- Implement lyrics-to-melody alignment using syllable counting and prosody
- Add melodic scoring functions (contour smoothness, range, harmonic fit, climax placement)
- Implement rebracketing marker emphasis (melodic highlighting of ontological pivots)
- Add rhythm pattern generation synchronized with lyric syllables
- Support for call-and-response, verse-chorus contrast, and bridge transitions
- Integration with chord progressions (harmonic notes, passing tones, neighbor tones)
- MIDI export with lyrics embedded as meta-events

## Impact
- Affected specs: melody-generator (new capability)
- Affected code:
  - `app/generators/midi/melody/` (new directory)
  - `app/generators/midi/melody/generator.py` - main melody generation
  - `app/generators/midi/melody/prosody.py` - lyrics-to-rhythm alignment
  - `app/generators/midi/melody/scoring.py` - melodic quality scoring
  - Integration with existing `ChordProgressionGenerator`
- Dependencies: pronouncing (for syllable counting), python-Levenshtein (for lyric alignment)
- Complexity: High - melody is more constrained than chords (must fit lyrics AND harmony)

## Design Considerations

### Lyrics-First Approach
Unlike typical melody generators, we start with **lyrics** (concept text with rebracketing):
1. Parse lyrics into syllables
2. Identify rebracketing markers
3. Generate rhythm from prosody (stressed/unstressed syllables)
4. Create melodic contour fitting rhythm
5. Align melody to harmony (chord tones, passing tones)

### Rebracketing Marker Emphasis
Ontological pivot points should be melodically prominent:
- Higher pitch for rebracketing markers
- Longer durations
- Harmonic emphasis (chord tones vs passing tones)
- Melodic contour changes (ascending to marker, descending after)

### Singability Constraints
Generated melodies must be performable by human singers:
- Reasonable vocal range (2-3 octaves max, typically less)
- Stepwise motion preferred over large leaps
- Breathing points at phrase boundaries
- Avoid awkward intervals (augmented seconds, etc.)

### Integration with Training Models (Future)
Eventually, trained models can guide melody:
- Chromatic style model suggests melodic character for each mode
- Temporal sequence model suggests melodic pacing across verses
- Rebracketing classifier validates ontological emphasis

But for now, use music theory + search + scoring.
