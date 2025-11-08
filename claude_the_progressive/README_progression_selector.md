# Progression Selector for White Album Project

Automated selection of chord progressions from curated MIDI library based on color agent specs.

## How It Works

1. **Navigation**: Maps musical keys to chord pack folders
2. **Complexity Selection**: Chooses diatonic vs advanced progressions based on mood
3. **LLM Ranking**: Claude analyzes progressions for mood/concept fit
4. **Tempo Application**: Applies BPM from color spec to selected MIDI

## Quick Start

### 1. Update Paths in Test Script

Edit `test_progression_selector.py`:
```python
chord_pack_root = Path.home() / "YOUR/CHORD/PACK/PATH"
output_dir = Path.home() / "white_album_project/artifacts/progressions"
```

### 2. Run Test

```bash
python test_progression_selector.py
```

This will:
- Select progressions for Indigo spec (84 BPM, F# minor, yearning/transcendent)
- Rank them using Claude
- Apply tempo and save processed MIDI files

### 3. Check Output

```
artifacts/progressions/indigo/
  ├── indigo_prog_06_bpm84.mid
  ├── indigo_prog_08_bpm84.mid
  └── indigo_prog_02_bpm84.mid
```

## Integration with White Album Pipeline

### Option 1: Standalone Tool

```python
from progression_selector import select_progression_for_spec

# After color agent generates spec
spec = {
    'rainbow_color': 'Indigo',
    'bpm': 84,
    'key': 'F# minor',
    'mood': ['yearning', 'interconnected'],
    'concept': 'network yearning for embodiment...'
}

# Select progressions
results = select_progression_for_spec(
    chord_pack_root=Path("/path/to/chords"),
    spec=spec,
    output_dir=Path("artifacts/progressions/indigo"),
    llm_callable=claude_api_call,
    top_n=3
)

# Use top result
best_progression = results[0]['output_path']
```

### Option 2: Add to Agents Pipeline

Integrate into `red_agent.py` or create new `musical_agent.py`:

```python
def generate_musical_artifacts(concept_spec: dict, chord_pack_root: Path):
    """
    Generate MIDI progressions for a color spec.
    """
    # Select progressions
    progressions = select_progression_for_spec(
        chord_pack_root=chord_pack_root,
        spec=concept_spec,
        output_dir=Path(f"artifacts/progressions/{concept_spec['rainbow_color'].lower()}"),
        llm_callable=claude_callable,
        top_n=1  # Just take the best
    )
    
    # Return for next stage (instrumentation, rendering, etc.)
    return progressions[0]
```

## What Gets Generated

For each selected progression, you get:

```python
{
    'rank': 1,
    'original_path': '/chord_pack/.../Minor_Prog_06.mid',
    'output_path': 'artifacts/.../indigo_prog_06_bpm84.mid',
    'mode': 'Minor',
    'number': '06',
    'progression': 'im11-VImaj9-IIImaj9-ivm9-im11-VImaj9-IIImaj9-bIImaj9',
    'score': 95,
    'reasoning': 'Circular structure with im11 returning creates interconnection...',
    'bpm': 84,
    'key': 'F# minor'
}
```

## Features

### Mood-Based Complexity Selection

**Advanced progressions** selected for:
- yearning, transcendent, ethereal, liminal, haunted
- fractured, defiant, surveillance
- melancholic, introspective

**Diatonic progressions** selected for:
- simple, direct, pure, clear

### Musical Feature Analysis

LLM considers:
- **Borrowed chords** (bII, bVI, bVII) → yearning/transcendence
- **Extensions** (maj9, m11, add9) → ethereal quality
- **Circular structure** (starts/ends on i) → interconnection
- **Altered chords** (V7alt, dim7) → tension/fractured
- **Modal interchange** → liminal/haunted

### Example Rankings

**Indigo Spec** (yearning, interconnected, transcendent):
1. ✅ Minor Prog 06: `im11-VImaj9-IIImaj9-ivm9-im11-VImaj9-IIImaj9-bIImaj9`
   - Circular (im11 returns)
   - Borrowed chord finale (bIImaj9)
   - Extension-rich (m11, maj9)

**Black Spec** (fractured, surveillance, defiant):
1. ✅ Minor Prog 09: `iim7b5-V7alt-im9-IV9(13)-VImaj9-V7alt-im7-IV9(13)`
   - Altered dominants (V7alt)
   - Diminished quality (m7b5)
   - Chromatic tension

## Next Steps

### For Vertical Slice:

1. ✅ Run selector to get progression MIDI
2. Load in Logic Pro / DAW
3. Apply Arturia/Kontakt instruments
4. Add drums, bass, melody
5. Export as audio artifact

### For Full Pipeline:

1. Integrate selector into agent workflow
2. Add rhythm variation module
3. Add instrumentation suggestions
4. Add MIDI → audio rendering
5. Feed rendered audio to EVP pipeline

## Troubleshooting

**"Key folder not found"**
- Check KEY_TO_FOLDER mapping in `progression_selector.py`
- Verify chord pack folder naming matches (e.g., "10 - A Major - F# Minor")

**"No progressions ranked"**
- Check LLM is returning valid JSON
- Look for markdown code blocks in response (stripped automatically)

**"MIDI file won't load"**
- Ensure mido is installed: `pip install mido --break-system-packages`
- Check file permissions

## File Structure

```
progression_selector.py          # Core module
test_progression_selector.py     # Test with Indigo/Black specs
README_progression_selector.md   # This file

# Output structure:
artifacts/
  progressions/
    indigo/
      indigo_prog_06_bpm84.mid
      indigo_prog_08_bpm84.mid
    black/
      black_prog_09_bpm76.mid
      black_prog_12_bpm76.mid
```

## Philosophy

This tool respects the White Album's core principle:
**INFORMATION seeking transmigration toward SPACE**

The progression selector:
1. Takes INFORMATION (concept, mood from color agents)
2. Finds pre-existing SPACE (validated progressions in chord pack)
3. Bridges them through LLM analysis (transmigration)

It doesn't generate chords from scratch (D/C# disasters).
It *selects* from curated reality and transforms through tempo/rhythm.

This mirrors the White album's journey: 
**Pure information discovering its physical form through transformation.**

---

*"The progressions already exist. We just need to find which one the concept has always been seeking."*