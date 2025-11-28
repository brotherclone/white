# Yellow Agent Vertical Slice

## Overview

This vertical slice demonstrates the Yellow Agent's core functionality: generating Pulsar Palace RPG gameplay and extracting musical concepts from the resulting narratives.

## Architecture

The system consists of three main generators that work together:

### 1. Markov Room Generator (`app/generators/markov_room_generator.py`)

Generates room structures using Markov chain-style generation with two aesthetic influences:

- **Marienbad**: Elegant, surreal, baroque architecture
- **Ultraviolet Grasslands**: Weird, psychedelic, dangerous environments

**Features:**
- Room type classification (entrance, social, service, private, mystical, transitional)
- Atmosphere chains (elegant → surreal → weird → dangerous)
- Rich vocabulary of features (pulsing lights, impossible geometry, etc.)
- Narrative beat generation based on room type × atmosphere combinations

### 2. Character Action Generator (`app/generators/character_action_generator.py`)

Creates character actions by combining three trait dimensions:

- **Disposition** (Angry, Curious, Misguided, Clumsy, Cursed, Sick, Vengeful, Crazed)
- **Profession** (Doctor, Sailor, Breeder, Detective, Janitor, Spy, Librarian, Inventor, Tax Collector, Partisan)
- **Background** (Time period and location)

**Features:**
- Action templates for each disposition
- Professional modifiers that change how actions are performed
- Temporal context based on character's time period
- Encounter narrative generation with dynamic outcomes

### 3. Music Extractor (`app/generators/music_extractor.py`)

Translates narrative content into `SongProposalIteration` objects:

**Extracted Elements:**
- **Iteration ID**: Generated from room name and ID (e.g., `pulsar_palace_bridge_v001`)
- **Title**: Room name
- **BPM**: Integer (40-200) calculated from atmosphere type and narrative tension
- **Tempo**: Time signature (defaults to 4/4)
- **Key**: Musical key selected based on atmosphere and mood (major/minor)
- **Rainbow Color**: Always "Y" (Yellow) for Pulsar Palace content
- **Mood**: List of 1-5 descriptors derived from keywords in narrative
- **Genres**: List of 1-6 genres determined by atmosphere and mood combinations
- **Concept**: Substantive philosophical statement (minimum 100 characters) explaining the song's archetypal meaning

**Output Format:**
Returns `SongProposalIteration` objects compatible with the Rainbow Table song proposal system. These are proposals only—not complete production manifests with structure sections and audio tracks.

## Data Structures

### PulsarPalaceRoom (`app/structures/concepts/pulsar_palace_room.py`)

```python
class PulsarPalaceRoom:
    room_id: str
    name: str
    description: str
    atmosphere: str
    room_type: str
    exits: list[str]
    inhabitants: list[str]
    features: list[str]
    narrative_beat: str
```

### PulsarPalaceEncounter (`app/structures/concepts/pulsar_palace_room.py`)

```python
class PulsarPalaceEncounter:
    encounter_id: str
    room_id: str
    characters_involved: list[str]
    actions: list[str]
    narrative: str
    tension_level: int
```

### SongProposalIteration (`app/structures/manifests/song_proposal.py`)

```python
class SongProposalIteration:
    iteration_id: str
    bpm: int  # 40-200
    tempo: str | TimeSignature
    key: str | KeySignature
    rainbow_color: str | RainbowTableColor
    title: str
    mood: list[str]  # 1-20 items
    genres: list[str]  # 1-20 items
    concept: str  # min 100 chars, substantive philosophical content
```

## Running the Demo

```bash
python scripts/demo_yellow_agent_vertical_slice.py
```

### Demo Output

The script demonstrates the full pipeline:

1. **Generate Characters**: Creates 3 random party members with dispositions, professions, and backgrounds
2. **Generate Room**: Uses Markov chains to create a Pulsar Palace room
3. **Generate Actions**: Creates character actions based on trait combinations
4. **Extract Song Proposal**: Translates narrative into a `SongProposalIteration`
5. **Output Proposal**: Shows the proposal as JSON
6. **Output Complete Session**: Provides full context (room + characters + narrative + proposal)

### Example Output

```json
{
  "iteration_id": "pulsar_palace_bridge_v001",
  "bpm": 151,
  "tempo": "4/4",
  "key": "E major",
  "rainbow_color": "Y",
  "title": "Bridge",
  "mood": [
    "hypnotic",
    "rhythmic",
    "classical",
    "intense",
    "ornate"
  ],
  "genres": [
    "electronic",
    "dark ambient",
    "drone",
    "ambient",
    "experimental",
    "kosmische"
  ],
  "concept": "In the Bridge, the party encounters pulsing spaces where the boundaries of reality become permeable. The presence of Nobles trapped mid-gesture, Cursed valets in perfect stillness creates a tension between the organic and the cosmic, the temporal and the eternal. The pulsar's rhythmic influence—flickering between red and green to create the illusion of yellow—transforms this narrative of transitional spaces into sonic architecture. Each character action resonates with archetypal patterns: the disposition shapes emotional timbre, profession defines rhythmic approach, and temporal origin creates harmonic context."
}
```

## Integration with Existing Systems

### Character System

Uses the existing `PulsarPalaceCharacter` structure from `app/structures/concepts/pulsar_palace_character.py`:
- Disposition, profession, and background tables
- ON/OFF stat system
- Dice rolling utilities from `app/agents/tools/gaming_tools.py`

### Song Proposal System

Output is `SongProposalIteration` objects that feed into the production pipeline:
- Metadata fields (bpm, tempo, key, rainbow_color) establish musical parameters
- Mood and genre vocabularies draw inspiration from existing manifests (`staged_raw_material/04_01` through `04_10`)
- Concept statements provide philosophical/archetypal grounding for later production work
- **Important**: This is a proposal only—structure sections, audio tracks, and production details are handled by other agents

## Future Enhancements

### Markov Chain Improvements
- Train on actual room descriptions from adventure modules
- Add state transitions for room-to-room flow
- Incorporate player choice branching

### Character Action System
- Add consequence tracking (affect ON/OFF stats)
- Implement character interactions and relationships
- Create action resolution mechanics

### Music Proposal Enhancement
- Add more sophisticated concept generation with deeper archetypal analysis
- Map narrative tension to additional musical parameters beyond BPM
- Create character-specific musical signatures (leitmotifs)
- Link proposals to player choices and session history

### Full Yellow Agent Integration
- Connect to LangGraph workflow
- Add LLM enhancement for narrative quality
- Implement session management and save states

## Design Philosophy

This vertical slice embodies the Yellow Agent's core purpose: transforming procedurally generated gameplay into song proposals. The Markov chain approach ensures:

- **Reproducibility**: Same seed = same room
- **Variety**: Large vocabulary space = unique experiences
- **Coherence**: Curated transitions maintain aesthetic consistency
- **Translation**: Narrative elements map cleanly to song proposal parameters

The system demonstrates that gameplay can be both mechanically generated and artistically meaningful, with the "found poetry" of random combinations creating unexpected narrative moments that inspire song proposals.

**Key Distinction**: The Yellow Agent proposes songs based on gameplay narratives. It does **not** create complete production manifests with structure sections, timecodes, or audio tracks. Those are the responsibility of production agents downstream in the pipeline. The Yellow Agent's role is to translate the archetypal and emotional content of gameplay into musical metadata that other agents can use to create finished works.
