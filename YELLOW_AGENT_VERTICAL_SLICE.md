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

Translates narrative content into musical metadata:

**Extracted Elements:**
- **Mood**: Derived from keywords in narrative (baroque, cosmic, mysterious, etc.)
- **BPM**: Calculated from atmosphere type and narrative tension
- **Key**: Selected based on atmosphere and mood (major/minor)
- **Structure**: Song sections mapped from narrative beats
- **Genres**: Determined by atmosphere and mood combinations
- **Concept**: Generated summary statement

**Output Format:**
Compatible with existing YAML song manifests (see `staged_raw_material/04_01` through `04_10`)

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

## Running the Demo

```bash
python scripts/demo_yellow_agent_vertical_slice.py
```

### Demo Output

The script demonstrates the full pipeline:

1. **Generate Characters**: Creates 3 random party members with dispositions, professions, and backgrounds
2. **Generate Room**: Uses Markov chains to create a Pulsar Palace room
3. **Generate Actions**: Creates character actions based on trait combinations
4. **Extract Music**: Translates narrative into musical concepts
5. **Output YAML**: Formats as a song manifest
6. **Output JSON**: Provides complete data structure

### Example Output

```yaml
bpm: 93.24
manifest_id: 'yellow_generated_01'
tempo: 4/4
key: C major
rainbow_color: Y
title: Gallery
mood:
  - lush
  - baroque
  - rhythmic
  - mysterious
  - intense
genres:
  - art pop
  - experimental
  - kosmische
  - ambient
structure:
  - section_name: Gathering
    start_time: '[00:00.000]'
    end_time: '[00:30.000]'
    description: The party enters the space
  # ... more sections
concept: In the Gallery, the party encounters opulent spaces...
```

## Integration with Existing Systems

### Character System

Uses the existing `PulsarPalaceCharacter` structure from `app/structures/concepts/pulsar_palace_character.py`:
- Disposition, profession, and background tables
- ON/OFF stat system
- Dice rolling utilities from `app/agents/tools/gaming_tools.py`

### YAML Manifests

Output format matches existing song manifests in `staged_raw_material/04_01` through `04_10`:
- Same metadata fields (bpm, tempo, key, rainbow_color)
- Compatible structure format
- Matching mood and genre vocabularies

## Future Enhancements

### Markov Chain Improvements
- Train on actual room descriptions from adventure modules
- Add state transitions for room-to-room flow
- Incorporate player choice branching

### Character Action System
- Add consequence tracking (affect ON/OFF stats)
- Implement character interactions and relationships
- Create action resolution mechanics

### Music Extraction
- Integrate with actual music generation (MIDI output)
- Map narrative tension to dynamic BPM changes
- Create leitmotifs for recurring characters/locations

### Full Yellow Agent Integration
- Connect to LangGraph workflow
- Add LLM enhancement for narrative quality
- Implement session management and save states

## Design Philosophy

This vertical slice embodies the Yellow Agent's core purpose: transforming procedurally generated gameplay into musical experiences. The Markov chain approach ensures:

- **Reproducibility**: Same seed = same room
- **Variety**: Large vocabulary space = unique experiences
- **Coherence**: Curated transitions maintain aesthetic consistency
- **Translation**: Narrative elements map cleanly to musical parameters

The system demonstrates that gameplay can be both mechanically generated and artistically meaningful, with the "found poetry" of random combinations creating unexpected narrative moments that inspire musical expression.
