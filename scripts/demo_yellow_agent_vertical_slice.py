#!/usr/bin/env python3
"""
Yellow Agent Vertical Slice Demo

This script demonstrates the full Yellow Agent flow:
1. Generate a Pulsar Palace room using Markov chains (Marienbad × Ultraviolet Grasslands)
2. Generate character actions based on disposition + profession + background
3. Extract a SongProposalIteration from the resulting narrative
4. Output the song proposal for the production pipeline

Run with: python scripts/demo_yellow_agent_vertical_slice.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.structures.generators.character_action_generator import (
    CharacterActionGenerator,
)
from app.structures.generators.markov_room_generator import MarkovRoomGenerator
from app.structures.generators.music_extractor import MusicExtractor
from app.structures.concepts.pulsar_palace_character import PulsarPalaceCharacter


def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def print_section(title: str, content: str):
    """Print a formatted section"""
    print(f"\n--- {title} ---")
    print(content)
    print()


def main():
    print_header("YELLOW AGENT VERTICAL SLICE DEMO")
    print(
        "Generating one room of Pulsar Palace gameplay and extracting musical concepts..."
    )

    print_section(
        "STEP 1: Generate Random Characters",
        "Creating party members with random dispositions, professions, and backgrounds",
    )

    num_characters = 3
    characters = []
    for i in range(num_characters):
        char = PulsarPalaceCharacter.create_random()
        characters.append(char)
        print(f"Character {i + 1}:")
        print(f"  Disposition: {char.disposition.disposition}")
        print(f"  Profession: {char.profession.profession}")
        print(f"  Background: {char.background.place} ({char.background.time})")
        print(f"  ON: {char.on_current}/{char.on_max}")
        print(f"  OFF: {char.off_current}/{char.off_max}")
        print()

    print_section(
        "STEP 2: Generate Room (Markov Chain)",
        "Using Markov chains with Marienbad × Ultraviolet Grasslands aesthetics",
    )

    room_generator = MarkovRoomGenerator()
    room = room_generator.generate_room(room_number=1)

    print(f"Room: {room.name}")
    print(f"Type: {room.room_type}")
    print(f"Atmosphere: {room.atmosphere}")
    print("\nDescription:")
    print(f"  {room.description}")
    print("\nNarrative Beat:")
    print(f"  {room.narrative_beat}")
    print("\nFeatures:")
    for feature in room.features:
        print(f"  - {feature}")
    print("\nInhabitants:")
    for inhabitant in room.inhabitants:
        print(f"  - {inhabitant}")
    print(f"\nExits: {', '.join(room.exits)}")

    print_section(
        "STEP 3: Generate Character Actions",
        "Creating wild actions based on character trait combinations",
    )

    action_generator = CharacterActionGenerator()
    encounter_narrative = action_generator.generate_encounter_narrative(
        room.description, characters
    )

    print("Full Encounter Narrative:")
    print(f"  {encounter_narrative}")
    print()

    print("Individual Character Actions:")
    for i, char in enumerate(characters):
        action = action_generator.generate_action(char)
        print(f"  {i + 1}. {action}")

    print_section(
        "STEP 4: Extract Song Proposal",
        "Translating narrative into a SongProposalIteration (mood, BPM, key, genres, concept)",
    )

    music_extractor = MusicExtractor()
    proposal = music_extractor.extract_song_proposal(room, encounter_narrative)

    print(f"Iteration ID: {proposal.iteration_id}")
    print(f"Title: {proposal.title}")
    print(f"BPM: {proposal.bpm}")
    print(f"Key: {proposal.key}")
    print(f"Tempo: {proposal.tempo}")
    print(f"Rainbow Color: {proposal.rainbow_color}")
    print(f"\nMood: {', '.join(proposal.mood)}")
    print(f"\nGenres: {', '.join(proposal.genres)}")
    print("\nConcept:")
    concept_words = proposal.concept.split()
    for i in range(0, len(concept_words), 12):
        print(f"  {' '.join(concept_words[i:i+12])}")

    print_section(
        "STEP 5: Song Proposal as JSON",
        "SongProposalIteration object formatted for the Yellow Agent",
    )

    print(json.dumps(proposal.model_dump(), indent=2))

    print_section(
        "STEP 6: Complete Session Data",
        "Full context including room, characters, narrative, and song proposal",
    )

    full_output = {
        "room": {
            "room_id": room.room_id,
            "name": room.name,
            "type": room.room_type,
            "atmosphere": room.atmosphere,
            "description": room.description,
            "narrative_beat": room.narrative_beat,
            "features": room.features,
            "inhabitants": room.inhabitants,
            "exits": room.exits,
        },
        "characters": [
            {
                "disposition": char.disposition.disposition,
                "profession": char.profession.profession,
                "background": {
                    "place": char.background.place,
                    "time": char.background.time,
                },
                "stats": {
                    "on": f"{char.on_current}/{char.on_max}",
                    "off": f"{char.off_current}/{char.off_max}",
                },
            }
            for char in characters
        ],
        "encounter_narrative": encounter_narrative,
        "song_proposal": proposal.model_dump(),
    }

    print(json.dumps(full_output, indent=2))

    print_header("VERTICAL SLICE COMPLETE")
    print("The Yellow Agent successfully:")
    print("  ✓ Generated a room using Markov chains")
    print("  ✓ Created character actions from trait combinations")
    print("  ✓ Extracted a SongProposalIteration from the narrative")
    print("  ✓ Produced a song proposal ready for the production pipeline")
    print("\nThis demonstrates the core gameplay → song proposal pipeline.\n")


if __name__ == "__main__":
    main()
