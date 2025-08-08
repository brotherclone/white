import random

from typing import Any, Dict, Optional
from transformers import AutoModel

from app.agents.BaseRainbowAgent import BaseRainbowAgent
from app.enums.chord_quality import ChordQuality
from app.objects.chord import Chord
from app.objects.chord_progression import ChordProgression
from app.objects.rainbow_song_meta import RainbowSongStructureModel


class Nancarrow(BaseRainbowAgent):

    model: Any = None
    midi_data: Any = None
    chord_templates: Dict[str, list[list[str]]] = {}
    key_signatures: Dict[str, Dict[str, list[str]]] = {}

    def __init__(self, **data):
        super().__init__(**data)
        self.model = None
        self.chord_templates = self._load_chord_templates()
        self.key_signatures = self._load_key_signatures()

    def initialize(self):
        self.agent_state = None
        if not self.llm_model_name:
            self.llm_model_name = "gpt2"
        self.model = AutoModel.from_pretrained(self.llm_model_name, use_safetensors=True).to(self.device)


    @staticmethod
    def _load_chord_templates() -> dict[str, list[list[str]]]:
        """Load common chord progression templates by section type"""
        return {
            "verse": [
                ["I", "V", "vi", "IV"],  # Classic pop progression
                ["vi", "IV", "I", "V"],  # vi-IV-I-V
                ["I", "vi", "IV", "V"],  # 50s progression
                ["i", "VII", "VI", "VII"],  # Minor progression
                ["i", "iv", "VII", "III"],  # Dorian-flavored
                ["I", "II", "V", "I"],  # Jazz-influenced
            ],
            "chorus": [
                ["I", "V", "vi", "IV"],
                ["vi", "IV", "I", "V"],
                ["I", "IV", "vi", "V"],
                ["I", "vi", "ii", "V"],
                ["i", "VI", "III", "VII"],
            ],
            "bridge": [
                ["ii", "V", "I", "vi"],
                ["IV", "V", "iii", "vi"],
                ["vi", "ii", "V", "I"],
                ["iii", "vi", "ii", "V"],
            ],
            "intro": [
                ["I", "V"],
                ["vi", "IV"],
                ["I", "vi", "IV", "V"],
            ],
            "outro": [
                ["I", "V", "I"],
                ["vi", "IV", "I"],
                ["I", "IV", "I"],
            ],
            "interlude": [
                ["vi", "IV", "I", "V"],
                ["ii", "V", "I"],
                ["IV", "V", "vi"],
            ]
        }

    @staticmethod
    def _load_key_signatures() -> Dict[str, Dict[str, list[str]]]:
        """Load scale degrees for different keys and modes"""
        # ToDo: Simplified - in real implementation, this would be more comprehensive
        major_scale = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
        minor_scale = ["i", "ii°", "III", "iv", "v", "VI", "VII"]

        return {
            "major": {
                "C": ["C", "Dm", "Em", "F", "G", "Am", "Bdim"],
                "G": ["G", "Am", "Bm", "C", "D", "Em", "F#dim"],
                "F": ["F", "Gm", "Am", "Bb", "C", "Dm", "Edim"],
                # ToDo: Add more keys as needed
            },
            "minor": {
                "Am": ["Am", "Bdim", "C", "Dm", "Em", "F", "G"],
                "Em": ["Em", "F#dim", "G", "Am", "Bm", "C", "D"],
                "Dm": ["Dm", "Edim", "F", "Gm", "Am", "Bb", "C"],
                # ToDo: Add more keys as needed
            }
        }

    def generate_progression_for_section(self,
                                         section: RainbowSongStructureModel,
                                         key: str,
                                         mood_tags: list[str],
                                         previous_progression: Optional[ChordProgression] = None) -> ChordProgression:
        """Generate a chord progression for a specific song section"""

        # Parse key and mode
        key_parts = key.split()
        root_key = key_parts[0]
        mode = key_parts[1] if len(key_parts) > 1 else "major"

        # Determine a section type from name
        section_type = self._classify_section(section.section_name)

        # Get appropriate chord templates
        templates = self.chord_templates.get(section_type, self.chord_templates["verse"])

        # Select template based on mood
        template = self._select_template_by_mood(templates, mood_tags)

        # Convert roman numerals to actual chords
        chords = self._roman_to_chords(template, root_key, mode)

        # Add complexity based on mood
        chords = self._add_chord_complexity(chords, mood_tags)

        # Determine bars per chord based on section duration
        bars_per_chord = self._calculate_bars_per_chord(section, len(chords))

        return ChordProgression(
            chords=chords,
            section_name=section.section_name,
            bars_per_chord=bars_per_chord,
            key=key,
            mode=mode
        )

    @staticmethod
    def _classify_section(section_name: str) -> str:
        """Classify a section type from name"""
        name_lower = section_name.lower()

        if "verse" in name_lower:
            return "verse"
        elif "chorus" in name_lower:
            return "chorus"
        elif "bridge" in name_lower:
            return "bridge"
        elif "intro" in name_lower:
            return "intro"
        elif "outro" in name_lower:
            return "outro"
        elif "interlude" in name_lower:
            return "interlude"
        else:
            return "verse"  # Default

    @staticmethod
    def _select_template_by_mood(templates: list[list[str]], mood_tags: list[str]) -> list[str]:
        """Select chord template based on mood"""
        mood_str = " ".join(mood_tags).lower()

        # Dark/mysterious moods prefer minor progressions
        if any(word in mood_str for word in ["dark", "mysterious", "haunting", "eerie", "gothic"]):
            minor_templates = [t for t in templates if any("i" in chord for chord in t)]
            if minor_templates:
                return random.choice(minor_templates)

        # Ethereal/ambient moods prefer extended chords
        if any(word in mood_str for word in ["ethereal", "otherworldly", "atmospheric"]):
            return random.choice(templates)

        # Default selection
        return random.choice(templates)

    @staticmethod
    def _roman_to_chords(roman_numerals: list[str], key: str, mode: str) -> list[Chord]:
        """Convert roman numeral progression to actual chords"""
        chords = []

        # ToDo: This is simplified - real implementation would use proper music theory
        chord_map = {
            "major": {
                "C": {"I": "C", "ii": "Dm", "iii": "Em", "IV": "F", "V": "G", "vi": "Am", "vii°": "Bdim"},
                "G": {"I": "G", "ii": "Am", "iii": "Bm", "IV": "C", "V": "D", "vi": "Em", "vii°": "F#dim"},
                "F": {"I": "F", "ii": "Gm", "iii": "Am", "IV": "Bb", "V": "C", "vi": "Dm", "vii°": "Edim"},
            },
            "minor": {
                "Am": {"i": "Am", "ii°": "Bdim", "III": "C", "iv": "Dm", "v": "Em", "VI": "F", "VII": "G"},
            }
        }

        key_chords = chord_map.get(mode, {}).get(key, {})

        for roman in roman_numerals:
            chord_name = key_chords.get(roman, "C")  # Fallback to C

            # Determine quality from a roman numeral
            if roman.islower() or "°" in roman:
                if "°" in roman:
                    quality = ChordQuality.DIMINISHED
                else:
                    quality = ChordQuality.MINOR
            else:
                quality = ChordQuality.MAJOR

            # Extract root note
            root = chord_name.replace("m", "").replace("dim", "")

            chords.append(Chord(root=root, quality=quality))

        return chords

    @staticmethod
    def _add_chord_complexity(self, chords: list[Chord], mood_tags: list[str]) -> list[Chord]:
        """Add extensions and alterations based on mood"""
        mood_str = " ".join(mood_tags).lower()

        # Experimental/avant-garde moods get more complex chords
        if any(word in mood_str for word in ["experimental", "avant-garde", "complex"]):
            for i, chord in enumerate(chords):
                if random.random() < 0.3:  # 30% chance
                    if chord.quality == ChordQuality.MAJOR:
                        chords[i].quality = ChordQuality.MAJOR7
                    elif chord.quality == ChordQuality.MINOR:
                        chords[i].quality = ChordQuality.MINOR7

        # Ethereal moods get suspended chords
        if any(word in mood_str for word in ["ethereal", "dreamy", "atmospheric"]):
            for i, chord in enumerate(chords):
                if random.random() < 0.2:  # 20% chance
                    chords[i].quality = random.choice([ChordQuality.SUSPENDED2, ChordQuality.SUSPENDED4])

        return chords

    @staticmethod
    def _calculate_bars_per_chord(section: RainbowSongStructureModel, num_chords: int) -> list[int]:
        """Calculate how many bars each chord should last"""
        # ToDo: This is simplified - real implementation would consider tempo, section length, etc.

        # Default to 1 bar per chord for now
        bars_per_chord = [1] * num_chords

        # For longer sections, extend some chords
        if section.section_description and "long" in section.section_description.lower():
            for i in range(0, len(bars_per_chord), 2):
                bars_per_chord[i] = 2

        return bars_per_chord
