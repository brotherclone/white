import random
import re
from typing import Dict, List

from app.structures.concepts.pulsar_palace_room import PulsarPalaceRoom


class MusicExtractor:
    """
    Extracts musical elements (mood, BPM, key, structure) from Pulsar Palace room narratives.
    Translates gameplay into musical concepts for the Yellow Agent.
    """

    def __init__(self):
        self.mood_keywords = {
            "elegant": ["baroque", "grandiose", "sophisticated"],
            "opulent": ["baroque", "lush", "dreamy"],
            "surreal": ["dreamlike", "mysterious", "cosmic"],
            "dreamlike": ["dreamy", "mysterious", "ethereal"],
            "impossible": ["mysterious", "cosmic", "otherworldly"],
            "shifting": ["unstable", "mysterious", "hypnotic"],
            "mirrored": ["reflective", "mysterious", "ethereal"],
            "psychedelic": ["trippy", "mind-melting", "cosmic"],
            "alien": ["otherworldly", "cosmic", "eerie"],
            "warped": ["distorted", "unsettling", "disorienting"],
            "mutant": ["grotesque", "weird", "disturbing"],
            "threatening": ["dark", "ominous", "intense"],
            "unstable": ["chaotic", "anxious", "disorienting"],
            "corrupted": ["dark", "disturbing", "ominous"],
            "pulsing": ["rhythmic", "hypnotic", "intense"],
            "crystalline": ["bright", "shimmering", "ethereal"],
            "baroque": ["baroque", "ornate", "classical"],
            "frozen": ["cold", "still", "eerie"],
            "weeping": ["melancholic", "sad", "emotional"],
            "violent": ["aggressive", "chaotic", "intense"],
            "cosmic": ["cosmic", "vast", "transcendent"],
        }

        self.atmosphere_to_bpm = {
            "elegant": (70, 90),
            "surreal": (85, 110),
            "weird": (100, 140),
            "dangerous": (120, 160),
        }

        self.atmosphere_to_keys = {
            "elegant": ["C major", "G major", "D major", "A major"],
            "surreal": ["E minor", "A minor", "D minor", "F# minor"],
            "weird": ["Eb minor", "Bb minor", "F minor", "C# minor"],
            "dangerous": ["E major", "B major", "F# major", "C# major"],
        }

        self.room_type_to_structure = {
            "entrance": ["arrival", "threshold", "first_steps"],
            "social": ["gathering", "interaction", "social_dance", "tension"],
            "service": ["labor", "routine", "disruption"],
            "private": ["intimacy", "revelation", "secrets"],
            "mystical": ["ritual", "vision", "transformation"],
            "transitional": ["passage", "journey", "crossing"],
        }

        self.genres_base = [
            "electronic",
            "ambient",
            "experimental",
            "kosmische",
        ]

        self.genres_by_atmosphere = {
            "elegant": ["baroque pop", "art pop", "chamber pop"],
            "surreal": ["psychedelic", "dream pop", "new age"],
            "weird": ["noise", "industrial", "avant-garde"],
            "dangerous": ["dark ambient", "drone", "post-industrial"],
        }

    def extract_music_concept(
        self, room: PulsarPalaceRoom, encounter_narrative: str
    ) -> Dict:
        """Extract complete musical concept from room and narrative"""

        atmosphere_type = room.atmosphere.split(" - ")[0]

        mood = self._extract_mood(room, encounter_narrative)

        bpm = self._calculate_bpm(atmosphere_type, encounter_narrative)

        key = self._determine_key(atmosphere_type, mood)

        structure = self._generate_structure(room, encounter_narrative)

        genres = self._determine_genres(atmosphere_type, mood)

        concept = self._generate_concept(room, encounter_narrative)

        return {
            "bpm": bpm,
            "tempo": "4/4",
            "key": key,
            "rainbow_color": "Y",
            "title": room.name,
            "mood": mood,
            "genres": genres,
            "structure": structure,
            "concept": concept,
            "narrative_source": encounter_narrative,
        }

    def _extract_mood(self, room: PulsarPalaceRoom, narrative: str) -> List[str]:
        """Extract mood descriptors from room and narrative"""

        moods = set()

        text = f"{room.description} {room.atmosphere} {narrative}".lower()

        for keyword, mood_list in self.mood_keywords.items():
            if keyword in text:
                moods.update(mood_list)

        if "party" in text and "dance" in text:
            moods.add("playful")
        if "scream" in text or "terror" in text:
            moods.add("terrifying")
        if "weep" in text or "tear" in text:
            moods.add("melancholic")

        moods.add("mysterious")

        return list(moods)[:5]

    def _calculate_bpm(self, atmosphere_type: str, narrative: str) -> float:
        """Calculate BPM based on atmosphere and narrative tension"""

        base_range = self.atmosphere_to_bpm.get(atmosphere_type, (90, 120))

        tension_keywords = [
            "violent",
            "chaos",
            "scream",
            "run",
            "attack",
            "panic",
        ]
        calm_keywords = ["still", "frozen", "quiet", "gentle", "slow"]

        tension_score = sum(1 for kw in tension_keywords if kw in narrative.lower())
        calm_score = sum(1 for kw in calm_keywords if kw in narrative.lower())

        adjustment = (tension_score - calm_score) * 5

        base_bpm = random.uniform(*base_range)
        final_bpm = max(60, min(180, base_bpm + adjustment))

        return round(final_bpm, 2)

    def _determine_key(self, atmosphere_type: str, mood: List[str]) -> str:
        """Determine musical key based on atmosphere and mood"""

        keys = self.atmosphere_to_keys.get(atmosphere_type, ["C major"])

        if "dark" in mood or "ominous" in mood or "disturbing" in mood:
            minor_keys = [k for k in keys if "minor" in k]
            if minor_keys:
                return random.choice(minor_keys)

        return random.choice(keys)

    def _generate_structure(self, room: PulsarPalaceRoom, narrative: str) -> List[Dict]:
        """Generate song structure sections from narrative"""

        base_sections = self.room_type_to_structure.get(
            room.room_type, ["intro", "development", "conclusion"]
        )

        sentences = [s.strip() for s in narrative.split(".") if s.strip()]

        structure = []
        time_cursor = 0.0
        section_duration = 30.0

        for i, section_name in enumerate(base_sections):
            if i < len(sentences):
                description = sentences[i]
            else:
                description = room.narrative_beat

            start_time = f"[{self._format_time(time_cursor)}]"
            time_cursor += section_duration
            end_time = f"[{self._format_time(time_cursor)}]"

            structure.append(
                {
                    "section_name": section_name.replace("_", " ").title(),
                    "start_time": start_time,
                    "end_time": end_time,
                    "description": description,
                }
            )

        return structure

    def _format_time(self, seconds: float) -> str:
        """Format time as MM:SS.mmm"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"

    def _determine_genres(self, atmosphere_type: str, mood: List[str]) -> List[str]:
        """Determine musical genres based on atmosphere and mood"""

        genres = self.genres_base.copy()

        atmosphere_genres = self.genres_by_atmosphere.get(atmosphere_type, [])
        genres.extend(
            random.sample(atmosphere_genres, k=min(2, len(atmosphere_genres)))
        )

        if "baroque" in mood:
            genres.append("baroque pop")
        if "playful" in mood:
            genres.append("novelty")
        if "terrifying" in mood or "dark" in mood:
            genres.append("dark ambient")

        return list(set(genres))[:6]

    def _generate_concept(self, room: PulsarPalaceRoom, narrative: str) -> str:
        """Generate a concept statement from room and narrative"""

        concept_parts = [
            f"In the {room.name}, the party encounters {room.atmosphere.split(' - ')[1]} spaces.",
        ]

        if room.inhabitants:
            concept_parts.append(
                f"They interact with {', '.join(room.inhabitants[:2])}."
            )

        key_phrase = self._extract_key_phrase(narrative)
        if key_phrase:
            concept_parts.append(key_phrase)

        concept_parts.append(
            "The pulsar's influence transforms the experience into sound."
        )

        return " ".join(concept_parts)

    def _extract_key_phrase(self, narrative: str) -> str:
        """Extract a key dramatic phrase from the narrative"""

        dramatic_patterns = [
            r"(chaos erupts.*?\.)",
            r"(reality.*?\.)",
            r"(the pulsar.*?\.)",
            r"(something.*?breaks.*?\.)",
            r"(time.*?\.)",
        ]

        for pattern in dramatic_patterns:
            match = re.search(pattern, narrative, re.IGNORECASE)
            if match:
                return match.group(1)

        return ""
