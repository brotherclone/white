import random
import re
from typing import List

from app.structures.concepts.pulsar_palace_room import PulsarPalaceRoom
from app.structures.manifests.song_proposal import SongProposalIteration


class MusicExtractor:
    """
    Extracts musical elements (mood, BPM, key, genres, concept) from Pulsar Palace room narratives.
    Translates gameplay into SongProposalIteration objects for the Yellow Agent.
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

    def extract_song_proposal(
        self, room: PulsarPalaceRoom, encounter_narrative: str
    ) -> SongProposalIteration:
        """Extract a song proposal from room and narrative"""

        atmosphere_type = room.atmosphere.split(" - ")[0]

        mood = self._extract_mood(room, encounter_narrative)

        bpm = self._calculate_bpm(atmosphere_type, encounter_narrative)

        key = self._determine_key(atmosphere_type, mood)

        genres = self._determine_genres(atmosphere_type, mood)

        concept = self._generate_concept(room, encounter_narrative)

        iteration_id = self._generate_iteration_id(room.name, room.room_id)

        return SongProposalIteration(
            iteration_id=iteration_id,
            bpm=bpm,
            tempo="4/4",
            key=key,
            rainbow_color="Y",
            title=room.name,
            mood=mood,
            genres=genres,
            concept=concept,
        )

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

    def _calculate_bpm(self, atmosphere_type: str, narrative: str) -> int:
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
        final_bpm = max(40, min(200, base_bpm + adjustment))

        return int(round(final_bpm))

    def _determine_key(self, atmosphere_type: str, mood: List[str]) -> str:
        """Determine musical key based on atmosphere and mood"""

        keys = self.atmosphere_to_keys.get(atmosphere_type, ["C major"])

        if "dark" in mood or "ominous" in mood or "disturbing" in mood:
            minor_keys = [k for k in keys if "minor" in k]
            if minor_keys:
                return random.choice(minor_keys)

        return random.choice(keys)

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
        """Generate a substantive concept statement from room and narrative (min 100 chars)"""

        atmosphere_descriptor = room.atmosphere.split(" - ")[1]

        concept_parts = [
            f"In the {room.name}, the party encounters {atmosphere_descriptor} spaces where the boundaries of reality become permeable."
        ]

        if room.inhabitants:
            concept_parts.append(
                f"The presence of {', '.join(room.inhabitants[:2])} creates a tension between the organic and the cosmic, the temporal and the eternal."
            )

        key_phrase = self._extract_key_phrase(narrative)
        if key_phrase:
            concept_parts.append(key_phrase)

        concept_parts.append(
            f"The pulsar's rhythmic influence—flickering between red and green to create the illusion of yellow—transforms this narrative of {room.room_type} spaces into sonic architecture."
        )

        concept_parts.append(
            "Each character action resonates with archetypal patterns: the disposition shapes emotional timbre, profession defines rhythmic approach, and temporal origin creates harmonic context."
        )

        concept = " ".join(concept_parts)

        if len(concept) < 100:
            concept += " This transmutation from gameplay to music embodies the Palace's nature as a liminal space where narrative becomes melody, action becomes rhythm, and experience crystallizes into sound."

        return concept

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

    def _generate_iteration_id(self, room_name: str, room_id: str) -> str:
        """Generate iteration_id in format 'descriptive_name_v#'"""

        name_part = room_name.lower().replace(" ", "_").replace("-", "_")

        name_part = re.sub(r"[^a-z0-9_]", "", name_part)

        number_match = re.search(r"(\d+)$", room_id)
        version = number_match.group(1) if number_match else "1"

        return f"pulsar_palace_{name_part}_v{version}"
