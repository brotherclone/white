import random
from typing import List

from app.structures.concepts.pulsar_palace_room import PulsarPalaceRoom


class MarkovRoomGenerator:
    """
    Generates Pulsar Palace rooms using Markov chains.
    Aesthetic: Marienbad (elegant, surreal, baroque) × Ultraviolet Grasslands (weird, psychedelic, dangerous)
    """

    def __init__(self):
        self.room_types = {
            "entrance": ["foyer", "atrium", "vestibule", "portal chamber"],
            "social": ["ballroom", "salon", "gallery", "conservatory"],
            "service": ["kitchen", "pantry", "laundry", "servant quarters"],
            "private": ["guest quarters", "study", "library", "bedchamber"],
            "mystical": ["observatory", "ritual chamber", "meditation hall", "shrine"],
            "transitional": ["corridor", "stairwell", "bridge", "passageway"],
        }

        self.atmosphere_chains = {
            "elegant": ["opulent", "baroque", "gilded", "crystalline"],
            "surreal": ["dreamlike", "impossible", "shifting", "mirrored"],
            "weird": ["psychedelic", "alien", "warped", "mutant"],
            "dangerous": ["threatening", "unstable", "corrupted", "pulsing"],
        }

        self.features_vocab = [
            "red and green pulsing lights",
            "baroque mirror arrangements",
            "impossible geometric statues",
            "levitating furniture",
            "walls that breathe",
            "floors of liquid crystal",
            "cascading ribbons of light",
            "time-frozen servants",
            "singing fountains",
            "mutant topiary",
            "telepathic paintings",
            "chromatic shadows",
            "gravity-defying chandeliers",
            "doors that exist only when watched",
            "windows showing other times",
        ]

        self.inhabitants_vocab = [
            "Cursed valets in perfect stillness",
            "Sirens weeping yellow tears",
            "Heralds with crystalline trumpets",
            "Nobles trapped mid-gesture",
            "Large childlike guardians",
            "Telepathic scholars",
            "Ribbon-entangled servants",
            "Time-lost guests",
            "Psychedelic gardeners",
            "Mirror-dwelling entities",
        ]

    def generate_room(self, room_number: int = 1) -> PulsarPalaceRoom:
        """Generate a single room using Markov-style chaining"""

        room_category = random.choice(list(self.room_types.keys()))
        room_name = random.choice(self.room_types[room_category])

        atmosphere_type = random.choice(list(self.atmosphere_chains.keys()))
        atmosphere = random.choice(self.atmosphere_chains[atmosphere_type])

        features = random.sample(self.features_vocab, k=random.randint(2, 4))

        inhabitants = random.sample(self.inhabitants_vocab, k=random.randint(0, 2))

        description = self._generate_description(
            room_name, atmosphere, features, inhabitants
        )

        narrative_beat = self._generate_narrative_beat(room_category, atmosphere_type)

        exits = self._generate_exits()

        return PulsarPalaceRoom(
            room_id=f"room_{room_number:03d}",
            name=room_name.title(),
            description=description,
            atmosphere=f"{atmosphere_type} - {atmosphere}",
            room_type=room_category,
            exits=exits,
            inhabitants=inhabitants,
            features=features,
            narrative_beat=narrative_beat,
        )

    def _generate_description(
        self,
        room_name: str,
        atmosphere: str,
        features: List[str],
        inhabitants: List[str],
    ) -> str:
        """Create a descriptive text for the room"""

        description_parts = [
            f"A {atmosphere} {room_name} stretches before the party.",
            f"The space is dominated by {features[0]}.",
        ]

        if len(features) > 1:
            description_parts.append(
                f"Near the edges, {features[1]} create an unsettling ambiance."
            )

        if inhabitants:
            description_parts.append(f"{inhabitants[0]} occupy the chamber.")

        if len(features) > 2:
            description_parts.append(f"The party notices {features[2]}.")

        return " ".join(description_parts)

    def _generate_narrative_beat(self, room_category: str, atmosphere_type: str) -> str:
        """Generate a narrative beat based on room type and atmosphere"""

        beats = {
            (
                "entrance",
                "elegant",
            ): "The party arrives with ceremonial grandeur, but something is subtly wrong.",
            (
                "entrance",
                "surreal",
            ): "Reality bends as the party crosses the threshold.",
            (
                "entrance",
                "weird",
            ): "The party's arrival triggers psychedelic distortions in space.",
            (
                "entrance",
                "dangerous",
            ): "The entrance pulses with threatening energy as the party enters.",
            (
                "social",
                "elegant",
            ): "Frozen guests perform an eternal dance of aristocratic decay.",
            (
                "social",
                "surreal",
            ): "Time layers collapse as past and present guests occupy the same space.",
            (
                "social",
                "weird",
            ): "Guests undergo grotesque transformations while maintaining perfect manners.",
            ("social", "dangerous"): "The social gathering conceals predatory intent.",
            (
                "service",
                "elegant",
            ): "Servants continue their duties despite cosmic corruption.",
            (
                "service",
                "surreal",
            ): "The boundaries between server and served dissolve.",
            ("service", "weird"): "Mundane tasks become alien rituals.",
            ("service", "dangerous"): "Service personnel have become hostile entities.",
            ("private", "elegant"): "Personal spaces reveal the palace's former glory.",
            ("private", "surreal"): "Intimate chambers contain impossible memories.",
            ("private", "weird"): "Privacy itself becomes a mutant concept.",
            (
                "private",
                "dangerous",
            ): "What sleeps in private quarters should not wake.",
            ("mystical", "elegant"): "Sacred spaces maintain baroque perfection.",
            (
                "mystical",
                "surreal",
            ): "Divination reveals multiple contradictory truths.",
            ("mystical", "weird"): "The mystical becomes viscerally, dangerously real.",
            (
                "mystical",
                "dangerous",
            ): "Powers beyond comprehension stir in ritual spaces.",
            (
                "transitional",
                "elegant",
            ): "Passages maintain their architectural dignity while defying physics.",
            (
                "transitional",
                "surreal",
            ): "Corridors loop through spaces that shouldn't connect.",
            (
                "transitional",
                "weird",
            ): "The journey itself becomes the destination, mutated.",
            (
                "transitional",
                "dangerous",
            ): "Movement through space attracts unwanted attention.",
        }

        key = (room_category, atmosphere_type)
        return beats.get(
            key, "The party enters a space that defies easy categorization."
        )

    def _generate_exits(self) -> List[str]:
        """Generate potential exits"""

        exit_types = [
            "north",
            "south",
            "east",
            "west",
            "up",
            "down",
            "through the mirror",
            "behind the tapestry",
            "across the pulsar threshold",
        ]

        num_exits = random.randint(1, 3)
        return random.sample(exit_types, k=num_exits)
