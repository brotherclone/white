import random
from typing import List

from app.structures.concepts.pulsar_palace_character import (
    PulsarPalaceCharacter,
)


class CharacterActionGenerator:
    """
    Generates wild character actions based on disposition + profession + background combinations.
    Creates narrative beats for Pulsar Palace encounters.
    """

    def __init__(self):
        self.disposition_actions = {
            "Angry": [
                "lashes out violently at {target}",
                "smashes {object} in frustration",
                "screams accusations at {target}",
                "attempts to destroy {object}",
            ],
            "Curious": [
                "obsessively examines {object}",
                "asks endless questions about {target}",
                "touches {object} despite warnings",
                "peers too closely at {target}",
            ],
            "Misguided": [
                "confidently does the exact wrong thing with {object}",
                "offers terrible advice to {target}",
                "misinterprets {object} completely",
                "leads the party astray toward {target}",
            ],
            "Clumsy": [
                "accidentally breaks {object}",
                "trips into {target}",
                "knocks over {object}",
                "fumbles {object} at a critical moment",
            ],
            "Cursed": [
                "spreads corruption to {object}",
                "causes {target} to suffer misfortune",
                "triggers ominous events near {object}",
                "cannot escape doom related to {target}",
            ],
            "Sick": [
                "contaminates {object} with mysterious illness",
                "weakly reaches for {target}",
                "coughs something unnatural onto {object}",
                "collapses near {target}",
            ],
            "Vengeful": [
                "plots revenge against {target}",
                "sabotages {object} deliberately",
                "confronts {target} with past wrongs",
                "uses {object} as a weapon",
            ],
            "Crazed": [
                "babbles madly about {object}",
                "performs ritual actions with {target}",
                "sees things in {object} that aren't there",
                "becomes fixated on {target} beyond reason",
            ],
        }

        self.profession_modifiers = {
            "Doctor": [
                "attempts to diagnose",
                "applies medical knowledge to",
                "performs surgery on",
                "prescribes treatment for",
            ],
            "Sailor": [
                "navigates using",
                "ties nautical knots with",
                "reads weather patterns in",
                "searches for water near",
            ],
            "Breeder": [
                "examines bloodlines of",
                "attempts to domesticate",
                "studies reproduction of",
                "cultivates relationship with",
            ],
            "Detective": [
                "investigates clues about",
                "interrogates",
                "deduces secrets from",
                "follows trail of",
            ],
            "Janitor": [
                "compulsively cleans",
                "maintains order around",
                "sweeps away evidence of",
                "organizes chaos involving",
            ],
            "Spy": [
                "covertly observes",
                "steals information from",
                "disguises themselves using",
                "reports intelligence about",
            ],
            "Librarian": [
                "catalogs and categorizes",
                "shushes loudly near",
                "researches historical context of",
                "preserves knowledge of",
            ],
            "Inventor": [
                "jury-rigs a device from",
                "theorizes about the mechanism of",
                "disassembles",
                "creates something new with",
            ],
            "Tax Collector": [
                "assesses the value of",
                "demands payment from",
                "audits the legitimacy of",
                "confiscates",
            ],
            "Partisan": [
                "fights for control of",
                "rallies others around",
                "defends ideologically",
                "attacks enemies near",
            ],
        }

        self.background_contexts = {
            "futuristic": [
                "using advanced technology",
                "with cybernetic precision",
                "through digital interfaces",
                "via neural link",
            ],
            "vintage": [
                "with old-world methods",
                "using period-appropriate tools",
                "in the style of their era",
                "honoring tradition",
            ],
        }

        self.objects = [
            "the pulsing lights",
            "a baroque mirror",
            "a frozen servant",
            "the crystalline floor",
            "a levitating object",
            "the red-green ribbons",
            "a time-locked door",
            "the singing fountain",
            "a telepathic painting",
            "the breathing walls",
        ]

        self.targets = [
            "another character",
            "a palace inhabitant",
            "the party leader",
            "a mysterious figure",
            "their own reflection",
            "the shadows",
            "an apparition",
            "the herald",
            "the sirens",
            "the pulsar itself",
        ]

    def generate_action(self, character: PulsarPalaceCharacter) -> str:
        """Generate a wild action based on character's traits"""

        disposition = character.disposition.disposition
        profession = character.profession.profession
        time_period = character.background.time

        disposition_template = random.choice(
            self.disposition_actions.get(disposition, ["acts mysteriously"])
        )

        profession_modifier = random.choice(
            self.profession_modifiers.get(profession, ["interacts with"])
        )

        object_or_target = random.choice(
            [random.choice(self.objects), random.choice(self.targets)]
        )

        context = self._get_temporal_context(time_period)

        action = disposition_template.format(
            target=object_or_target, object=object_or_target
        )

        full_action = f"The {disposition} {profession} {profession_modifier} {object_or_target} {context}, and {action}"

        return full_action

    def _get_temporal_context(self, time_period: int) -> str:
        """Add temporal flavor based on character's time period"""

        if time_period < 1900:
            return random.choice(self.background_contexts["vintage"])
        elif time_period > 2000:
            return random.choice(self.background_contexts["futuristic"])
        else:
            return random.choice(
                self.background_contexts["vintage"]
                + self.background_contexts["futuristic"]
            )

    def generate_encounter_narrative(
        self,
        room_description: str,
        characters: List[PulsarPalaceCharacter],
        return_character_updates: bool = False,
    ):
        """
        Generate a full encounter narrative from room and characters

        Args:
            room_description: Description of the room
            characters: List of characters in the encounter
            return_character_updates: If True, returns (narrative, updated_characters)
                                     If False, returns just narrative (default)

        Returns:
            str or tuple: Narrative string, or (narrative, updated_characters) if return_character_updates=True
        """

        narrative_parts = [f"The party enters the space. {room_description}"]

        # Create a copy of characters for potential mutations
        updated_characters = characters if not return_character_updates else []

        for character in characters:
            action = self.generate_action(character)
            narrative_parts.append(action)

            # Apply character state mutations if requested
            if return_character_updates:
                mutated_char = self._apply_encounter_mutations(character)
                updated_characters.append(mutated_char)

        outcome = self._generate_outcome(len(characters))
        narrative_parts.append(outcome)

        narrative = " ".join(narrative_parts)

        if return_character_updates:
            return narrative, updated_characters
        return narrative

    def _apply_encounter_mutations(
        self, character: PulsarPalaceCharacter
    ) -> PulsarPalaceCharacter:
        """
        Apply stat mutations to a character based on encounter outcomes.
        Characters can gain or lose ON/OFF based on their disposition and actions.

        Returns a modified copy of the character.
        """
        # Create a copy to avoid mutating the original
        import copy

        mutated = copy.deepcopy(character)

        disposition = character.disposition.disposition

        # Disposition-based stat changes
        disposition_effects = {
            "Angry": (-1, 2),  # Lose ON, gain OFF (anger burns energy, builds tension)
            "Curious": (
                1,
                -1,
            ),  # Gain ON, lose OFF (curiosity energizes, reduces negativity)
            "Misguided": (
                -2,
                1,
            ),  # Lose ON, gain OFF (mistakes drain energy, cause frustration)
            "Clumsy": (
                -1,
                1,
            ),  # Lose ON, gain OFF (accidents tire, create problems)
            "Cursed": (
                -2,
                3,
            ),  # Lose ON, gain OFF (curse drains vitality, amplifies darkness)
            "Sick": (
                -3,
                2,
            ),  # Lose ON, gain OFF (illness weakens, creates suffering)
            "Vengeful": (
                0,
                2,
            ),  # Neutral ON, gain OFF (vengeance feeds darkness)
            "Crazed": (
                -1,
                -1,
            ),  # Lose both (madness is chaotic and unstable)
        }

        on_change, off_change = disposition_effects.get(disposition, (0, 0))

        # Apply random variance (-1 to +1)
        on_change += random.randint(-1, 1)
        off_change += random.randint(-1, 1)

        # Update current stats (can't go below 0 or above max + 10)
        mutated.on_current = max(
            0, min(mutated.on_current + on_change, mutated.on_max + 10)
        )
        mutated.off_current = max(
            0, min(mutated.off_current + off_change, mutated.off_max + 10)
        )

        return mutated

    def _generate_outcome(self, num_characters: int) -> str:
        """Generate an outcome based on the chaos level"""

        outcomes_low = [
            "The situation resolves with minimal incident.",
            "The party moves forward, slightly shaken.",
            "An uneasy calm settles over the room.",
        ]

        outcomes_medium = [
            "Reality ripples and the pulsar throbs more intensely.",
            "Time briefly stutters, leaving everyone disoriented.",
            "The red and green lights pulse in warning.",
        ]

        outcomes_high = [
            "Chaos erupts as the palace itself responds to the disturbance.",
            "The fabric of space tears slightly, revealing yellow void beyond.",
            "The pulsar's influence intensifies dramatically, pulling at consciousness.",
            "Something fundamental breaks, and the way forward becomes uncertain.",
        ]

        if num_characters <= 1:
            return random.choice(outcomes_low)
        elif num_characters <= 3:
            return random.choice(outcomes_medium)
        else:
            return random.choice(outcomes_high)
