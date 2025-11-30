from abc import ABC

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.concepts.pulsar_palace_character import PulsarPalaceCharacter


class PulsarPalaceCharacterSheet(ChainArtifact, ABC):

    character: PulsarPalaceCharacter

    def __init__(self, **data):
        super().__init__(**data)

    def save_file(self):
        """Return a little GI-Joe-esque profile care with a portrait"""
        pass
