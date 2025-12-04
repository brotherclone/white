from app.structures.artifacts.pulsar_palace_character_sheet import (
    PulsarPalaceCharacterSheet,
)
from app.structures.enums.chain_artifact_type import ChainArtifactType


def test_character_sheet_artifact():
    character_sheet = PulsarPalaceCharacterSheet(thread_id="test-thread")
    assert character_sheet.thread_id == "test-thread"
    assert character_sheet.chain_artifact_type == ChainArtifactType.CHARACTER_SHEET
