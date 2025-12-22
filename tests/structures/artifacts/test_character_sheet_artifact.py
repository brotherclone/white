from app.structures.artifacts.pulsar_palace_character_sheet import (
    PulsarPalaceCharacterSheet,
)
from app.structures.concepts.pulsar_palace_character import (
    PulsarPalaceCharacter,
    PulsarPalaceCharacterBackground,
    PulsarPalaceCharacterDisposition,
    PulsarPalaceCharacterProfession,
)
from app.structures.enums.chain_artifact_type import ChainArtifactType

# ToDo: You call this a test? :)


def test_character_sheet_artifact():
    # Create a test character with all required fields
    test_character = PulsarPalaceCharacter(
        thread_id="test-thread",
        encounter_id="test-encounter",
        background=PulsarPalaceCharacterBackground(
            rollId=1, time=2121, place="Test City"
        ),
        disposition=PulsarPalaceCharacterDisposition(
            rollId=1, disposition="Test Disposition"
        ),
        profession=PulsarPalaceCharacterProfession(
            rollId=1, profession="Test Profession"
        ),
        on_max=10,
        on_current=10,
        off_max=10,
        off_current=10,
    )

    character_sheet = PulsarPalaceCharacterSheet(
        thread_id="test-thread", sheet_content=test_character
    )
    assert character_sheet.thread_id == "test-thread"
    assert character_sheet.chain_artifact_type == ChainArtifactType.CHARACTER_SHEET
    assert isinstance(character_sheet.sheet_content, PulsarPalaceCharacter)
    assert character_sheet.sheet_content.thread_id == "test-thread"
    assert character_sheet.sheet_content.encounter_id == "test-encounter"
