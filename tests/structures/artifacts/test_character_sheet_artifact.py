import pytest
from pydantic import ValidationError

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


IMAGE_PATH = "img"


def test_character_sheet_artifact():
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
        thread_id="test-thread", sheet_content=test_character, image_path=IMAGE_PATH
    )
    assert character_sheet.thread_id == "test-thread"
    assert character_sheet.chain_artifact_type == ChainArtifactType.CHARACTER_SHEET
    assert isinstance(character_sheet.sheet_content, PulsarPalaceCharacter)
    assert character_sheet.sheet_content.thread_id == "test-thread"
    assert character_sheet.sheet_content.encounter_id == "test-encounter"


def test_character_sheet_artifact_defaults():
    # image_path is required by HtmlChainArtifactFile; but sheet_content is optional
    character_sheet = PulsarPalaceCharacterSheet(
        thread_id="test-thread", image_path=IMAGE_PATH
    )
    assert character_sheet.sheet_content is None


def test_character_sheet_artifact_with_character():
    # PulsarPalaceCharacter requires both thread_id and encounter_id
    character = PulsarPalaceCharacter(thread_id="test-thread", encounter_id="enc-1")
    character_sheet = PulsarPalaceCharacterSheet(
        thread_id="test-thread", sheet_content=character, image_path=IMAGE_PATH
    )
    assert character_sheet.sheet_content == character


def test_character_sheet_artifact_type():
    default = PulsarPalaceCharacterSheet.model_fields["chain_artifact_type"].default
    assert default == ChainArtifactType.CHARACTER_SHEET


def test_missing_fields_raises_validation_error():
    # image_path has a default value, so this should succeed
    sheet = PulsarPalaceCharacterSheet(thread_id="test-thread")
    assert sheet.thread_id == "test-thread"
    assert sheet.image_path == ""  # default value
    sheet = PulsarPalaceCharacterSheet(thread_id="test-thread")
    assert sheet.thread_id == "test-thread"
    assert sheet.image_path == ""  # default value


def test_wrong_type_raises_validation_error():
    with pytest.raises(ValidationError):
        PulsarPalaceCharacterSheet(
            thread_id="test-thread", image_path=IMAGE_PATH, sheet_content=123
        )


def test_flatten_and_for_prompt():
    test_character = PulsarPalaceCharacter(
        thread_id="thread-x",
        encounter_id="enc-x",
        background=PulsarPalaceCharacterBackground(
            rollId=2, time=1999, place="Sample Place"
        ),
        disposition=PulsarPalaceCharacterDisposition(rollId=2, disposition="Curious"),
        profession=PulsarPalaceCharacterProfession(rollId=2, profession="Sailor"),
        on_max=5,
        on_current=3,
        off_max=7,
        off_current=2,
    )

    sheet = PulsarPalaceCharacterSheet(
        thread_id="thread-x", sheet_content=test_character, image_path=IMAGE_PATH
    )
    flat = sheet.flatten()
    assert isinstance(flat, dict)
    assert "sheet_content" in flat
    assert flat["sheet_content"]["thread_id"] == "thread-x"

    prompt_text = sheet.for_prompt()
    assert "Character Sheet" in prompt_text
    assert "ON: 3/5" in prompt_text
    assert "OFF: 2/7" in prompt_text
