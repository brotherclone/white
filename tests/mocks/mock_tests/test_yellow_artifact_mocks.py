from pathlib import Path

import yaml

from app.structures.artifacts.pulsar_palace_character_sheet import (
    PulsarPalaceCharacterSheet,
)
from app.structures.artifacts.pulsar_palace_encounter_artifact import (
    PulsarPalaceEncounterArtifact,
)
from app.structures.concepts.pulsar_palace_character import PulsarPalaceCharacter


def test_character_sheets_mocks():
    path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "mocks"
        / "yellow_character_sheet_one_mock.yml"
    )
    with path.open("r") as f:
        data = yaml.safe_load(f)
    character_sheet = PulsarPalaceCharacterSheet(**data)

    # Check that sheet_content is a PulsarPalaceCharacter instance
    assert isinstance(character_sheet.sheet_content, PulsarPalaceCharacter)
    assert character_sheet.thread_id == "mock_thread_001"

    # Check nested character properties
    char = character_sheet.sheet_content
    assert char.thread_id == "mock_thread_001"
    assert char.encounter_id == "mock_encounter_001"
    assert char.background.place == "London"
    assert char.background.time == 1973
    assert char.disposition.disposition == "Curious"
    assert char.profession.profession == "Detective"
    assert char.on_max == 15
    assert char.on_current == 15
    assert char.off_max == 12
    assert char.off_current == 12


def test_game_run_mocks():
    path = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "mocks"
        / "yellow_game_run_mock.yml"
    )
    with path.open("r") as f:
        data = yaml.safe_load(f)
    game_run = PulsarPalaceEncounterArtifact(**data)
    assert game_run.encounter_id == "mock_001"
