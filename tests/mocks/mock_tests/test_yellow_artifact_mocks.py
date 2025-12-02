from pathlib import Path

import yaml

from app.structures.artifacts.pulsar_palace_character_sheet import (
    PulsarPalaceCharacterSheet,
)
from app.structures.artifacts.pulsar_palace_encounter_artifact import (
    PulsarPalaceEncounterArtifact,
)


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
    assert character_sheet.sheet_content == "# Yellow Character Sheet One Mock"
    assert character_sheet.thread_id == "mock_thread_001"


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
