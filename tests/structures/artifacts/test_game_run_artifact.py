import pytest
from pydantic import ValidationError

from app.structures.artifacts.pulsar_palace_encounter_artifact import (
    PulsarPalaceEncounterArtifact,
)


def test_game_run_artifact():
    game_run = PulsarPalaceEncounterArtifact(thread_id="test-thread")
    assert game_run.thread_id == "test-thread"
    assert game_run.encounter_id is None


def test_missing_fields_raises_validation_error():
    # Test that explicitly passing an empty characters list violates min_length=1
    with pytest.raises(ValidationError):
        PulsarPalaceEncounterArtifact(thread_id="test-thread", characters=[])
