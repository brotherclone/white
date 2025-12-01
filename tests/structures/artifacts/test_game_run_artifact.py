from app.structures.artifacts.pulsar_palace_encounter_artifact import (
    PulsarPalaceEncounterArtifact,
)


def test_game_run_artifact():
    game_run = PulsarPalaceEncounterArtifact(thread_id="test-thread")
    assert game_run.thread_id == "test-thread"
    assert game_run.encounter_id is None
