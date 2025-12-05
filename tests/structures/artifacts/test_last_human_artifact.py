from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.artifacts.last_human_artifact import LastHumanArtifact


def test_inheritance():
    assert issubclass(LastHumanArtifact, ChainArtifact)


# ToDo: Add additional tests for LastHumanArtifact fields and validation logic
