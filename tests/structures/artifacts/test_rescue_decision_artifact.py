from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.artifacts.rescue_decision_artifact import RescueDecisionArtifact


def test_inheritance():
    assert issubclass(RescueDecisionArtifact, ChainArtifact)


# ToDo: Add additional tests for RescueDecisionArtifact fields and validation logic
